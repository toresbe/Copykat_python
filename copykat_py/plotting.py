"""Heatmap plotting for CNA results, mirroring heatmap.3.R visualizations."""

import logging
import re
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypedDict, cast

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

matplotlib.use("Agg")
import fastcluster
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from copykat_py._logging import with_default_progress_output
from copykat_py._types import DistanceMetric, FloatArray, IntArray, LinkageMatrix
from copykat_py.baseline import _collapse_repeated_features, _ward_linkage
from copykat_py.genomic_coordinates import chromosome_label
from copykat_py.metadata_colors import ContinuousAnnotation, continuous_annotation, is_continuous
from copykat_py.metadata_labels import annotation_title, is_prediction_column, legend_label, warning_caption
from copykat_py.metadata_layout import annotation_layout
from copykat_py.metadata_legend import draw_metadata_legends

logger = logging.getLogger(__name__)


@contextmanager
def _thread_limited_numeric_ops(max_threads: int = 1) -> Iterator[None]:
    """Limit BLAS/OpenMP thread fan-out during plotting-time clustering."""
    try:
        with (
            threadpool_limits(limits=max_threads, user_api="blas"),
            threadpool_limits(limits=max_threads, user_api="openmp"),
        ):
            yield
    except ValueError:
        with threadpool_limits(limits=max_threads):
            yield


def _build_plot_embedding(mat: FloatArray, random_state: int = 1234) -> npt.NDArray[np.float32]:
    """Return a float32 embedding optimized for large-cell heatmap ordering."""
    data = np.asarray(mat.T, dtype=np.float32, order="C")
    n_cells, n_features = data.shape
    if n_cells < 12000 or n_features <= 64:
        return data

    n_components = min(48, n_features - 1)
    if n_components < 8:
        return data

    with _thread_limited_numeric_ops(1):
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(data)
    return np.asarray(reduced, dtype=np.float32, order="C")


def _simple_cell_order(mat: FloatArray, predictions: Mapping[str, str] | None = None) -> IntArray:
    """Cheap fallback ordering used when clustering is unavailable or unsafe."""
    if predictions is not None:
        pred_list = list(predictions.values())
        pred_rank = np.array(
            [0 if "aneuploid" in str(p) else 1 if "diploid" in str(p) else 2 for p in pred_list],
            dtype=np.int16,
        )
        cna_magnitude = np.sum(np.abs(mat), axis=0)
        return np.lexsort((-cna_magnitude, pred_rank))

    cna_magnitude = np.sum(np.abs(mat), axis=0)
    return np.argsort(cna_magnitude)[::-1]


def _compute_distance(
    mat: FloatArray,
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    n_cores: int = 1,
) -> FloatArray:
    """Compute distance matrix for cells.

    Parameters
    ----------
    mat : np.ndarray, shape (n_bins, n_cells)
        CNA matrix.
    distance : DistanceMetric
        "euclidean", "pearson", or "spearman".

    Returns
    -------
    dist : np.ndarray
        Condensed distance matrix.
    """
    distance = DistanceMetric(distance)
    if distance == DistanceMetric.EUCLIDEAN:
        return pdist(mat.T, metric="euclidean")
    elif distance == DistanceMetric.PEARSON:
        corr = np.corrcoef(mat.T)
        corr = np.clip(corr, -1, 1)
        return pdist(1 - corr)
    elif distance == DistanceMetric.SPEARMAN:
        from scipy.stats import spearmanr

        corr, _ = spearmanr(mat, axis=0)
        corr = np.clip(corr, -1, 1)
        return pdist(1 - corr)
    else:
        return pdist(mat.T, metric=cast(Any, distance))  # validated by scipy at runtime


def _safe_linkage(
    mat: FloatArray,
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    method: str = "ward",
    n_cores: int = 1,
    max_cells: int = 65536,
) -> LinkageMatrix:
    """Compute linkage with fastcluster-first execution.

    For Ward + Euclidean linkage, keep the full matrix on fastcluster
    regardless of cell count so plotting matches the main clustering path.
    The ``max_cells`` argument is retained for compatibility.
    """
    if distance == DistanceMetric.EUCLIDEAN and method.startswith("ward"):
        data = mat.T
        collapsed = _collapse_repeated_features(data)
        return _ward_linkage(data if collapsed is None else collapsed, n_cores=n_cores)[0]

    dist = _compute_distance(mat, distance, n_cores)
    return fastcluster.linkage(dist, method=method)


class _BlockLayout(TypedDict):
    cell_order: IntArray
    labels: IntArray
    cluster_order: list[int]
    cluster_sizes: IntArray
    centroid_linkage: LinkageMatrix | None


def _clustered_block_layout(mat: FloatArray, n_clusters: int = 128, random_state: int = 1234) -> _BlockLayout:
    """Fast cell ordering plus a cluster-level dendrogram layout."""
    n_cells = mat.shape[1]
    if n_cells <= 1:
        return {
            "cell_order": np.arange(n_cells, dtype=int),
            "labels": np.zeros(n_cells, dtype=int),
            "cluster_order": [0],
            "cluster_sizes": np.array([n_cells], dtype=int),
            "centroid_linkage": None,
        }

    n_clusters = max(2, min(int(n_clusters), n_cells))
    data = _build_plot_embedding(mat, random_state=random_state)

    if n_cells > 4000:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(2048, n_cells),
            n_init=1 if n_cells >= 20000 else 3,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

    with _thread_limited_numeric_ops(1):
        labels = model.fit_predict(data)
    centroids = model.cluster_centers_
    present_clusters = sorted(np.unique(labels).tolist())

    if len(present_clusters) > 1:
        centroid_mat = centroids[present_clusters]
        with _thread_limited_numeric_ops(1):
            centroid_linkage = fastcluster.linkage_vector(centroid_mat, method="ward", metric="euclidean")
        cluster_order_local = dendrogram(centroid_linkage, no_plot=True)["leaves"]
        assert cluster_order_local is not None  # always set unless dendrogram plots truncated
        cluster_order = [present_clusters[i] for i in cluster_order_local]
    else:
        centroid_linkage = None
        cluster_order = [0]

    ordered_cells = []
    cluster_sizes = []
    for cluster_id in cluster_order:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        cluster_sizes.append(len(idx))
        centroid = centroids[cluster_id]
        denom = np.linalg.norm(centroid)
        if denom > 0:
            scores = data[idx] @ centroid / denom
            idx = idx[np.argsort(scores)]
        ordered_cells.extend(idx.tolist())

    if len(ordered_cells) != n_cells:
        missing = sorted(set(range(n_cells)) - set(ordered_cells))
        ordered_cells.extend(missing)
    return {
        "cell_order": np.asarray(ordered_cells, dtype=int),
        "labels": labels,
        "cluster_order": cluster_order,
        "cluster_sizes": np.asarray(cluster_sizes, dtype=int),
        "centroid_linkage": centroid_linkage,
    }


def _clustered_block_order(mat: FloatArray, n_clusters: int = 128, random_state: int = 1234) -> IntArray:
    return _clustered_block_layout(mat, n_clusters=n_clusters, random_state=random_state)["cell_order"]


def _draw_cluster_dendrogram(ax: Axes, centroid_linkage: LinkageMatrix | None, cluster_sizes: IntArray) -> None:
    """Render a cluster-level dendrogram aligned to block heights in the heatmap."""
    if centroid_linkage is None or len(cluster_sizes) <= 1:
        ax.text(0.5, 0.5, "cluster\ndendrogram\nunavailable", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    dendro = dendrogram(centroid_linkage, no_plot=True)
    block_edges = np.concatenate([[0.0], np.cumsum(cluster_sizes, dtype=float)])
    block_centers = 0.5 * (block_edges[:-1] + block_edges[1:]) - 0.5

    src_centers = np.array([5.0 + 10.0 * i for i in range(len(cluster_sizes))], dtype=float)

    def _map_y(yvals: FloatArray) -> FloatArray:
        return np.interp(yvals, src_centers, block_centers)

    segments = []
    max_height = 0.0
    for icoord, dcoord in zip(dendro["icoord"], dendro["dcoord"], strict=True):
        ys = _map_y(np.asarray(icoord, dtype=float))
        xs = np.asarray(dcoord, dtype=float)
        max_height = max(max_height, float(xs.max()))
        points = np.column_stack([xs, ys])
        segments.extend(
            [
                [points[0], points[1]],
                [points[1], points[2]],
                [points[2], points[3]],
            ]
        )

    lc = LineCollection(segments, colors="black", linewidths=0.8)
    ax.add_collection(lc)
    ax.set_xlim(max_height * 1.05 if max_height > 0 else 1.0, 0.0)
    ax.set_ylim(block_edges[-1] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe_dendrogram_with_recursion_management(Z: LinkageMatrix, ax: Axes, n_cells: int) -> None:
    """Safely plot dendrogram with increased recursion limit.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    n_cells : int
        Number of cells (for estimating required recursion depth).
    """
    # Estimate required recursion depth based on number of cells
    # Rough estimate: depth ~ log2(n_cells) * 2
    estimated_depth = max(1000, int(np.log2(n_cells + 1) * 10 + 500))
    old_limit = sys.getrecursionlimit()

    try:
        # Temporarily increase recursion limit
        sys.setrecursionlimit(min(estimated_depth, 1000000))  # Cap at 1M to avoid stack overflow
        dendrogram(
            Z,
            orientation="left",
            ax=ax,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="black",
            link_color_func=lambda _: "black",
        )
    except RecursionError:
        # Still failed, draw placeholder
        ax.text(0.5, 0.5, "dendrogram\nskipped\n(recursion)", ha="center", va="center", fontsize=8)
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_limit)


def _add_chr_labels(ax: Axes, chrom_info: npt.NDArray[Any], below: bool = False, genome: str = "hg20") -> None:
    """Place chromosome name labels beside the chromosome-bar axes.

    Uses a mixed-coordinate transform (x in data coordinates, y in axes
    fraction) so labels sit outside the coloured band without overlapping it.
    Numeric sex chromosome codes are decoded for the supplied genome.
    """
    chrom_arr = np.asarray(chrom_info)
    n_bins = len(chrom_arr)

    chrom_s = np.array([str(c) for c in chrom_arr])
    boundary_mask = np.concatenate([[True], chrom_s[1:] != chrom_s[:-1]])
    starts = np.where(boundary_mask)[0]
    ends = np.concatenate([starts[1:], [n_bins]])
    chr_ids = chrom_arr[starts]

    # x in data coordinates, y in axes fraction
    trans = ax.get_xaxis_transform()
    ax.set_xlim(-0.5, n_bins - 0.5)
    for cid, s, e in zip(chr_ids, starts, ends, strict=True):
        mid = (s + e - 1) / 2.0
        ax.text(
            mid,
            -0.2 if below else 1.08,
            chromosome_label(cid, genome),
            transform=trans,
            ha="center",
            va="top" if below else "bottom",
            fontsize=8,
            color="black",
            clip_on=False,
        )


@with_default_progress_output
def plot_heatmap(
    mat: FloatArray,
    chrom_info: npt.NDArray[Any],
    predictions: Mapping[str, str] | None = None,
    sample_name: str = "",
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    n_cores: int = 1,
    WNS1: str = "",
    WNS: str = "",
    output_path: str | None = None,
    genome: str = "hg20",
) -> None:
    """Plot CNA heatmap with hierarchical clustering dendrogram.

    Layout mirrors R copykat heatmap.3:
      Row 0 (thin):  [empty] [empty] [chr bar]
      Row 1 (main):  [dendrogram] [pred sidebar] [heatmap]

    For datasets > 200k cells, uses optimized approximate clustering.

    Parameters
    ----------
    mat : np.ndarray, shape (n_bins, n_cells)
        CNA values (bins x cells).
    chrom_info : np.ndarray
        Chromosome IDs for each bin.
    predictions : dict or None
        Cell name -> "aneuploid"/"diploid" predictions.
    sample_name : str
        Sample name for title.
    distance : DistanceMetric
        Distance metric.
    n_cores : int
        Number of cores.
    WNS1 : str
        Data quality warning.
    WNS : str
        Classification warning.
    output_path : str or None
        Path to save figure.
    """
    distance = DistanceMetric(distance)
    if output_path is None:
        output_path = f"{sample_name}_copykat_heatmap.png"

    plot_start = time.perf_counter()
    n_cells = mat.shape[1]

    # Determine cell ordering strategy based on dataset size
    # Strategy 1: Standard hierarchical clustering (up to 20k cells)
    # Strategy 2: K-means clustering (20k-200k cells)
    # Strategy 3: Simple ordering by prediction/CNA (>200k cells)

    max_dendro_cells = 3000
    max_kmeans_cells = 200000
    Z = None
    Z_summary = None
    cell_order: list[int] | np.ndarray | None = None
    skip_dendrogram = False
    cluster_sizes = None

    if n_cells <= max_dendro_cells:
        # Full hierarchical clustering with dendrogram
        logger.info(f"  Step 10a: Computing dendrogram for {n_cells} cells...")
        try:
            Z = _safe_linkage(mat, distance, "ward", n_cores)
            # Use safe dendrogram with recursion management
            leaves: list[int] | None = None
            try:
                old_limit = sys.getrecursionlimit()
                estimated_depth = max(1000, int(np.log2(n_cells + 1) * 10 + 500))
                sys.setrecursionlimit(min(estimated_depth, 1000000))
                leaves = dendrogram(Z, no_plot=True)["leaves"]
                sys.setrecursionlimit(old_limit)
            except RecursionError:
                sys.setrecursionlimit(old_limit)
                logger.warning("  WARNING: dendrogram recursion limit reached; using K-means ordering.")
                leaves = None

            if leaves is not None:
                cell_order = leaves
            else:
                # Fallback to fast block ordering
                cell_order = _clustered_block_order(mat, n_clusters=min(96, max(24, n_cells // 40)))
        except Exception as e:
            logger.warning(f"  WARNING: dendrogram computation failed ({e}); using fast ordering.")
            cell_order = _clustered_block_order(mat, n_clusters=min(96, max(24, n_cells // 40)))

    elif n_cells <= max_kmeans_cells:
        # Fast clustered ordering for large datasets with a summarized dendrogram.
        logger.info(f"  Step 10a: Computing fast clustered ordering for {n_cells} cells...")
        try:
            layout = _clustered_block_layout(mat, n_clusters=min(128, max(32, n_cells // 160)))
            cell_order = layout["cell_order"]
            Z_summary = layout["centroid_linkage"]
            cluster_sizes = layout["cluster_sizes"]
        except Exception as e:
            logger.warning(f"  WARNING: fast clustered ordering failed ({e}); using simple ordering.")
            skip_dendrogram = True
            cell_order = _simple_cell_order(mat, predictions=predictions)

    else:
        # For very large datasets (>200k cells), use simple ordering
        logger.info(f"  Step 10a: Using simple ordering for {n_cells} cells (too large for clustering).")
        skip_dendrogram = True
        cell_order = _simple_cell_order(mat, predictions=predictions)

    # Ensure cell_order is valid
    if cell_order is None:
        cell_order = np.arange(n_cells)

    # Reorder matrix
    mat_ordered = mat[:, cell_order]

    # --- Figure & GridSpec ------------------------------------------------
    h = 10 if n_cells < 3000 else 15
    has_pred = predictions is not None

    # Columns: dendrogram | heatmap | pred sidebar | colorbar | legend
    # Rows:    annotation headings | main | chromosome bar
    n_cols = 5 if has_pred else 3
    if has_pred:
        width_ratios = [8, 50, 1.0, 0.8, 8]
        col_dendro, col_heat, col_pred, col_cbar, col_legend = 0, 1, 2, 3, 4
    else:
        width_ratios = [8, 50, 1.2]
        col_dendro, col_heat, col_cbar = 0, 1, 2

    gs = GridSpec(3, n_cols, height_ratios=[1, 50, 1], width_ratios=width_ratios, hspace=0.02, wspace=0.02)
    fig = plt.figure(figsize=(22, h))
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f"{sample_name}  ·  {n_cells:,} cells", fontsize=15, fontweight="bold", y=0.99)
    caption = warning_caption(WNS1, WNS)
    if caption:
        fig.text(0.5, 0.965, caption, ha="center", va="top", fontsize=11, color="#555555")

    # --- Chromosome bar (bottom, below heatmap only) -----------------------
    ax_chr = fig.add_subplot(gs[2, col_heat])
    chr_colors = (chrom_info.astype(int) % 2).astype(float)
    ax_chr.imshow(chr_colors.reshape(1, -1), aspect="auto", cmap="binary", interpolation="nearest")
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])
    _add_chr_labels(ax_chr, chrom_info, below=True, genome=genome)
    ax_chr.set_xlabel("Genomic position", fontsize=13, labelpad=22)

    # --- Main heatmap -----------------------------------------------------
    ax_heat = fig.add_subplot(gs[1, col_heat])

    vmin, vmax = -0.5, 0.5
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # interpolation_stage="data" samples the data down to screen resolution
    # before norm + colormap (matplotlib's default before 3.10) instead of
    # colormapping every cell x bin value first ("auto" picks "rgba" when
    # downsampling). With nearest-neighbour interpolation both pick the same
    # source values and norm/colormap are per-pixel, so the image is the same,
    # at a fraction of the time and memory.
    im = ax_heat.imshow(
        mat_ordered.T, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", interpolation_stage="data"
    )
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    # Chromosome boundaries
    chrom_changes = np.where(np.diff(chrom_info.astype(int)))[0]
    for pos in chrom_changes:
        ax_heat.axvline(x=pos + 0.5, color="gray", linewidth=0.3, alpha=0.5)

    # --- Dendrogram (left) ------------------------------------------------
    ax_dendro = fig.add_subplot(gs[1, col_dendro])
    if Z is not None and not skip_dendrogram:
        logger.info("  Step 10b: Rendering dendrogram...")
        _safe_dendrogram_with_recursion_management(Z, ax_dendro, n_cells)
    elif Z_summary is not None and cluster_sizes is not None:
        logger.info("  Step 10b: Rendering cluster dendrogram...")
        _draw_cluster_dendrogram(ax_dendro, Z_summary, cluster_sizes)
    else:
        msg = "fast order\n(no dendrogram)" if skip_dendrogram else "dendrogram\nskipped"
        ax_dendro.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9)
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    if Z is not None and not skip_dendrogram:
        # Invert so first leaf is at top (matching imshow origin='upper')
        ax_dendro.invert_yaxis()
        # Tighten y-limits to leaf range so leaves align with heatmap rows
        leaf_step = 10  # scipy default leaf spacing
        leaf_min = 5
        leaf_max = 5 + (n_cells - 1) * leaf_step
        ax_dendro.set_ylim(leaf_max + leaf_step / 2, leaf_min - leaf_step / 2)
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # --- Prediction sidebar (right of the heatmap) ------------------------
    if predictions is not None:
        ax_pred = fig.add_subplot(gs[1, col_pred], sharey=ax_heat)
        pred_list = list(predictions.values())
        pred_ordered = [pred_list[i] if i < len(pred_list) else "not.defined" for i in cell_order]
        pred_palette = _assign_cat_colors(pred_ordered)
        pred_colors = np.array([mcolors.to_rgb(pred_palette[str(p)]) for p in pred_ordered]).reshape(n_cells, 1, 3)
        ax_pred.imshow(
            pred_colors, aspect="auto", interpolation="nearest"
        )
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred_top = fig.add_subplot(gs[0, col_pred])
        ax_pred_top.axis("off")
        ax_pred_top.text(0.5, 0.02, "CopyKAT Python", ha="left", va="bottom", fontsize=10,
                         rotation=45, rotation_mode="anchor", transform=ax_pred_top.transAxes)
        for spine in ax_pred.spines.values():
            spine.set_linewidth(0.6)

    # Hide empty top-left cells
    for c in range(n_cols):
        if c == col_heat or (has_pred and c == col_pred):
            continue
        ax_empty = fig.add_subplot(gs[0, c])
        ax_empty.axis("off")

    # --- Legend for predictions -------------------------------------------
    if has_pred:
        ax_legend = fig.add_subplot(gs[1, col_legend])
        ax_legend.axis("off")
        entries = [(pred_palette[cat], legend_label(cat)) for cat in sorted(pred_palette, key=_natural_sort_key)]
        draw_metadata_legends(ax_legend, [("CopyKAT Python", entries)], start=0.96)

    # --- Colorbar (vertical, right side) ----------------------------------
    ax_cbar = fig.add_subplot(gs[1, col_cbar])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    ax_cbar_top = fig.add_subplot(gs[0, col_cbar])
    ax_cbar_top.axis("off")
    ax_cbar_top.text(0.5, 0.02, "Relative CNA", ha="left", va="bottom", fontsize=10,
                    rotation=45, rotation_mode="anchor", transform=ax_cbar_top.transAxes)
    cbar.ax.tick_params(labelsize=10)
    for ax in [ax_heat, ax_chr, ax_cbar]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#999999")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Heatmap saved to {output_path} in {time.perf_counter() - plot_start:.2f}s")


# ---------------------------------------------------------------------------
# Annotated heatmap with metadata sidebars and row splitting
# ---------------------------------------------------------------------------


def _natural_sort_key(s: object) -> list[int | str]:
    """Sort key that orders numeric substrings numerically (e.g. "10" after "9")."""

    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(s))]


# Fixed colors for copykat prediction values (aneuploid=orange, diploid=blue)
_COPYKAT_PRED_COLORS = {
    "aneuploid": "#E8601C",  # orange
    "c2:aneuploid:low.conf": "#F4A86A",  # light orange
    "diploid": "#3A87C8",  # blue
    "c1:diploid:low.conf": "#9EC8E8",  # light blue
    "not.defined": "#B0B0B0",  # grey
    "unknown": "#D4D4D4",  # light grey
}


def _assign_cat_colors(values: Iterable[object]) -> dict[str, str]:
    """Return {category: hex_color} for every unique value in *values*.

    CopyKAT prediction columns (containing "aneuploid" or "diploid" values)
    always use the fixed palette: orange for aneuploid, blue for diploid.
    All other columns use tab10/tab20/HSV auto-assignment.
    The "unknown" category (cells missing from the meta CSV) is always grey.
    """
    cats = sorted({str(v) for v in values}, key=_natural_sort_key)

    # Detect copykat prediction columns by value content
    if any("aneuploid" in c or "diploid" in c for c in cats):
        return {cat: _COPYKAT_PRED_COLORS.get(cat, "#B0B0B0") for cat in cats}

    n = len(cats)
    if n <= 10:
        palette = [mcolors.to_hex(plt.cm.tab10(i)) for i in range(10)]
    elif n <= 20:
        palette = [mcolors.to_hex(plt.cm.tab20(i)) for i in range(20)]
    else:
        palette = [mcolors.to_hex(plt.cm.hsv(i / n)) for i in range(n)]
    cmap = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
    if "unknown" in cmap:
        cmap["unknown"] = "#cccccc"
    return cmap


def _detect_csv_header(path: str) -> bool:
    """Return True if the CSV first row looks like column names.

    Heuristic: if the second field of row 0 can be parsed as a float it is
    data (no header); otherwise the row is a header.
    """

    row0 = pd.read_csv(path, header=None, nrows=1).iloc[0]
    if len(row0) > 1:
        try:
            float(str(row0.iloc[1]))
            return False
        except (ValueError, TypeError):
            return True
    return True


def _read_meta_csv(path: str) -> pd.DataFrame:
    """Read annotation CSV; first column is used as the cell-name index.

    Auto-detects whether the file has a header row.  All column names are
    cast to strings so they are safe for dict keys and plot labels.
    """

    header = 0 if _detect_csv_header(path) else None
    df = pd.read_csv(path, header=header)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _order_group(
    mat_grp: FloatArray,
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    n_cores: int = 1,
) -> IntArray:
    """Return a cell-ordering index array for one CNA sub-matrix.

    Uses full Ward linkage for groups ≤ 3 000 cells; K-means block ordering
    (same strategy as the main ``plot_heatmap``) for larger groups.
    """
    n = mat_grp.shape[1]
    if n <= 1:
        return np.arange(n, dtype=int)

    if n <= 3000:
        old_lim = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(max(10000, n * 10))
            Z = _safe_linkage(mat_grp, distance, "ward", n_cores)
            dn = dendrogram(Z, no_plot=True)
            return np.array(dn["leaves"], dtype=int)
        except Exception:
            pass
        finally:
            sys.setrecursionlimit(old_lim)

    n_clust = min(128, max(32, n // 160))
    try:
        return _clustered_block_layout(mat_grp, n_clusters=n_clust)["cell_order"]
    except Exception:
        return np.arange(n, dtype=int)


@with_default_progress_output
def plot_heatmap_annotated(
    mat: FloatArray,
    cell_names: Sequence[str],
    chrom_info: npt.NDArray[Any],
    meta_csv: str,
    row_split_col: str | None = None,
    sample_name: str = "",
    distance: DistanceMetric = DistanceMetric.EUCLIDEAN,
    n_cores: int = 1,
    output_path: str | None = None,
    continuous_meta: Sequence[str] | None = None,
    genome: str = "hg20",
) -> None:
    """Plot CNA heatmap with per-cell metadata annotation bars and row splitting.

    Reads a CSV where the first column is the cell name and every remaining
    column is drawn as a narrow coloured sidebar. CopyKAT predictions precede
    continuous numeric measurements on the right; other categories stay left.
    Rows are split into labelled groups according to *row_split_col*; within
    each group cells are ordered by hierarchical or K-means clustering so the
    intra-group CNA structure is preserved.

    Parameters
    ----------
    mat : np.ndarray, shape (n_bins, n_cells)
        CNA log-ratio values (bins × cells).
    cell_names : list of str
        Cell names in the same column order as *mat*.
    chrom_info : np.ndarray
        Integer chromosome ID per bin.
    meta_csv : str
        Annotation CSV path.  First column = cell name; remaining columns
        become annotation sidebars.  Header row is auto-detected.
        Cells present in *mat* but absent from the CSV are labelled "unknown".
    row_split_col : str or None
        Column name used to split rows into labelled groups.  When *None* the
        second column of the CSV is used. An empty string disables row splitting
        and clusters all cells together.
    sample_name : str
        Label shown in the figure title and used for the default filename.
    distance : DistanceMetric
        Distance metric for within-group clustering
        (``"euclidean"``, ``"pearson"``, or ``"spearman"``).
    n_cores : int
        Parallel threads passed to the clustering backend.
    output_path : str or None
        PNG save path.  Defaults to
        ``"{sample_name}_copykat_annotated_heatmap.png"``.
    continuous_meta : sequence of str or None
        Explicit continuous columns, useful for integer measurements with few
        distinct values. By default numeric measurements are detected; small
        integer-coded categories and the row-split column remain categorical.
    """
    distance = DistanceMetric(distance)
    if output_path is None:
        output_path = f"{sample_name}_copykat_annotated_heatmap.png"

    t0 = time.perf_counter()
    n_bins, n_cells = mat.shape
    logger.info(f"  plot_heatmap_annotated: {n_cells} cells × {n_bins} bins")

    # ── 1. Load and align metadata ────────────────────────────────────────
    meta_df = _read_meta_csv(meta_csv)
    ann_cols = meta_df.columns.tolist()

    if row_split_col is None:
        row_split_col = ann_cols[0]
    if row_split_col and row_split_col not in ann_cols:
        raise ValueError(f"row_split_col '{row_split_col}' not found; available: {ann_cols}")
    forced_continuous = set(continuous_meta or [])
    if forced_continuous - set(ann_cols):
        raise ValueError(f"Continuous metadata columns not found: {sorted(forced_continuous - set(ann_cols))}")
    if row_split_col in forced_continuous:
        raise ValueError("The row-split column must remain categorical")

    # Ensure row_split_col is always the leftmost annotation sidebar
    if row_split_col:
        ann_cols = [row_split_col] + [c for c in ann_cols if c != row_split_col]

    cell_names_str = [str(c) for c in cell_names]
    meta_aligned = meta_df.reindex(cell_names_str).fillna("unknown")

    # ── 2. Per-group ordering: sort groups, then cluster cells within ─────
    split_vals = meta_aligned[row_split_col].astype(str).to_numpy() if row_split_col else np.full(n_cells, "")
    group_names = sorted(np.unique(split_vals), key=_natural_sort_key)

    ordered_indices = []
    group_boundaries = [0]

    for grp in group_names:
        grp_idx = np.where(split_vals == grp)[0]
        local_order = _order_group(mat[:, grp_idx], distance, n_cores)
        ordered_indices.extend(grp_idx[local_order].tolist())
        group_boundaries.append(len(ordered_indices))
        logger.info(f"    '{grp}': {len(grp_idx)} cells ordered")

    order = np.array(ordered_indices, dtype=int)
    mat_ordered = mat[:, order]
    meta_ordered = meta_aligned.iloc[order].reset_index(drop=True)

    # ── 3. Build categorical or continuous RGB image arrays ───────────────
    ann_cmaps = {}  # col → {category: hex}
    ann_imgs = {}  # col → float32 array (n_cells, 1, 3)
    continuous: dict[str, ContinuousAnnotation] = {}

    for col in ann_cols:
        if col in forced_continuous or (col != row_split_col and is_continuous(meta_df[col])):
            annotation = continuous_annotation(meta_ordered[col])
            continuous[col] = annotation
            ann_imgs[col] = annotation.colors
            continue
        vals = meta_ordered[col].astype(str)
        cmap_dict = _assign_cat_colors(vals)
        ann_cmaps[col] = cmap_dict
        ann_imgs[col] = np.array([mcolors.to_rgb(cmap_dict[v]) for v in vals], dtype=np.float32).reshape(n_cells, 1, 3)

    # ── 4. Figure layout ──────────────────────────────────────────────────
    layout = annotation_layout(ann_cols, continuous)
    col_grp = 0  # group-name labels
    col_heat = layout.heatmap
    col_cbar = layout.colorbar
    col_leg = layout.legend
    n_cols_total = len(layout.width_ratios)

    width_ratios = layout.width_ratios
    gs = GridSpec(
        3,
        n_cols_total,
        height_ratios=[1, 50, 1],
        width_ratios=width_ratios,
        hspace=0.02,
        wspace=0.02,
    )
    fig_h = max(15.0, min(28.0, 10.0 + n_cells / 20000.0))
    fig = plt.figure(figsize=(22, fig_h))
    fig.subplots_adjust(top=0.90)

    # ── 5. Main heatmap ───────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, col_heat])
    norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax_heat.imshow(
        mat_ordered.T,
        aspect="auto",
        cmap=plt.cm.RdBu_r,
        norm=norm,
        interpolation="nearest",
        interpolation_stage="data",  # see plot_heatmap
    )
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])

    for pos in np.where(np.diff(chrom_info.astype(int)))[0]:
        ax_heat.axvline(x=pos + 0.5, color="gray", linewidth=0.3, alpha=0.5)
    for bd in group_boundaries[1:-1]:
        ax_heat.axhline(y=bd - 0.5, color="white", linewidth=1.5)

    # ── 6. Chromosome bar (bottom row) ────────────────────────────────────
    ax_chr = fig.add_subplot(gs[2, col_heat])
    ax_chr.imshow(
        (chrom_info.astype(int) % 2).reshape(1, -1).astype(float),
        aspect="auto",
        cmap="binary",
        interpolation="nearest",
    )
    ax_chr.set_xticks([])
    ax_chr.set_yticks([])
    _add_chr_labels(ax_chr, chrom_info, below=True, genome=genome)
    ax_chr.set_xlabel("Genomic position", fontsize=13, labelpad=22)

    # ── 7. Annotation sidebars ────────────────────────────────────────────
    for col in ann_cols:
        sidebar_col = layout.annotation_columns[col]
        ax_a = fig.add_subplot(gs[1, sidebar_col], sharey=ax_heat)
        ax_a.imshow(ann_imgs[col], aspect="auto", interpolation="nearest")
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        for bd in group_boundaries[1:-1]:
            ax_a.axhline(y=bd - 0.5, color="white", linewidth=1.5)

        # column name in the top-row cell above each sidebar
        ax_top = fig.add_subplot(gs[0, sidebar_col])
        ax_top.axis("off")
        ax_top.text(
            0.5,
            0.02,
            annotation_title(col),
            ha="left",
            va="bottom",
            fontsize=10,
            rotation=45,
            rotation_mode="anchor",
            transform=ax_top.transAxes,
        )

    # ── 8. Group-name labels (leftmost column) ────────────────────────────
    ax_grp = fig.add_subplot(gs[1, col_grp], sharey=ax_heat)
    ax_grp.set_xlim(0, 1)
    ax_grp.axis("off")
    for i, grp in enumerate(group_names):
        if not row_split_col:
            continue
        y_mid = (group_boundaries[i] + group_boundaries[i + 1]) / 2.0
        ax_grp.text(
            0.98,
            y_mid,
            str(grp),
            ha="right",
            va="center",
            fontsize=10,
        )

    # ── 9. Colour bar ─────────────────────────────────────────────────────
    ax_cbar = fig.add_subplot(gs[1, col_cbar])
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    ax_cbar_top = fig.add_subplot(gs[0, col_cbar])
    ax_cbar_top.axis("off")
    ax_cbar_top.text(0.5, 0.02, "Relative CNA", ha="left", va="bottom", fontsize=10,
                    rotation=45, rotation_mode="anchor", transform=ax_cbar_top.transAxes)
    cbar.ax.tick_params(labelsize=10)

    # ── 10. Categorical legend ────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[1, col_leg])
    ax_leg.axis("off")
    for i, (col, annotation) in enumerate(continuous.items()):
        scale_ax = ax_leg.inset_axes([0.2, 0.92 - i * 0.11, 0.8, 0.02])
        scale = matplotlib.cm.ScalarMappable(norm=annotation.norm, cmap="viridis")
        scale_bar = fig.colorbar(scale, cax=scale_ax, orientation="horizontal", ticks=annotation.ticks)
        scale_ax.set_title(annotation_title(col), fontsize=11, fontweight="bold", loc="left", pad=8)
        scale_bar.ax.tick_params(labelsize=9)
        scale_bar.outline.set_visible(False)
        if col == "n_umi":
            from matplotlib.ticker import StrMethodFormatter

            scale_bar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    legend_groups = []
    for col in ann_cols:
        if col in continuous:
            if continuous[col].has_missing:
                legend_groups.append((annotation_title(col), [("#cccccc", "Missing / nonfinite")]))
            continue
        prediction_column = is_prediction_column(col, ann_cmaps[col])
        entries = [
            (ann_cmaps[col][cat], legend_label(cat, prediction=prediction_column))
            for cat in sorted(ann_cmaps[col], key=_natural_sort_key)
        ]
        legend_groups.append((annotation_title(col), entries))
    draw_metadata_legends(ax_leg, legend_groups, start=0.96 - 0.11 * len(continuous))

    # ── 11. Empty top-row placeholders ───────────────────────────────────
    for c in [col_grp, col_leg]:
        ax_e = fig.add_subplot(gs[0, c])
        ax_e.axis("off")

    for ax in [ax_heat, ax_chr, ax_cbar]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#999999")
    fig.suptitle(f"{sample_name}  ·  {n_cells:,} cells", fontsize=15, fontweight="bold", y=0.99)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {output_path}  ({time.perf_counter() - t0:.2f}s)")
