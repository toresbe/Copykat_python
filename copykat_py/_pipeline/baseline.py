"""Select a normal reference and subtract it from smoothed expression."""

import logging
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from copykat_py import anchor as _anchor
from copykat_py._pipeline.runtime import _format_seconds, _record_step
from copykat_py._types import (
    AnchorPath,
    AnchorStrategy,
    BaselineWarning,
    CellLineMode,
    ClusterLabels,
    DataQualityStatus,
    FloatArray,
    GeneProfile,
    Genome,
    IntArray,
    ReferenceMode,
    RuntimeInfo,
)
from copykat_py.baseline import (
    FULL_CLUSTER_MAX_CELLS,
    _fit_gmm_3component,
    _hierarchical_cluster,
    baseline_gmm,
    baseline_norm_cl,
    baseline_synthetic,
    get_last_cluster_info,
)
from copykat_py.normal_cells import normal_cells_to_names

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BaselineState:
    """Warning and normal-cell reference selected across baseline strategies.

    Attributes
    ----------
    warning : BaselineWarning
        Classification confidence or reference mode, used in runtime reporting,
        prediction confidence labels, and heatmap warnings.
    normal_cells : list of str, IntArray, or None
        Normal-cell names, or zero-based indices into the cell names supplied
        to baseline selection. ``None`` denotes a synthetic cell-line reference.
    """

    warning: BaselineWarning
    normal_cells: list[str] | IntArray | None


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineOptions:
    """Strategy and clustering settings for selecting a normal reference.

    Attributes
    ----------
    cell_line : CellLineMode
        ``YES`` selects a synthetic baseline for pure cell-line data; ``NO``
        uses supplied normal cells or automatically detects a normal reference.
    anchor : AnchorStrategy
        Strategy for automatic reference selection: ``SIGMA`` ranks clusters
        by their fitted neutral-profile spread; ``MARKERS`` also considers
        immune and endothelial marker evidence. Supplied normals and synthetic
        references take precedence over this setting.
    genome : Genome
        Reference genome, ``HG20`` or ``MM10``, passed to baseline estimation.
    n_cores : int
        Maximum number of CPU workers requested for baseline clustering.
    pca_components : int
        Resolved PCA component cap for large clustering steps. This is the
        effective value selected from the input cell count and genome or from
        the caller's override, rather than the optional requested value.
    """

    cell_line: CellLineMode
    anchor: AnchorStrategy
    genome: Genome
    n_cores: int
    pca_components: int


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    """Relative expression, aligned cluster labels, and the selected reference.

    The array fields are owned by the caller and are not made read-only.

    Attributes
    ----------
    relative_expression : FloatArray
        Baseline-subtracted expression with shape ``(genes, cells)``, retaining
        the gene and cell order of the smoothed input.
    cluster_labels : ClusterLabels
        One 1-based cluster ID per column of ``relative_expression``. Automatic
        GMM fallback retains the labels from the original clustering candidate.
    reference : _BaselineState
        Selected normal-cell reference and its confidence or mode warning.
    """

    relative_expression: FloatArray
    cluster_labels: ClusterLabels
    reference: _BaselineState


def select_baseline(
    norm_mat_smooth: FloatArray,
    cell_name_list: list[str],
    norm_cell_names: str | list[str],
    marker_counts: tuple[pd.Series, pd.Series] | None,
    quality: DataQualityStatus,
    *,
    options: BaselineOptions,
    runtime_info: RuntimeInfo,
) -> BaselineSelection:
    """Select synthetic, supplied, or automatic normals without copying inputs."""
    logger.info("step 4: measuring baselines ...")
    step_start = time.perf_counter()

    if options.cell_line is CellLineMode.YES:
        logger.info("  running pure cell line mode")
        relt = baseline_synthetic(
            norm_mat_smooth,
            min_cells=10,
            n_cores=options.n_cores,
            pca_components=options.pca_components,
            genome=options.genome,
        )
        norm_mat_relat = relt.relative_expression
        CL = relt.cluster_labels
        baseline_state = _BaselineState(BaselineWarning.CELL_LINE, None)
    elif isinstance(norm_cell_names, list) and len(norm_cell_names) > 1:
        # Known normal cells provided
        norm_cell_set = set(norm_cell_names)
        known_normal_mask = np.array([c in norm_cell_set for c in cell_name_list], dtype=bool)
        NNN = known_normal_mask.sum()
        logger.info(f"  {NNN} known normal cells found in dataset")

        if NNN == 0:
            raise ValueError("Known normal cells provided but none found in dataset")

        logger.info("  run with known normal...")
        basel = np.median(norm_mat_smooth[:, known_normal_mask], axis=1)

        # Cluster all cells
        data_t = norm_mat_smooth.T
        step4_reduce = data_t.shape[0] > FULL_CLUSTER_MAX_CELLS
        km = 6
        CL, Z = _hierarchical_cluster(
            data_t,
            km,
            method="ward",
            metric="euclidean",
            n_cores=options.n_cores,
            reduce=step4_reduce,
            pca_components=options.pca_components,
        )

        while not all(np.bincount(CL)[np.bincount(CL) > 0] > 5):
            km -= 1
            if Z is not None:
                CL = fcluster(Z, t=km, criterion="maxclust")
            else:
                CL, Z = _hierarchical_cluster(
                    data_t,
                    km,
                    method="ward",
                    metric="euclidean",
                    n_cores=options.n_cores,
                    reduce=step4_reduce,
                    pca_components=options.pca_components,
                )
            if km == 2:
                break

        baseline_state = _BaselineState(
            BaselineWarning.KNOWN_NORMAL,
            np.asarray(cell_name_list, dtype=object)[known_normal_mask].tolist(),
        )
        norm_mat_relat = norm_mat_smooth - basel[:, np.newaxis]
    else:
        # Auto-detect normal cells
        anchor_selector = None
        if options.anchor is AnchorStrategy.MARKERS and marker_counts is not None:
            immune_counts = marker_counts[0].groupby(level=0).first().reindex(cell_name_list).fillna(0).to_numpy()
            endothelial_counts = marker_counts[1].groupby(level=0).first().reindex(cell_name_list).fillna(0).to_numpy()

            def anchor_selector(labels: ClusterLabels, sigma_cluster: int) -> tuple[int, AnchorPath]:
                selected, path = _anchor.choose_anchor_cluster(labels, immune_counts, endothelial_counts, sigma_cluster)
                return int(selected), path

        basa = baseline_norm_cl(
            norm_mat_smooth,
            min_cells=5,
            n_cores=options.n_cores,
            cell_names=cell_name_list,
            pca_components=options.pca_components,
            genome=options.genome,
            anchor_selector=anchor_selector,
        )
        assert basa.cluster_labels is not None  # baseline_norm_cl always clusters
        CL = basa.cluster_labels
        runtime_info["anchor_path"] = basa.anchor_path
        if options.anchor is AnchorStrategy.MARKERS:
            basa = replace(
                basa,
                warning=(
                    BaselineWarning.NONE
                    if basa.anchor_path in {AnchorPath.IMMUNE, AnchorPath.ENDOTHELIAL}
                    else BaselineWarning.UNCLASSIFIED
                ),
            )
            logger.info(
                f"  normal reference from markers: path={runtime_info['anchor_path']}, cells={len(basa.normal_cells)}"
            )

        if basa.warning is BaselineWarning.UNCLASSIFIED and options.anchor is not AnchorStrategy.MARKERS:
            cluster_preN = list(basa.normal_cells)
            keep_cluster_anchor = quality is DataQualityStatus.LOW and len(cluster_preN) >= max(
                50, int(0.05 * len(cell_name_list))
            )
            if keep_cluster_anchor:
                logger.info("  low-data-quality mode: keeping cluster-based normal anchor")
            else:
                basa_cluster = basa
                basa_gmm = baseline_gmm(
                    norm_mat_smooth,
                    cell_name_list,
                    max_normal=5,
                    mu_cut=0.05,
                    Nfraq_cut=0.99,
                    RE_before=basa_cluster,
                    n_cores=options.n_cores,
                    pca_components=options.pca_components,
                    genome=options.genome,
                    cluster=False,  # only basel/preN are used; CL stays from clustering
                )

                # baseline_gmm anchors on a handful of individually-scanned
                # cells (it stops at the first `max_normal` hits in raw cell
                # order) and can be far noisier than the clustering candidate
                # it is meant to replace -- a contaminated anchor set here
                # silently inverts the final diploid/aneuploid call downstream,
                # since cluster identity is decided purely by preN overlap.
                # Only adopt the fallback when its baseline profile is a
                # tighter, more confidently-neutral fit than the candidate it
                # would discard; otherwise keep the clustering answer even
                # though confidence is flagged low.
                #
                # Compare the two candidates with the same 3-component GMM
                # sigma that baseline_norm_cl already uses to rank its own
                # six clusters against each other, rather than a raw
                # mean(|basel|) magnitude. Magnitude is fit over genes for
                # both candidates, so it isn't literally biased by the
                # cell-count each basel was averaged over -- but a bigger,
                # more heterogeneous candidate can still land on a smaller
                # mean(|basel|) via cross-subpopulation cancellation rather
                # than genuine uniform neutrality, without that cancellation
                # showing up as a tighter (lower-sigma) GMM fit. Sigma
                # measures how cleanly the profile separates into
                # loss/neutral/gain, which is what "confidently neutral"
                # actually means here, so it is the more consistent yardstick
                # to reuse for this cross-candidate comparison.
                def _basel_sigma(basel_vec: GeneProfile) -> float:
                    sigma_init = max(0.05, 0.5 * float(np.std(basel_vec)))
                    return _fit_gmm_3component(basel_vec, sigma_init=sigma_init, max_iter=5000)[2]

                clustering_sigma = float(_basel_sigma(basa_cluster.baseline))
                gmm_sigma = float(_basel_sigma(basa_gmm.baseline))
                if gmm_sigma < clustering_sigma:
                    basa = basa_gmm
                else:
                    logger.info(
                        f"  GMM fallback baseline (sigma={gmm_sigma:.4f}) is not tighter/more confidently "
                        f"neutral than the clustering candidate (sigma={clustering_sigma:.4f}); "
                        "keeping cluster-based normal anchor"
                    )
                basa = replace(basa, warning=BaselineWarning.UNCLASSIFIED)

        norm_mat_relat = norm_mat_smooth - basa.baseline[:, np.newaxis]
        baseline_state = _BaselineState(basa.warning, basa.normal_cells)
    del norm_mat_smooth
    baseline_cluster_info = get_last_cluster_info()
    if options.cell_line is CellLineMode.YES:
        reference_mode = ReferenceMode.SYNTHETIC
    elif baseline_state.warning is BaselineWarning.KNOWN_NORMAL:
        reference_mode = ReferenceMode.KNOWN_NORMAL
    else:
        reference_mode = ReferenceMode.AUTOMATIC
    runtime_info["reference"] = {
        "mode": reference_mode,
        "supplied_count": len(set(norm_cell_names)) if isinstance(norm_cell_names, list) else 0,
        "matched_supplied_count": len(set(norm_cell_names).intersection(cell_name_list))
        if isinstance(norm_cell_names, list)
        else 0,
        "baseline_anchor_count": len(
            normal_cells_to_names(baseline_state.normal_cells, cell_name_list).intersection(cell_name_list)
        ),
    }
    elapsed = _record_step(
        runtime_info,
        "baseline_estimation",
        step_start,
        parallel_info=baseline_cluster_info,
        extra={"warning": baseline_state.warning},
    )
    logger.info(
        f"  baseline runtime: {_format_seconds(elapsed)} "
        f"(parallel={baseline_cluster_info['parallel']}, cores={baseline_cluster_info['effective_cores']}, "
        f"engine={baseline_cluster_info.get('engine', 'n/a')})"
    )

    return BaselineSelection(norm_mat_relat, CL, baseline_state)
