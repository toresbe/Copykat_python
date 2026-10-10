"""Memory-aware serialization of CopyKAT result matrices and segments."""

import io
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from copykat_py._types import FloatArray
from copykat_py.baseline import _effective_threads

_WRITE_CHUNK_BYTES = 512 << 20
# Parallel formatting keeps several chunks (plus their rounded copies and
# text) in flight, so it gets a smaller total budget. Below the minimum rows
# per chunk, per-column pyarrow overhead outweighs the gain from parallelism,
# so very wide matrices stay serial.
_PARALLEL_WRITE_BYTES = 128 << 20
_PARALLEL_WRITE_MIN_ROWS = 128


def _row_run_starts(values: FloatArray, block_elems: int = 1 << 26) -> npt.NDArray[np.intp]:
    """Find value rows that differ in representation, including signed zero."""
    n_rows, n_cols = values.shape
    if n_rows < 2:
        return np.arange(n_rows, dtype=np.intp)
    change = np.empty(n_rows - 1, dtype=bool)
    step = max(1, block_elems // max(1, n_cols))
    for lo in range(0, n_rows - 1, step):
        hi = min(n_rows - 1, lo + step)
        block = np.ascontiguousarray(values[lo : hi + 1])
        if block.dtype.kind in "fc":
            row_dtype = np.dtype((np.void, block.dtype.itemsize * n_cols))
            row_bytes = block.view(row_dtype).reshape(-1)
            change[lo:hi] = row_bytes[1:] != row_bytes[:-1]
        else:
            change[lo:hi] = np.any(block[1:] != block[:-1], axis=1)
    return np.concatenate(([0], np.flatnonzero(change) + 1)).astype(np.intp, copy=False)


def _csv_lines(table: Any, write_kwargs: dict[str, Any], include_header: bool = False) -> list[bytes]:
    buffer = io.BytesIO()
    options = pa_csv.WriteOptions(include_header=include_header, **write_kwargs)
    pa_csv.write_csv(table, buffer, write_options=options)
    return buffer.getvalue().split(b"\n")[:-1]


def _write_csv_repeated_rows(
    path: str,
    lead_table: Any,
    schema: Any,
    values: FloatArray,
    run_starts: npt.NDArray[np.intp],
    round_floats: bool,
    write_kwargs: dict[str, Any],
    value_type: Any,
) -> bool:
    """Format each repeated value row once, retaining per-row annotations."""
    n_rows = values.shape[0]
    unique = values[run_starts]
    if round_floats and np.issubdtype(unique.dtype, np.floating):
        unique = np.round(unique, 6)
    unique = np.asfortranarray(unique)
    value_schema = pa.schema(list(schema)[lead_table.num_columns :])
    value_table = pa.Table.from_arrays(
        [pa.array(unique[:, j], type=value_type, from_pandas=True) for j in range(unique.shape[1])],
        schema=value_schema,
    )
    value_lines = _csv_lines(value_table, write_kwargs)
    lead_lines = _csv_lines(lead_table, write_kwargs)
    header = _csv_lines(schema.empty_table(), write_kwargs, include_header=True)
    if len(value_lines) != len(run_starts) or len(lead_lines) != n_rows or len(header) != 1:
        return False
    run_of_row = np.repeat(np.arange(len(run_starts)), np.diff(np.concatenate([run_starts, [n_rows]])))
    with open(path, "wb") as handle:
        handle.write(header[0] + b"\n")
        batch = []
        for i in range(n_rows):
            batch.extend((lead_lines[i], b"\t", value_lines[run_of_row[i]], b"\n"))
            if len(batch) >= 4096:
                handle.write(b"".join(batch))
                batch.clear()
        handle.write(b"".join(batch))
    return True


def _write_cna_csv(
    path: str,
    lead_df: pd.DataFrame,
    values: FloatArray,
    value_columns: Sequence[str],
    round_floats: bool = True,
    quote_strings: bool = True,
    n_cores: int = 1,
) -> None:
    """Write ``lead_df`` columns followed by ``values`` columns as TSV, in row chunks.

    Equivalent to writing ``pd.concat([lead_df, pd.DataFrame(values, columns=value_columns)], axis=1)``
    but never materializes that frame (or a rounded copy of it); only one chunk
    of rows is converted at a time. Uses pyarrow (fast) with 6 d.p. float
    precision. With ``round_floats=False`` floats keep their shortest
    round-trip repr, and with ``quote_strings=False`` strings and the header
    are left unquoted; both together match pandas ``.to_csv(sep="\\t", index=False)``
    output.

    With ``n_cores > 1``, up to ``n_cores`` chunks are formatted
    concurrently and written in order, producing the same bytes; the total
    in-flight chunk size stays within ``_PARALLEL_WRITE_BYTES``.
    """
    lead_df = lead_df.reset_index(drop=True)
    if round_floats:
        lead_float_cols = [c for c in lead_df.columns if pd.api.types.is_float_dtype(lead_df[c])]
        if lead_float_cols:
            lead_df = lead_df.copy()
            lead_df[lead_float_cols] = lead_df[lead_float_cols].round(6)
    value_columns = list(value_columns)
    n_rows = values.shape[0]
    row_bytes = max(1, len(value_columns) * values.itemsize)
    chunk_rows = max(1, _WRITE_CHUNK_BYTES // row_bytes)

    def _value_chunks() -> Iterator[tuple[int, int, FloatArray]]:
        for start in range(0, n_rows, chunk_rows):
            stop = min(start + chunk_rows, n_rows)
            block = values[start:stop]
            if round_floats and np.issubdtype(block.dtype, np.floating):
                block = np.round(block, 6)
            yield start, stop, block

    write_kwargs = {"delimiter": "\t"}
    if not quote_strings:
        write_kwargs["quoting_style"] = "none"
        write_kwargs["quoting_header"] = "none"
    write_options = pa_csv.WriteOptions(**write_kwargs)

    lead_table = pa.Table.from_pandas(lead_df, preserve_index=False).replace_schema_metadata(None)
    value_type = pa.from_numpy_dtype(values.dtype)
    schema = pa.schema(list(lead_table.schema) + [pa.field(str(c), value_type) for c in value_columns])

    if n_rows > 0 and values.shape[1] > 0:
        run_starts = _row_run_starts(values)
        if 2 * len(run_starts) <= n_rows and _write_csv_repeated_rows(
            path, lead_table, schema, values, run_starts, round_floats, write_kwargs, value_type
        ):
            release_unused = getattr(pa.default_memory_pool(), "release_unused", None)
            if release_unused is not None:
                release_unused()
            return

    def _chunk_table(start: int, stop: int, block: FloatArray) -> Any:
        block = np.asfortranarray(block)  # contiguous columns
        arrays = [col.combine_chunks() for col in lead_table.slice(start, stop - start).columns]
        arrays += [pa.array(block[:, j], type=value_type, from_pandas=True) for j in range(block.shape[1])]
        return pa.Table.from_arrays(arrays, schema=schema)

    n_threads = _effective_threads(n_cores)
    parallel_rows = _PARALLEL_WRITE_BYTES // (n_threads * row_bytes)
    if n_threads > 1 and parallel_rows >= _PARALLEL_WRITE_MIN_ROWS and n_rows > parallel_rows:
        chunk_rows = parallel_rows

        def _format(start: int) -> bytes:
            stop = min(start + chunk_rows, n_rows)
            block = values[start:stop]
            if round_floats and np.issubdtype(block.dtype, np.floating):
                block = np.round(block, 6)
            buf = io.BytesIO()
            options = pa_csv.WriteOptions(include_header=(start == 0), **write_kwargs)
            pa_csv.write_csv(_chunk_table(start, stop, block), buf, write_options=options)
            return buf.getvalue()

        pending: deque[Future[bytes]] = deque()
        with open(path, "wb") as f, ThreadPoolExecutor(n_threads) as executor:
            for start in range(0, n_rows, chunk_rows):
                pending.append(executor.submit(_format, start))
                if len(pending) >= n_threads:
                    f.write(pending.popleft().result())
            while pending:
                f.write(pending.popleft().result())
    else:
        with open(path, "wb") as f, pa_csv.CSVWriter(f, schema, write_options=write_options) as writer:
            for start, stop, block in _value_chunks():
                writer.write_table(_chunk_table(start, stop, block))
    # Arrow's allocator keeps freed chunk buffers cached; hand them back
    # so they don't inflate the peak of the steps that follow.
    pa.default_memory_pool().release_unused()


def _frame_with_leading_columns(
    lead_df: pd.DataFrame, values: FloatArray, value_columns: Sequence[str]
) -> pd.DataFrame:
    """DataFrame of ``lead_df`` columns followed by ``values``, without copying ``values``."""
    frame = pd.DataFrame(values, columns=list(value_columns), copy=False)
    lead_df = lead_df.reset_index(drop=True)
    for pos, col in enumerate(lead_df.columns):
        frame.insert(pos, col, lead_df[col].to_numpy())
    return frame


def _write_seg_file(RNA_adj_df: pd.DataFrame, mat_adj: FloatArray, cell_cols: Sequence[str], sample_name: str) -> None:
    """Write .seg file for IGV visualization."""
    rows = []
    chroms = RNA_adj_df["chrom"].to_numpy()
    chrompos = RNA_adj_df["chrompos"].values
    unique_chroms = np.unique(chroms)

    for ci, cell in enumerate(cell_cols):
        vals = mat_adj[:, ci]
        for chrom_val in unique_chroms:
            mask = chroms == chrom_val
            sub_vals = vals[mask]
            sub_pos = chrompos[mask]

            # RLE encoding
            if len(sub_vals) == 0:
                continue

            changes = np.where(np.diff(sub_vals) != 0)[0] + 1
            starts = np.concatenate([[0], changes])
            ends = np.concatenate([changes, [len(sub_vals)]])

            for s, e in zip(starts, ends, strict=True):
                rows.append(
                    {
                        "ID": cell,
                        "chrom": chrom_val,
                        "loc.start": sub_pos[s],
                        "loc.end": sub_pos[e - 1],
                        "num.mark": e - s,
                        "seg.mean": sub_vals[s],
                    }
                )

    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(f"{sample_name}CNA_results.seg", sep="\t", index=False)
