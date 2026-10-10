"""Small independent checks for input alignment and abstention-aware scoring."""

import gzip
import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np
from navin_accuracy_data import col, labels, stream_mtx
from navin_accuracy_worker import metrics

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    content = b"%%MatrixMarket matrix coordinate integer general\n% fixture\n3 4 5\n1 1 2\n2 2 3\n3 4 5\n1 4 7\n1 4 1\n"
    for suffix in [".mtx", ".mtx.gz"]:
        path = root / ("counts" + suffix)
        if suffix.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(content)
        else:
            path.write_bytes(content)
        matrix, digest = stream_mtx(path, [3, 0], (3, 4))
        np.testing.assert_array_equal(matrix.toarray(), [[8, 2], [0, 0], [5, 0]])
        assert digest == hashlib.sha256(content).hexdigest()
        try:
            stream_mtx(path, [3, 0], (4, 4))
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched metadata axes accepted")
    bad = root / "fractional.mtx"
    bad.write_bytes(content.replace(b"1 1 2", b"1 1 2.5"))
    try:
        stream_mtx(bad, [0], (3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("Normalized counts accepted")
    empty = root / "empty.mtx"
    empty.write_text("%%MatrixMarket matrix coordinate integer general\n2 3 0\n")
    assert stream_mtx(empty, [1], (2, 3))[0].nnz == 0
    with h5py.File(root / "categories.h5", "w") as f:
        node = f.create_group("donor")
        node.create_dataset("categories", data=np.array(["A", "B"], dtype=h5py.string_dtype()))
        node.create_dataset("codes", data=[1, -1, 0])
        np.testing.assert_array_equal(col(f, "donor"), ["B", "", "A"])
truth, _ = labels(["Malignant", "T cells", "Unknown", None, " malignant "])
np.testing.assert_array_equal(truth, [1, 0, -1, -1, 1])
r = metrics([1, 1, 0, 0, -1], [1, -1, 0, 1, 1])
assert (r["tp"], r["tn"], r["fp"], r["fn"]) == (1, 1, 1, 0)
assert r["coverage"] == 0.75 and r["balanced_accuracy_defined"] == 0.75
assert r["coverage_adjusted_balanced_recall"] == 0.5 and r["abstained_positive"] == 1
r = metrics([0, 0, 0], [0, 1, -1])
assert r["balanced_accuracy_defined"] is None and r["coverage_adjusted_specificity"] == 1 / 3
r = metrics([1, 0], [1, 0], [False, True])
assert r["positive"] == 0 and r["negative"] == 1
print(
    "PASS: sparse axis selection, gzip/source digest, count validation, categorical "
    "metadata, unknown labels, abstention penalties, healthy specificity, masks"
)
