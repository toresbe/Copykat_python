"""Shared type definitions: names for the data shapes passed between pipeline steps."""

from typing import TypeAlias

from scipy import sparse

# Sparse count matrices as they reach the pipeline: COO from scipy.io.mmread,
# CSC from the AnnData wrapper, CSR from callers and after row selection.
SparseMatrix: TypeAlias = sparse.coo_matrix | sparse.csr_matrix | sparse.csc_matrix
