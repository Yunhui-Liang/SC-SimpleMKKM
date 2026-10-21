from __future__ import annotations

"""Sparse matrix--thin-matrix multiplication backends.

The production pipeline is memory-bandwidth bound once dense kernels are
removed.  SciPy's CSR @ dense path is an excellent portable fallback, but many
SciPy builds do not parallelize sparse kernels.  This module therefore offers
an optional Numba backend that parallelizes independent CSR rows without
changing operator semantics.  Numba remains optional; if unavailable, ``auto``
falls back to SciPy.
"""

import os
from typing import Literal

import numpy as np
from scipy import sparse

try:  # optional acceleration only
    import numba as _nb  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    _nb = None


if _nb is not None:
    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _csr_spmm_numba(indptr, indices, data, X, out):  # pragma: no cover - JIT body
        n = indptr.shape[0] - 1
        c = X.shape[1]
        for i in _nb.prange(n):
            p0 = indptr[i]
            p1 = indptr[i + 1]
            for t in range(c):
                acc = 0.0
                for p in range(p0, p1):
                    acc += data[p] * X[indices[p], t]
                out[i, t] = acc


def numba_available() -> bool:
    return _nb is not None


def set_numba_threads(n: int) -> int | None:
    if _nb is None:
        return None
    n = max(1, int(n))
    try:
        max_threads = int(getattr(_nb.config, 'NUMBA_NUM_THREADS', n))
        n = min(n, max_threads)
        _nb.set_num_threads(n)
        return int(_nb.get_num_threads())
    except Exception:
        return None


def resolve_spmm_backend(
    backend: str,
    A: sparse.csr_matrix,
    X: np.ndarray,
    *,
    min_nnz_for_numba: int = 5_000_000,
) -> Literal['scipy', 'numba']:
    key = str(backend or 'auto').strip().lower().replace('-', '_')
    if key in {'scipy', 'csr', 'scipy_csr'}:
        return 'scipy'
    if key in {'numba', 'parallel', 'numba_parallel'}:
        if _nb is None:
            raise RuntimeError('spmm_backend=numba requested but numba is not installed')
        return 'numba'
    if key != 'auto':
        raise ValueError("spmm_backend must be 'auto', 'scipy', or 'numba'")
    # The JIT launch overhead is not worthwhile for tiny matrices.  At scale,
    # row-parallel CSR execution is preferred when Numba is available.
    if _nb is not None and int(A.nnz) >= int(min_nnz_for_numba) and X.ndim == 2:
        return 'numba'
    return 'scipy'


def csr_spmm(
    A: sparse.csr_matrix,
    X: np.ndarray,
    *,
    backend: str = 'auto',
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``A @ X`` without changing mathematical semantics.

    ``A`` must be CSR.  ``X`` may be float32 or float64.  The output dtype is
    the NumPy result type of A.data and X.  An optional reusable ``out`` buffer
    avoids repeated large allocations in iterative eigensolvers.
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    X = np.asarray(X)
    if X.ndim == 1:
        X2 = X[:, None]
        vector = True
    elif X.ndim == 2:
        X2 = X
        vector = False
    else:
        raise ValueError('X must be one- or two-dimensional')
    if X2.shape[0] != A.shape[1]:
        raise ValueError('dimension mismatch in csr_spmm')

    dtype = np.result_type(A.data.dtype, X2.dtype)
    mode = resolve_spmm_backend(backend, A, X2)
    if out is None:
        Y = np.empty((A.shape[0], X2.shape[1]), dtype=dtype, order='C')
    else:
        Y = np.asarray(out)
        if Y.shape != (A.shape[0], X2.shape[1]):
            raise ValueError('out has wrong shape')
        if Y.dtype != dtype:
            raise ValueError(f'out dtype {Y.dtype} does not match required {dtype}')

    if mode == 'numba':
        # Keep the CSR structure in its compact production dtype.  Numba will
        # promote each multiply/accumulate as required by X/Y dtype.
        _csr_spmm_numba(A.indptr, A.indices, A.data, X2, Y)
    else:
        Y[...] = A @ X2
    return Y[:, 0] if vector else Y


def backend_info() -> dict:
    info = {'numba_available': bool(_nb is not None), 'numba_threads': None}
    if _nb is not None:
        try:
            info['numba_threads'] = int(_nb.get_num_threads())
        except Exception:
            pass
        info['numba_version'] = getattr(_nb, '__version__', None)
    info['pid'] = os.getpid()
    return info
