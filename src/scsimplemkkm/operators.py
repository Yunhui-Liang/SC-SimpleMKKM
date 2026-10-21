from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from scipy import sparse

from .spmm import csr_spmm


def _scale_csr_columns_inplace(A: sparse.csr_matrix, scale: np.ndarray, block_nnz: int = 8_000_000) -> None:
    """In-place ``A[:,j] *= scale[j]`` with bounded temporary memory."""
    scale = np.asarray(scale)
    nnz = int(A.nnz)
    step = max(1, int(block_nnz))
    for p in range(0, nnz, step):
        q = min(nnz, p + step)
        A.data[p:q] *= scale[A.indices[p:q]]


def _scale_csr_rows_inplace(A: sparse.csr_matrix, scale: np.ndarray, block_rows: int = 262_144) -> None:
    """In-place ``A[i,:] *= scale[i]`` without constructing a sparse diagonal."""
    scale = np.asarray(scale)
    n = int(A.shape[0])
    bs = max(1, int(block_rows))
    for s in range(0, n, bs):
        e = min(n, s + bs)
        p0 = int(A.indptr[s]); p1 = int(A.indptr[e])
        if p1 <= p0:
            continue
        counts = np.diff(A.indptr[s:e + 1])
        row_scale = np.repeat(scale[s:e], counts)
        A.data[p0:p1] *= row_scale


def _scale_csr_both_sides_inplace(
    A: sparse.csr_matrix,
    scale: np.ndarray,
    *,
    block_rows: int = 262_144,
    block_nnz: int = 8_000_000,
) -> None:
    _scale_csr_columns_inplace(A, scale, block_nnz=block_nnz)
    _scale_csr_rows_inplace(A, scale, block_rows=block_rows)


def _dot_vec_mat64(v: np.ndarray, Z: np.ndarray, block_rows: int = 262_144) -> np.ndarray:
    """Return ``v.T @ Z`` with float64 accumulation and bounded temporaries."""
    v = np.asarray(v)
    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    if v.shape[0] != Z.shape[0]:
        raise ValueError('dimension mismatch in vector-matrix dot')
    out = np.zeros(Z.shape[1], dtype=np.float64)
    bs = max(1, int(block_rows))
    for s in range(0, Z.shape[0], bs):
        e = min(Z.shape[0], s + bs)
        # einsum accumulates in float64 without materializing a full n x c copy.
        out += np.einsum('i,ij->j', v[s:e], Z[s:e], dtype=np.float64, optimize=True)
    return out


def _fro_inner64(A: np.ndarray, B: np.ndarray, block_rows: int = 262_144) -> float:
    A = np.asarray(A); B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError('shape mismatch in Frobenius inner product')
    total = 0.0
    bs = max(1, int(block_rows))
    for s in range(0, A.shape[0], bs):
        e = min(A.shape[0], s + bs)
        total += float(np.einsum('ij,ij->', A[s:e], B[s:e], dtype=np.float64, optimize=True))
    return float(total)


@dataclass
class ImplicitCenteredNormalizedKernel:
    """Exact Q^-1/2 C P^q C Q^-1/2 operator.

    For q=1 (the scale path), the centered/normalized operator is represented
    as a sparse core plus rank-one corrections.  v0.7.6 performs all sparse
    diagonal scalings in-place and blockwise, avoiding O(nnz) scale temporaries
    and avoiding a second full CSR copy when the caller transfers ownership.
    """

    P: sparse.csr_matrix
    diffusion_power: int = 1
    eps: float = 1e-8
    dtype: type = np.float32
    copy_input: bool = True
    spmm_backend: str = 'scipy'
    reduction_block_rows: int = 262_144

    def __post_init__(self) -> None:
        self.P = sparse.csr_matrix(self.P, dtype=self.dtype, copy=bool(self.copy_input))
        self.P.sum_duplicates(); self.P.sort_indices()
        if self.P.shape[0] != self.P.shape[1]:
            raise ValueError('P must be square')
        if self.diffusion_power not in (1, 2):
            raise ValueError('diffusion_power currently supports only 1 or 2')
        self.n = int(self.P.shape[0])
        self.reduction_block_rows = max(4096, int(self.reduction_block_rows))

        # Row sums avoid allocating an n-vector of ones.
        r1 = np.asarray(self.P.sum(axis=1)).ravel().astype(np.float64, copy=False)
        if self.diffusion_power == 1:
            row_sum_B = r1
            diag_B = self.P.diagonal().astype(np.float64, copy=False)
        else:
            row_sum_B = np.asarray(self.P @ r1.astype(self.dtype)).ravel().astype(np.float64, copy=False)
            diag_B = np.asarray(self.P.multiply(self.P).sum(axis=1)).ravel().astype(np.float64, copy=False)
        sum_B = float(np.sum(row_sum_B, dtype=np.float64))
        qdiag64 = np.maximum(
            diag_B - 2.0 * row_sum_B / self.n + sum_B / (self.n * self.n),
            self.eps,
        )
        # The applied production operator stores float32 scaling.  Retaining an
        # additional float64 qdiag and row-sum vector served no runtime purpose
        # and costs 120 MB/kernel at n=10M.
        self.inv_sqrt_q = (1.0 / np.sqrt(qdiag64)).astype(self.dtype)
        # qdiag is an inspection aid, never used in production q=1 matmat.
        # In laptop mode the builder may drop it after initialization.
        self.qdiag = qdiag64.astype(self.dtype)

        self.sparse_core: sparse.csr_matrix | None = None
        self.d: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.s: float | None = None
        if self.diffusion_power == 1:
            D = self.inv_sqrt_q
            # P is no longer needed after q=1 preprocessing.  Reuse its CSR
            # storage as D P D instead of making another full sparse copy.
            S = self.P
            _scale_csr_both_sides_inplace(S, D, block_rows=self.reduction_block_rows)
            S.sum_duplicates(); S.sort_indices()
            self.sparse_core = S
            self.d = D  # alias inv_sqrt_q; D 1
            self.u = (D.astype(np.float64) * row_sum_B).astype(self.dtype)  # D B 1
            self.s = sum_B
            self.P = sparse.csr_matrix((0, 0), dtype=self.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def can_fuse_sparse_core(self) -> bool:
        return self.diffusion_power == 1 and self.sparse_core is not None

    def _center(self, Z: np.ndarray) -> np.ndarray:
        return Z - np.mean(Z, axis=0, keepdims=True, dtype=np.float64).astype(Z.dtype)

    def _apply_B_q2(self, Z: np.ndarray) -> np.ndarray:
        Y = csr_spmm(self.P, Z, backend=self.spmm_backend)
        Y = csr_spmm(self.P, Y, backend=self.spmm_backend)
        return np.asarray(Y, dtype=self.dtype)

    def matmat(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=self.dtype)
        was_vector = Z.ndim == 1
        if was_vector:
            Z = Z[:, None]
        if self.can_fuse_sparse_core:
            assert self.sparse_core is not None and self.d is not None and self.u is not None and self.s is not None
            Y = np.asarray(csr_spmm(self.sparse_core, Z, backend=self.spmm_backend), dtype=self.dtype)
            uz = _dot_vec_mat64(self.u, Z, self.reduction_block_rows)
            dz = _dot_vec_mat64(self.d, Z, self.reduction_block_rows)
            a = (uz / self.n).astype(self.dtype)
            b = (dz / self.n).astype(self.dtype)
            c = ((self.s / (self.n * self.n)) * dz).astype(self.dtype)
            bs = self.reduction_block_rows
            for s in range(0, self.n, bs):
                e = min(self.n, s + bs)
                Y[s:e] -= self.d[s:e, None] * a[None, :]
                Y[s:e] -= self.u[s:e, None] * b[None, :]
                Y[s:e] += self.d[s:e, None] * c[None, :]
        else:
            Y = self.inv_sqrt_q[:, None] * Z
            Y = self._center(Y)
            Y = self._apply_B_q2(Y)
            Y = self._center(Y)
            Y = self.inv_sqrt_q[:, None] * Y
        return Y[:, 0] if was_vector else Y

    def matmat64(self, Z: np.ndarray) -> np.ndarray:
        """Apply the same stored operator in float64 arithmetic without a float64 CSR copy."""
        Z = np.asarray(Z, dtype=np.float64)
        was_vector = Z.ndim == 1
        if was_vector:
            Z = Z[:, None]
        if self.can_fuse_sparse_core:
            assert self.sparse_core is not None and self.d is not None and self.u is not None and self.s is not None
            Y = np.asarray(csr_spmm(self.sparse_core, Z, backend=self.spmm_backend), dtype=np.float64)
            uz = _dot_vec_mat64(self.u, Z, self.reduction_block_rows)
            dz = _dot_vec_mat64(self.d, Z, self.reduction_block_rows)
            a = uz / self.n
            b = dz / self.n
            c = (self.s / (self.n * self.n)) * dz
            bs = self.reduction_block_rows
            for s in range(0, self.n, bs):
                e = min(self.n, s + bs)
                d = np.asarray(self.d[s:e], dtype=np.float64)[:, None]
                u = np.asarray(self.u[s:e], dtype=np.float64)[:, None]
                Y[s:e] -= d * a[None, :]
                Y[s:e] -= u * b[None, :]
                Y[s:e] += d * c[None, :]
        else:
            inv = np.asarray(self.inv_sqrt_q, dtype=np.float64)
            Y = inv[:, None] * Z
            Y = Y - np.mean(Y, axis=0, keepdims=True, dtype=np.float64)
            Y = np.asarray(csr_spmm(self.P, Y, backend=self.spmm_backend), dtype=np.float64)
            Y = np.asarray(csr_spmm(self.P, Y, backend=self.spmm_backend), dtype=np.float64)
            Y = Y - np.mean(Y, axis=0, keepdims=True, dtype=np.float64)
            Y = inv[:, None] * Y
        return Y[:, 0] if was_vector else Y

    def quadratic_trace(self, H: np.ndarray) -> float:
        KH = self.matmat(H)
        return _fro_inner64(H, KH, self.reduction_block_rows)

    def quadratic_trace64(self, H: np.ndarray) -> float:
        H64 = np.asarray(H, dtype=np.float64)
        KH = self.matmat64(H64)
        return _fro_inner64(H64, KH, self.reduction_block_rows)

    def memory_bytes(self) -> int:
        total = 0
        seen_arrays: set[int] = set()
        for arr in (self.inv_sqrt_q, self.qdiag, self.d, self.u):
            if arr is not None and id(arr) not in seen_arrays:
                total += arr.nbytes
                seen_arrays.add(id(arr))
        if self.sparse_core is not None:
            total += self.sparse_core.data.nbytes + self.sparse_core.indices.nbytes + self.sparse_core.indptr.nbytes
        if self.P.shape != (0, 0):
            total += self.P.data.nbytes + self.P.indices.nbytes + self.P.indptr.nbytes
        return int(total)

    def explicit(self) -> np.ndarray:
        """Testing helper only."""
        eye = np.eye(self.n, dtype=self.dtype)
        return self.matmat(eye).astype(np.float64)


class FusedCenteredOperator:
    """One outer-iteration fused operator for SimpleMKKM.

    Sparse cores are fused within identical CSR structures.  v0.7.6 optionally
    reuses one data buffer per support family across line-search trials, so a
    10M-scale trial does not repeatedly allocate/free hundreds of MB of fused
    sparse values.
    """

    def __init__(
        self,
        kernels: list[ImplicitCenteredNormalizedKernel],
        gamma: np.ndarray,
        dtype=np.float32,
        correction_block_rows: int = 262144,
        *,
        spmm_backend: str = 'scipy',
        data_buffers: dict[tuple[int, int], np.ndarray] | None = None,
        fused_buffer_dir: str | Path | None = None,
    ):
        self.kernels = kernels
        self.gamma = np.asarray(gamma, dtype=np.float64)
        self.dtype = dtype
        self.n = kernels[0].n
        self.correction_block_rows = max(4096, int(correction_block_rows))
        self.spmm_backend = str(spmm_backend)
        self.fast = all(k.can_fuse_sparse_core for k in kernels)
        self.cores: list[sparse.csr_matrix] = []
        self._data_buffers = data_buffers
        self.fused_buffer_dir = None if fused_buffer_dir is None else Path(fused_buffer_dir)
        if self.fast:
            groups: dict[tuple[int, int], list[tuple[float, sparse.csr_matrix]]] = {}
            for g, op in zip(self.gamma, kernels):
                if g == 0.0:
                    continue
                assert op.sparse_core is not None
                S = op.sparse_core
                key = (id(S.indices), id(S.indptr))
                groups.setdefault(key, []).append((float(g * g), S))
            for key, items in groups.items():
                base = items[0][1]
                if data_buffers is not None:
                    data = data_buffers.get(key)
                    if data is None or data.shape != base.data.shape or data.dtype != np.dtype(self.dtype):
                        if self.fused_buffer_dir is not None:
                            from .lowmem import check_disk_capacity
                            self.fused_buffer_dir.mkdir(parents=True, exist_ok=True)
                            check_disk_capacity(self.fused_buffer_dir, int(base.data.size) * np.dtype(self.dtype).itemsize)
                            path = self.fused_buffer_dir / f'fused_{len(data_buffers):03d}.npy'
                            data = np.lib.format.open_memmap(path, mode='w+', dtype=self.dtype, shape=base.data.shape)
                        else:
                            data = np.empty_like(base.data, dtype=self.dtype)
                        data_buffers[key] = data
                    data.fill(0)
                else:
                    data = np.zeros_like(base.data, dtype=self.dtype)
                for w, S in items:
                    # np.multiply(..., out=...) would need a second nnz buffer;
                    # keep one bounded temporary by processing the data in chunks.
                    step = 8_000_000
                    ww = np.float32(w)
                    for p in range(0, data.size, step):
                        q = min(data.size, p + step)
                        data[p:q] += ww * S.data[p:q]
                core = sparse.csr_matrix((data, base.indices, base.indptr), shape=base.shape, copy=False)
                self.cores.append(core)

    @property
    def n_sparse_groups(self) -> int:
        return len(self.cores) if self.fast else len(self.kernels)

    def _rank_updates(self, out: np.ndarray, Z: np.ndarray) -> None:
        n = self.n
        bs = self.correction_block_rows
        for g, op in zip(self.gamma, self.kernels):
            if g == 0.0:
                continue
            assert op.d is not None and op.u is not None and op.s is not None
            w = float(g * g)
            uz = _dot_vec_mat64(op.u, Z, bs)
            dz = _dot_vec_mat64(op.d, Z, bs)
            a = (-w * uz / n).astype(self.dtype)
            b = (-w * dz / n).astype(self.dtype)
            c = (w * (op.s / (n * n)) * dz).astype(self.dtype)
            for s in range(0, n, bs):
                e = min(n, s + bs)
                out[s:e] += op.d[s:e, None] * a[None, :]
                out[s:e] += op.u[s:e, None] * b[None, :]
                out[s:e] += op.d[s:e, None] * c[None, :]

    def matmat(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=self.dtype)
        if not self.fast:
            out = np.zeros_like(Z, dtype=self.dtype)
            for g, op in zip(self.gamma, self.kernels):
                if g != 0.0:
                    out += np.float32(g * g) * op.matmat(Z)
            return out
        if not self.cores:
            out = np.zeros_like(Z, dtype=self.dtype)
        else:
            out = np.asarray(csr_spmm(self.cores[0], Z, backend=self.spmm_backend), dtype=self.dtype)
            if len(self.cores) > 1:
                scratch = np.empty_like(out)
                for core in self.cores[1:]:
                    csr_spmm(core, Z, backend=self.spmm_backend, out=scratch)
                    out += scratch
        self._rank_updates(out, Z)
        return out

    def matmat64(self, Z: np.ndarray) -> np.ndarray:
        """High-precision application without materializing float64 sparse cores."""
        Z = np.asarray(Z, dtype=np.float64)
        if not self.fast:
            out = np.zeros_like(Z, dtype=np.float64)
            for g, op in zip(self.gamma, self.kernels):
                if g != 0.0:
                    if hasattr(op, 'matmat64'):
                        out += float(g * g) * op.matmat64(Z)
                    else:
                        out += float(g * g) * np.asarray(op.matmat(Z), dtype=np.float64)
            return out
        if not self.cores:
            out = np.zeros_like(Z, dtype=np.float64)
        else:
            out = np.asarray(csr_spmm(self.cores[0], Z, backend=self.spmm_backend), dtype=np.float64)
            if len(self.cores) > 1:
                scratch = np.empty_like(out)
                for core in self.cores[1:]:
                    csr_spmm(core, Z, backend=self.spmm_backend, out=scratch)
                    out += scratch
        n = self.n
        bs = self.correction_block_rows
        for g, op in zip(self.gamma, self.kernels):
            if g == 0.0:
                continue
            assert op.d is not None and op.u is not None and op.s is not None
            w = float(g * g)
            uz = _dot_vec_mat64(op.u, Z, bs)
            dz = _dot_vec_mat64(op.d, Z, bs)
            a = -w * uz / n
            b = -w * dz / n
            c = w * (op.s / (n * n)) * dz
            for s in range(0, n, bs):
                e = min(n, s + bs)
                d = np.asarray(op.d[s:e], dtype=np.float64)[:, None]
                u = np.asarray(op.u[s:e], dtype=np.float64)[:, None]
                out[s:e] += d * a[None, :]
                out[s:e] += u * b[None, :]
                out[s:e] += d * c[None, :]
        return out

    def memory_bytes(self) -> int:
        total = 0
        seen_struct: set[tuple[int, int]] = set()
        seen_data: set[int] = set()
        for core in self.cores:
            if id(core.data) not in seen_data:
                total += core.data.nbytes
                seen_data.add(id(core.data))
            key = (id(core.indices), id(core.indptr))
            if key not in seen_struct:
                total += core.indices.nbytes + core.indptr.nbytes
                seen_struct.add(key)
        return int(total)
