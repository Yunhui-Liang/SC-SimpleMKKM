from __future__ import annotations

"""Operator-level bridge from SC-SimpleMKKM to multiple-graph spectral clustering.

The production SC-SimpleMKKM representation supplies centered/diagonally
normalized PSD operators K_r that were induced from sparse view graphs.  For a
fixed weight vector gamma, the SimpleMKKM inner problem is exactly the leading
c-dimensional eigenspace of

    K_gamma = sum_r gamma_r^2 K_r.

Hence, when one graph-induced operator is supplied per view, a fixed-gamma
state is precisely spectral clustering on a weighted fused graph-induced PSD
operator.  This module exposes diagnostic baselines on *the same supplied
operators* so experiments can isolate graph fusion from outer weight learning.
"""

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from .operators import ImplicitCenteredNormalizedKernel, FusedCenteredOperator
from .discretize import discretize_embedding
from .evaluation import external_clustering_metrics


@dataclass
class GraphSpectralResult:
    gamma: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    objective: float
    residual: float
    solver: str
    eig_seconds: float
    discretize_seconds: float
    metrics: dict | None = None


def _materialize_fused_float64(
    kernels: list[ImplicitCenteredNormalizedKernel], gamma: np.ndarray
) -> np.ndarray:
    """Validation-only explicit materialization of the *same* fused operator."""
    fused = FusedCenteredOperator(kernels, gamma, dtype=np.float32)
    n = fused.n
    eye = np.eye(n, dtype=np.float64)
    K = fused.matmat64(eye)
    return 0.5 * (K + K.T)


def fixed_gamma_embedding(
    kernels: list[ImplicitCenteredNormalizedKernel],
    n_clusters: int,
    gamma: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 1200,
    dense_validation_n: int = 3000,
) -> tuple[np.ndarray, float, float, str, float]:
    """Leading eigenspace for a fixed fusion of the supplied operators.

    This does not optimize gamma.  It is intended for equal-fusion and
    single-view graph baselines that use exactly the same SC operators as the
    SimpleMKKM run.
    """
    if not kernels:
        raise ValueError('at least one operator is required')
    n = kernels[0].n
    c = int(n_clusters)
    g = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if g.size != len(kernels):
        raise ValueError('gamma length mismatch')
    if np.any(g < 0) or not np.all(np.isfinite(g)) or float(np.sum(g)) <= 0:
        raise ValueError('gamma must be finite, nonnegative, and nonzero')
    # Scale of gamma does not change the eigenspace. Normalize for a stable
    # objective convention that mirrors SimpleMKKM simplex weights.
    g = g / np.sum(g)
    fused = FusedCenteredOperator(kernels, g, dtype=np.float32)
    t0 = time.perf_counter()

    # Small-n/high-c closure uses an explicit solve of the identical operator,
    # matching v0.7.4 accepted-state validation philosophy.
    if n <= int(dense_validation_n) and c >= 64:
        K = _materialize_fused_float64(kernels, g)
        vals, H = eigh(K, subset_by_index=[n-c, n-1], driver='evr', check_finite=False, overwrite_a=True)
        solver = 'dense_eigh_validation'
    else:
        def mv(x):
            return fused.matmat64(np.asarray(x, dtype=np.float64).reshape(n, 1))[:, 0]
        def mm(X):
            return fused.matmat64(np.asarray(X, dtype=np.float64))
        A = LinearOperator((n, n), matvec=mv, matmat=mm, dtype=np.dtype(np.float64))
        ncv = min(n, max(2*c + 20, c + 2))
        try:
            vals, H = eigsh(A, k=c, which='LA', tol=max(tol * 0.1, 1e-10), maxiter=max_iter, ncv=ncv)
            solver = 'eigsh'
        except ArpackNoConvergence as exc:
            He = getattr(exc, 'eigenvectors', None)
            if He is None or He.shape[1] < c:
                raise
            H = np.asarray(He[:, -c:], dtype=np.float64)
            vals = np.empty(c, dtype=np.float64)
            solver = 'eigsh_partial'

    H, _ = np.linalg.qr(np.asarray(H, dtype=np.float64), mode='reduced')
    AH = fused.matmat64(H)
    T = H.T @ AH
    R = AH - H @ T
    denom = max(float(np.linalg.norm(AH, ord='fro')), np.finfo(np.float64).eps)
    residual = float(np.linalg.norm(R, ord='fro') / denom)
    objective = float(np.trace(T))
    return H.astype(np.float32), objective, residual, solver, time.perf_counter() - t0


def fixed_gamma_spectral_clustering(
    kernels: list[ImplicitCenteredNormalizedKernel],
    n_clusters: int,
    gamma: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    random_state: int = 0,
    discretization_backend: str = 'auto',
) -> GraphSpectralResult:
    H, obj, resid, solver, eig_seconds = fixed_gamma_embedding(
        kernels, n_clusters, gamma
    )
    t0 = time.perf_counter()
    disc = discretize_embedding(
        H, n_clusters, backend=discretization_backend, random_state=random_state
    )
    dsec = time.perf_counter() - t0
    metrics = None if y_true is None else external_clustering_metrics(np.asarray(y_true).reshape(-1), disc.labels)
    return GraphSpectralResult(
        gamma=np.asarray(gamma, dtype=np.float64) / np.sum(gamma),
        embedding=H,
        labels=np.asarray(disc.labels, dtype=np.int32),
        objective=obj,
        residual=resid,
        solver=solver,
        eig_seconds=eig_seconds,
        discretize_seconds=dsec,
        metrics=metrics,
    )


def uniform_graph_fusion(
    kernels: list[ImplicitCenteredNormalizedKernel],
    n_clusters: int,
    *,
    y_true: np.ndarray | None = None,
    random_state: int = 0,
) -> GraphSpectralResult:
    """Equal-weight spectral fusion on the same graph-induced operators."""
    m = len(kernels)
    return fixed_gamma_spectral_clustering(
        kernels, n_clusters, np.full(m, 1.0 / m, dtype=np.float64),
        y_true=y_true, random_state=random_state,
    )


def single_graph_spectral(
    kernels: list[ImplicitCenteredNormalizedKernel],
    n_clusters: int,
    index: int,
    *,
    y_true: np.ndarray | None = None,
    random_state: int = 0,
) -> GraphSpectralResult:
    """Spectral clustering using exactly one supplied graph-induced operator."""
    if not (0 <= int(index) < len(kernels)):
        raise IndexError(index)
    g = np.zeros(len(kernels), dtype=np.float64)
    g[int(index)] = 1.0
    return fixed_gamma_spectral_clustering(
        kernels, n_clusters, g, y_true=y_true, random_state=random_state
    )


def recover_sparse_psd_graph_operator(op: ImplicitCenteredNormalizedKernel):
    """Recover the pre-centering sparse PSD graph operator P for q=1.

    For q=1 the production object stores sparse_core = D P D with
    D=diag(inv_sqrt_q).  This diagnostic inverse scaling is exact up to stored
    floating-point coefficients and does not materialize any dense matrix.
    """
    if not op.can_fuse_sparse_core or op.sparse_core is None or op.d is None:
        raise ValueError('raw graph recovery is available only for q=1 operators')
    S = op.sparse_core.copy().tocsr()
    dinv = 1.0 / np.maximum(np.asarray(op.d, dtype=np.float64), np.finfo(np.float64).eps)
    # P_ij = (DPD)_ij / (D_i D_j)
    rows = np.repeat(np.arange(op.n, dtype=np.int64), np.diff(S.indptr))
    S.data = (S.data.astype(np.float64) * dinv[rows] * dinv[S.indices]).astype(np.float64)
    S = 0.5 * (S + S.T)
    S.sum_duplicates(); S.sort_indices(); S.eliminate_zeros()
    return S


def raw_uniform_graph_spectral_clustering(
    kernels: list[ImplicitCenteredNormalizedKernel],
    n_clusters: int,
    *,
    y_true: np.ndarray | None = None,
    random_state: int = 0,
    tol: float = 1e-8,
    max_iter: int = 3000,
) -> GraphSpectralResult:
    """Uniform spectral fusion of the pre-centering graph PSD operators P_v.

    This is the adjacency/operator-level multi-graph baseline.  It deliberately
    omits SC-SimpleMKKM's per-operator centering and diagonal normalization, so
    comparing it with ``uniform_graph_fusion`` isolates the representation step
    P_v -> Khat_v while holding supports/affinities fixed.
    """
    from scipy import sparse
    if not kernels:
        raise ValueError('at least one operator is required')
    P = None
    for op in kernels:
        Pi = recover_sparse_psd_graph_operator(op)
        P = Pi if P is None else (P + Pi)
    P = (P * (1.0 / len(kernels))).tocsr()
    n = P.shape[0]; c = int(n_clusters)
    t0=time.perf_counter()
    vals,H=eigsh(P,k=c,which='LA',tol=max(tol*0.1,1e-10),maxiter=max_iter,ncv=min(n,max(2*c+20,c+2)))
    H,_=np.linalg.qr(np.asarray(H,dtype=np.float64),mode='reduced')
    AH=np.asarray(P@H,dtype=np.float64); T=H.T@AH; R=AH-H@T
    residual=float(np.linalg.norm(R,'fro')/max(float(np.linalg.norm(AH,'fro')),np.finfo(float).eps))
    objective=float(np.trace(T)); es=time.perf_counter()-t0
    t1=time.perf_counter(); disc=discretize_embedding(H.astype(np.float32),c,backend='auto',random_state=random_state); ds=time.perf_counter()-t1
    metrics=None if y_true is None else external_clustering_metrics(np.asarray(y_true).reshape(-1),disc.labels)
    g=np.full(len(kernels),1.0/len(kernels),dtype=np.float64)
    return GraphSpectralResult(g,H.astype(np.float32),np.asarray(disc.labels,dtype=np.int32),objective,residual,'eigsh_raw_graph',es,ds,metrics)
