from __future__ import annotations

"""Dense reference implementation of the original SimpleMKKM protocol.

This module intentionally keeps the *dense* representation.  It is the small-/
medium-scale reference engine used for correctness and apples-to-apples tests.
The scalable engine lives in :mod:`scsimplemkkm.pipeline` and replaces only the
representation / numerical realization needed to avoid O(n^2) storage.

Important distinction
---------------------
The released SimpleMKKM code accepts pre-computed base kernels ``KH``.  Kernel
*generation from raw features* is therefore not part of the SimpleMKKM model.
For a self-contained runner we use the same configurable kernel bank as the
scalable engine, then apply the released code's two preprocessing operations:
``kcenter`` followed by ``knorm``.  The optimization semantics are inherited
from the faithful reduced-gradient/golden-section port in ``SparseSimpleMKKM``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import time
import warnings

import numpy as np
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from .kernels import KernelSpec, default_12_kernel_specs, exact_global_mean_pairwise_sqdist
from .data import prepare_float32_l2
from .runtime import ProgressSink, ResourcePolicy
from .profiling import ProfileBook
from .simplemkkm import SparseSimpleMKKM, SubspaceInfo
from .discretize import DiscretizationResult, standard_replicate_lloyd


@dataclass
class DenseKernelOperator:
    """A preprocessed dense base kernel with the operator API used by the core."""

    K: np.ndarray
    name: str = ''

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64, order='C')
        if self.K.ndim != 2 or self.K.shape[0] != self.K.shape[1]:
            raise ValueError('dense kernel must be square')
        self.n = int(self.K.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    @property
    def can_fuse_sparse_core(self) -> bool:
        return False

    def matmat(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self.K @ np.asarray(Z, dtype=np.float64), dtype=np.float64)

    def quadratic_trace(self, H: np.ndarray) -> float:
        HH = np.asarray(H, dtype=np.float64)
        KH = self.K @ HH
        return float(np.sum(HH.astype(np.float64) * KH.astype(np.float64), dtype=np.float64))

    def memory_bytes(self) -> int:
        return int(self.K.nbytes)


@dataclass
class DenseViewArtifacts:
    name: str
    specs: list[KernelSpec]
    global_s2: float
    kernel_names: list[str]
    prepared_path: str | None = None


@dataclass
class DenseMultiViewArtifacts:
    views: dict[str, DenseViewArtifacts]
    model: 'DenseReferenceSimpleMKKM'
    kernel_view: list[str]
    kernel_name: list[str]
    profile: list[dict] | None = None

    def gamma_table(self) -> list[dict[str, float | str]]:
        gamma = np.asarray(self.model.gamma_, dtype=np.float64)
        return [
            {'view': v, 'kernel': k, 'gamma': float(g), 'gamma2': float(g * g)}
            for v, k, g in zip(self.kernel_view, self.kernel_name, gamma)
        ]

    def view_weight_table(self) -> list[dict[str, float | str]]:
        gamma = np.asarray(self.model.gamma_, dtype=np.float64)
        out: list[dict[str, float | str]] = []
        for view in self.views:
            idx = [i for i, v in enumerate(self.kernel_view) if v == view]
            out.append({'view': view, 'sum_gamma': float(np.sum(gamma[idx])), 'sum_gamma2': float(np.sum(gamma[idx] ** 2))})
        return out


def dense_kcenter(K: np.ndarray, ridge: float = 1e-12) -> np.ndarray:
    """Released ``kcenter.m`` semantics for one kernel, in float32 storage."""
    A = np.asarray(K, dtype=np.float64)
    row = np.mean(A, axis=1)
    grand = float(np.mean(row))
    A = A - row[:, None] - row[None, :] + grand
    A = 0.5 * (A + A.T)
    # The released 3-D branch adds 1e-12*I before knorm.
    if ridge:
        A.flat[:: A.shape[0] + 1] += float(ridge)
    return np.asarray(A, dtype=np.float64)


def dense_knorm(K: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """Released ``knorm.m`` semantics with a guard for zero/negative roundoff."""
    A = np.asarray(K, dtype=np.float64)
    d = np.diag(A).copy()
    # A centered PSD kernel should have nonnegative diagonal.  Clamp only tiny
    # numerical negatives; a materially negative value is a protocol error.
    if float(np.min(d)) < -1e-8:
        raise FloatingPointError(f'centered kernel has negative diagonal min={float(np.min(d)):.3e}')
    d = np.maximum(d, eps)
    inv = 1.0 / np.sqrt(d)
    A = (inv[:, None] * A) * inv[None, :]
    A = 0.5 * (A + A.T)
    return np.asarray(A, dtype=np.float64)


def _dense_similarity(X: np.ndarray, *, block_rows: int = 4096, progress: ProgressSink | None = None, view: str = '') -> np.ndarray:
    progress = progress or ProgressSink()
    X = np.asarray(X, dtype=np.float32)
    n = int(X.shape[0])
    # Dense reference deliberately uses float64 arithmetic/storage to mirror
    # the released MATLAB code's default numeric precision.  The normalized
    # input itself remains the same float32 array used by the scalable engine,
    # so representation rather than input rounding is what differs.
    X64 = np.asarray(X, dtype=np.float64)
    S = np.empty((n, n), dtype=np.float64)
    nblocks = (n + block_rows - 1) // block_rows
    XT = X64.T
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows)
        S[s:e] = X64[s:e] @ XT
        progress.emit('dense_kernel:similarity', e / n, f'Dense all-pairs similarity {view}: block {ib+1}/{nblocks}', block_kind='dense_similarity', view=view, rows_start=s, rows_end=e, rows_total=n)
    # Symmetrize once to suppress BLAS roundoff asymmetry.
    S = np.asarray(0.5 * (S + S.T), dtype=np.float64)
    return S


def dense_kernel_bank_from_features(
    X_l2: np.ndarray,
    specs: Iterable[KernelSpec] | None = None,
    *,
    block_rows: int = 4096,
    progress: ProgressSink | None = None,
    view: str = '',
) -> tuple[list[DenseKernelOperator], list[KernelSpec], float]:
    """Generate all-pair dense kernels, then apply official center+knorm."""
    specs = list(default_12_kernel_specs() if specs is None else specs)
    progress = progress or ProgressSink()
    X = np.asarray(X_l2, dtype=np.float32)
    n = int(X.shape[0])
    S = _dense_similarity(X, block_rows=block_rows, progress=progress, view=view)
    global_s2 = exact_global_mean_pairwise_sqdist(X)
    d2 = np.maximum(0.0, 2.0 - 2.0 * S)
    ops: list[DenseKernelOperator] = []
    for i, spec in enumerate(specs):
        kind = spec.kind.lower()
        if kind == 'linear':
            K = S.copy()
        elif kind == 'polyplus':
            if spec.degree is None:
                raise ValueError('polyplus requires degree')
            K = np.power(1.0 + S, int(spec.degree))
        elif kind == 'poly':
            if spec.degree is None:
                raise ValueError('poly requires degree')
            K = np.power(S, int(spec.degree))
        elif kind == 'rbf':
            if spec.rbf_scale is None or spec.rbf_scale <= 0:
                raise ValueError('rbf requires positive rbf_scale')
            denom = float(spec.rbf_scale) * float(global_s2)
            K = np.exp(-d2 / denom)
        elif kind == 'rbf_self_tuned':
            # Dense reference for the Zelnik-Manor--Perona local-scale
            # Gaussian used by the sparse engine.  sigma_i is the distance
            # to the K-th *other* sample, i.e. the diagonal/self match is
            # excluded before selecting the local scale.
            kk = int(spec.local_scale_k or 7)
            if kk < 1:
                raise ValueError('rbf_self_tuned local_scale_k must be >= 1')
            if n <= kk:
                raise ValueError(
                    f'rbf_self_tuned local_scale_k={kk} requires at least {kk + 1} samples; got n={n}'
                )
            work = d2.copy()
            np.fill_diagonal(work, np.inf)
            kth_d2 = np.partition(work, kth=kk - 1, axis=1)[:, kk - 1]
            eps = np.finfo(np.float64).eps
            sigma = np.sqrt(np.maximum(kth_d2, eps))
            denom = np.maximum(sigma[:, None] * sigma[None, :], eps)
            K = np.exp(-d2 / denom)
            np.fill_diagonal(K, 1.0)
            del work, kth_d2, sigma, denom
        else:
            raise ValueError(f'unknown kernel kind: {spec.kind}')
        K = dense_knorm(dense_kcenter(K))
        ops.append(DenseKernelOperator(K, spec.name))
        progress.emit('dense_kernel:preprocess', (i + 1) / len(specs), f'Dense kernel {view}:{spec.name} + kcenter + knorm', view=view, kernel=spec.name)
    # Drop O(n^2) temporary before optimization.
    del S, d2
    return ops, specs, float(global_s2)


class DenseReferenceSimpleMKKM(SparseSimpleMKKM):
    """Dense reference engine.

    The outer optimizer is the same official reduced-gradient/golden-search port
    used by the scalable engine.  The inner problem uses a dense fused kernel
    and an ARPACK largest-algebraic partial eigensolve, mirroring MATLAB
    ``eigs(K, c, 'LA')`` as closely as SciPy permits.
    """

    def __init__(self, kernels: list[DenseKernelOperator], *args, reference_eig_tol: float = 1e-10, **kwargs):
        kwargs = dict(kwargs)
        # ``inner_solver`` is part of the parent API but is not used here.
        kwargs['inner_solver'] = 'reference_eigsh'
        kwargs['dtype'] = np.float64
        super().__init__(kernels, *args, **kwargs)  # type: ignore[arg-type]
        self.reference_eig_tol = float(reference_eig_tol)
        self.reuse_search_for_strict_acceptance = True

    def _subspace_solve(
        self,
        gamma: np.ndarray,
        H0: np.ndarray | None,
        *,
        tol: float | None = None,
        max_iter: int | None = None,
        outer_index: int = 0,
        trial_index: int = 0,
    ) -> tuple[np.ndarray, SubspaceInfo]:
        n = self.n
        c = self.c
        tol_eff = self.reference_eig_tol if tol is None else min(float(tol), self.reference_eig_tol)
        max_iter_eff = max(300, self.inner_max_iter if max_iter is None else int(max_iter))
        self.progress.emit('simplemkkm:fuse', None, 'Materializing dense fused kernel', outer=outer_index, trial=trial_index)
        Kf = np.zeros((n, n), dtype=np.float64)
        for g, op in zip(np.asarray(gamma, dtype=np.float64), self.kernels):
            if g != 0.0:
                Kf += float(g * g) * op.K
        Kf = np.asarray(0.5 * (Kf + Kf.T), dtype=np.float64)
        self.progress.emit('simplemkkm:inner', 0.0, f'Reference eigsh top-{c}: outer {outer_index}, trial {trial_index}', outer=outer_index, trial=trial_index, inner_solver='dense_eigsh')
        nit = 1
        try:
            # The released dense SimpleMKKM recomputes the partial eigensolve
            # at every cost evaluation and does not recycle the previous H as
            # an ARPACK starting vector.  Keeping the reference path un-warmed
            # also avoids pathological exact-eigenvector starts in tiny tests.
            vals, H = eigsh(Kf, k=c, which='LA', tol=tol_eff, maxiter=max_iter_eff)
        except ArpackNoConvergence as e:
            if e.eigenvectors is None or e.eigenvectors.shape[1] < c:
                raise
            vals, H = e.eigenvalues, e.eigenvectors
            nit = max_iter_eff
        order = np.argsort(vals)[::-1]
        vals = np.asarray(vals[order], dtype=np.float64)
        H = self._orth(np.asarray(H[:, order], dtype=np.float64))
        AH = np.asarray(Kf @ H, dtype=np.float64)
        T = H.T @ AH
        # Dense reference is small/medium scale: compute the residual explicitly
        # in float64 instead of the cancellation-prone norm identity.
        R = AH - H @ T
        rel = float(np.linalg.norm(R, ord='fro') / max(np.linalg.norm(AH, ord='fro'), np.finfo(float).eps))
        obj = float(np.trace(T))
        self.progress.emit('simplemkkm:inner', 1.0, f'Reference eigsh complete: residual {rel:.2e}', outer=outer_index, trial=trial_index, inner_iter=nit, residual=rel, inner_objective=obj, inner_solver='dense_eigsh', fused_operator_bytes=int(Kf.nbytes))
        return H, SubspaceInfo(nit, rel, obj, int(Kf.nbytes))

    def _strict_subspace_solve(
        self, gamma: np.ndarray, H0: np.ndarray | None, *, outer_index: int, trial_index: int
    ) -> tuple[np.ndarray, SubspaceInfo]:
        # Dense reference already evaluates every state in float64 with the
        # requested reference eigensolver tolerance.  Reuse that exact path.
        return self._subspace_solve(
            gamma, H0, tol=min(self.acceptance_tol, self.reference_eig_tol),
            max_iter=max(self.acceptance_max_iter, self.inner_max_iter),
            outer_index=outer_index, trial_index=trial_index,
        )

    def fit_predict(
        self,
        *,
        discretization_backend: str = 'standard_replicate_lloyd',
        discretization_block_rows: int = 262_144,
        discretization_max_iter: int = 100,
        discretization_tol: float = 1e-4,
        discretization_init_sample: int = 200_000,
        discretization_init_restarts: int = 10,
        discretization_init_local_iter: int = 8,
        labels_out: np.ndarray | None = None,
        faiss_gpu: bool = False,
    ) -> np.ndarray:
        self.fit()
        self.progress.emit('postprocess:kmeans', 0.0, 'Standard row-normalized H + replicated Lloyd')
        if discretization_backend in {'standard', 'standard_replicate_lloyd', 'official'}:
            res = standard_replicate_lloyd(
                self.embedding_, self.c,
                replicates=max(1, int(discretization_init_restarts)),
                block_rows=discretization_block_rows,
                max_iter=max(1, int(discretization_max_iter)),
                tol=discretization_tol,
                random_state=self.random_state,
                labels_out=labels_out,
                progress=self.progress,
            )
        else:
            # Permit the same scalable discretizers for controlled ablations.
            from .discretize import discretize_embedding
            res = discretize_embedding(
                self.embedding_, self.c,
                backend=discretization_backend,
                block_rows=discretization_block_rows,
                max_iter=discretization_max_iter,
                tol=discretization_tol,
                init_sample=discretization_init_sample,
                init_restarts=discretization_init_restarts,
                init_local_iter=discretization_init_local_iter,
                random_state=self.random_state,
                labels_out=labels_out,
                faiss_gpu=faiss_gpu,
                progress=self.progress,
            )
        self.discretization_result_ = res
        self.postprocess_backend_ = res.backend
        self.progress.emit('postprocess:kmeans', 1.0, f'{res.backend} complete', inertia=res.inertia, n_iter=res.n_iter)
        return res.labels


def fit_dense_reference_multiview(
    views: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    n_clusters: int,
    *,
    specs_by_view: Mapping[str, Sequence[KernelSpec]] | None = None,
    random_state: int = 0,
    workspace: str | Path | None = None,
    resource_policy: ResourcePolicy | None = None,
    progress_callback=None,
    control=None,
    profile: ProfileBook | None = None,
    verbose: bool = False,
    dense_block_rows: int = 4096,
    outer_tol: float = 1e-5,
    outer_max_iter: int = 100,
    reference_eig_tol: float = 1e-10,
    discretization_backend: str = 'standard_replicate_lloyd',
    discretization_block_rows: int = 262_144,
    discretization_max_iter: int = 100,
    discretization_tol: float = 1e-4,
    discretization_init_restarts: int = 10,
    labels_out: np.ndarray | None = None,
    iteration_observer=None,
) -> tuple[np.ndarray, DenseMultiViewArtifacts]:
    if isinstance(views, Mapping):
        view_map = dict(views)
    else:
        view_map = {f'view{i+1}': X for i, X in enumerate(views)}
    if not view_map:
        raise ValueError('at least one view is required')
    names = list(view_map)
    n = int(np.asarray(view_map[names[0]]).shape[0])
    for name, X in view_map.items():
        if np.asarray(X).ndim != 2 or int(np.asarray(X).shape[0]) != n:
            raise ValueError('all views must be 2D with identical sample counts/order')
    progress = ProgressSink(progress_callback, control)
    prof = profile or ProfileBook()
    all_ops: list[DenseKernelOperator] = []
    vas: dict[str, DenseViewArtifacts] = {}
    kernel_view: list[str] = []
    kernel_name: list[str] = []
    for iv, name in enumerate(names):
        specs = list(specs_by_view[name]) if specs_by_view is not None and name in specs_by_view else default_12_kernel_specs()
        with prof.stage(f'normalize:{name}', view=name, n=n, d=int(np.asarray(view_map[name]).shape[1])):
            prep = prepare_float32_l2(view_map[name], name=f'dense_{name}', workspace=workspace, block_rows=max(16384, dense_block_rows * 4), progress=progress)
        with prof.stage(f'dense_kernels:{name}', view=name, kernels=len(specs)):
            ops, specs2, s2 = dense_kernel_bank_from_features(prep.array, specs, block_rows=dense_block_rows, progress=progress, view=name)
        all_ops.extend(ops)
        vas[name] = DenseViewArtifacts(name, specs2, s2, [s.name for s in specs2], prep.path)
        kernel_view.extend([name] * len(specs2)); kernel_name.extend([s.name for s in specs2])
    model = DenseReferenceSimpleMKKM(
        all_ops, n_clusters=n_clusters, random_state=random_state, progress=progress,
        verbose=verbose, inner_tol=reference_eig_tol, inner_max_iter=500,
        outer_tol=outer_tol, outer_max_iter=outer_max_iter,
        reference_eig_tol=reference_eig_tol, iteration_observer=iteration_observer,
    )
    with prof.stage('simplemkkm_reference_dense', n=n, c=int(n_clusters), kernels=len(all_ops)):
        labels = model.fit_predict(
            discretization_backend=discretization_backend,
            discretization_block_rows=discretization_block_rows,
            discretization_max_iter=discretization_max_iter,
            discretization_tol=discretization_tol,
            discretization_init_restarts=discretization_init_restarts,
            labels_out=labels_out,
        )
    return labels, DenseMultiViewArtifacts(vas, model, kernel_view, kernel_name, profile=prof.as_dicts())
