from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np

from .support import NeighborGraph, build_neighbor_graph
from .kernels import KernelSpec, build_kernel_bank_grouped, iter_kernel_bank_grouped, default_12_kernel_specs
from .operators import ImplicitCenteredNormalizedKernel
from .simplemkkm import SparseSimpleMKKM
from .data import PreparedArray, prepare_float32_l2
from .runtime import ProgressSink, ResourcePolicy, RunControl
from .profiling import ProfileBook
from .lowmem import resolve_memory_mode, scratch_directory, disk_back_csr_values


@dataclass
class PipelineArtifacts:
    graphs: dict[str, NeighborGraph]
    specs: list[KernelSpec]
    global_s2: float
    model: SparseSimpleMKKM
    prepared_path: str | None = None

    @property
    def graph(self) -> NeighborGraph:
        if len(self.graphs) != 1:
            raise AttributeError('Multiple support graphs are present; use .graphs')
        return next(iter(self.graphs.values()))


@dataclass
class ViewArtifacts:
    name: str
    k: int
    graphs: dict[str, NeighborGraph]
    specs: list[KernelSpec]
    global_s2: float
    kernel_names: list[str]
    prepared_path: str | None = None


@dataclass
class MultiViewPipelineArtifacts:
    views: dict[str, ViewArtifacts]
    model: SparseSimpleMKKM
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


def _effective_specs(specs: Sequence[KernelSpec], support_policy: str) -> tuple[list[KernelSpec], list[str]]:
    policy = support_policy.lower()
    if policy == 'shared_geometry':
        specs_eff = [KernelSpec(s.name, s.kind, s.degree, s.rbf_scale, 'cosine') for s in specs]
        families = ['cosine']
    elif policy == 'ranking_grouped':
        specs_eff = list(specs)
        families = sorted({s.inferred_support_family() for s in specs_eff})
    else:
        raise ValueError("support_policy must be 'shared_geometry' or 'ranking_grouped'")
    return specs_eff, families


def _scoped_progress(parent: ProgressSink, view: str, family: str | None = None) -> ProgressSink:
    if parent.callback is None:
        return ProgressSink(None, parent.control)
    def cb(ev: dict) -> None:
        ev = dict(ev)
        detail = dict(ev.get('detail') or {})
        detail['view'] = view
        if family is not None:
            detail['family'] = family
        ev['detail'] = detail
        parent.callback(ev)
    return ProgressSink(cb, parent.control)


def _intern_sparse_structures(ops: list[ImplicitCenteredNormalizedKernel], specs: Sequence[KernelSpec]) -> None:
    """Share CSR index arrays when kernels in one family truly have identical patterns."""
    bases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for op, spec in zip(ops, specs):
        S = op.sparse_core
        if S is None:
            continue
        fam = spec.inferred_support_family()
        if fam not in bases:
            bases[fam] = (S.indices, S.indptr)
            continue
        bi, bp = bases[fam]
        if S.indices.shape == bi.shape and S.indptr.shape == bp.shape and np.array_equal(S.indices, bi) and np.array_equal(S.indptr, bp):
            S.indices = bi
            S.indptr = bp


def _build_view(
    X: np.ndarray,
    view_name: str,
    k: int,
    specs: Sequence[KernelSpec],
    *,
    search: str,
    ann_backend: str,
    support_mode: str,
    support_policy: str,
    diffusion_power: int,
    block_size: int,
    ef_search: int,
    hnsw_m: int,
    candidate_factor: int,
    query_batch_size: int,
    random_state: int,
    workspace: str | Path | None,
    resource_policy: ResourcePolicy,
    progress: ProgressSink,
    profile: ProfileBook,
    spmm_backend: str = 'auto',
    memory_mode: str = 'auto',
) -> tuple[list[ImplicitCenteredNormalizedKernel], ViewArtifacts]:
    specs_eff, families = _effective_specs(specs, support_policy)

    with profile.stage(f'normalize:{view_name}', view=view_name, n=int(X.shape[0]), d=int(X.shape[1])):
        prepared = prepare_float32_l2(
            X, name=view_name, workspace=workspace, block_rows=max(block_size * 8, 16384),
            progress=_scoped_progress(progress, view_name),
        )
    Xn = prepared.array
    actual_memory_mode = resolve_memory_mode(memory_mode, Xn.shape[0])
    scratch = scratch_directory(workspace) if actual_memory_mode == 'laptop32' else None
    if actual_memory_mode == 'laptop32':
        progress.emit('memory', 0.0, f'32GB safe mode active for {view_name}; offloading kernel CSR values to workspace', view=view_name)

    graphs: dict[str, NeighborGraph] = {}
    for fam in families:
        with profile.stage(f'support:{view_name}:{fam}', view=view_name, family=fam, k=int(k)):
            graph, _ = build_neighbor_graph(
                Xn,
                k=k,
                search=search,
                ann_backend=ann_backend,
                support_mode=support_mode,
                score_mode=fam,
                block_size=block_size,
                ef_search=ef_search,
                hnsw_m=hnsw_m,
                candidate_factor=candidate_factor,
                normalize=False,
                random_state=random_state,
                query_batch_size=query_batch_size,
                resource_policy=resource_policy,
                progress=_scoped_progress(progress, view_name, fam),
            )
            graphs[fam] = graph

    with profile.stage(f'kernels:{view_name}', view=view_name, kernels=len(specs_eff)):
        s2 = float(prepared.global_mean_pairwise_sqdist)
        ops: list[ImplicitCenteredNormalizedKernel] = []
        # Stream one raw sparse kernel at a time into its implicit operator.
        # This removes the v0.7.5 peak in which the whole raw bank coexisted
        # with the whole normalized operator bank.
        for i, (K, spec) in enumerate(iter_kernel_bank_grouped(
            graphs, specs_eff, global_s2=s2, progress=_scoped_progress(progress, view_name)
        )):
            progress.emit('operators', i / max(len(specs_eff), 1), f'Building implicit operator {view_name}:{spec.name}', view=view_name, kernel=spec.name)
            op = ImplicitCenteredNormalizedKernel(
                K, diffusion_power=diffusion_power, dtype=np.float32,
                copy_input=False, spmm_backend=spmm_backend,
            )
            if scratch is not None and op.sparse_core is not None:
                disk_back_csr_values(op, scratch / f'{i:03d}_core_values.npy')
                # qdiag is already encoded in inv_sqrt_q and is not used by matmat.
                op.qdiag = None
            ops.append(op)
            # Intern the new CSR structure immediately; do not retain m copies
            # of identical indices/indptr until the entire bank is built.
            _intern_sparse_structures(ops, specs_eff[:i + 1])
        progress.emit('operators', 1.0, f'Operators ready for {view_name}', view=view_name)

    va = ViewArtifacts(
        name=view_name,
        k=int(k),
        graphs=graphs,
        specs=list(specs_eff),
        global_s2=float(s2),
        kernel_names=[s.name for s in specs_eff],
        prepared_path=prepared.path,
    )
    return ops, va


def fit_sc_simplemkkm(
    X: np.ndarray,
    n_clusters: int,
    k: int = 10,
    search: str = 'ann',
    ann_backend: str = 'auto',
    support_mode: str = 'directed',
    support_policy: str = 'ranking_grouped',
    specs: list[KernelSpec] | None = None,
    diffusion_power: int = 1,
    block_size: int = 2048,
    ef_search: int = 128,
    hnsw_m: int = 32,
    candidate_factor: int = 4,
    query_batch_size: int = 4096,
    random_state: int = 0,
    workspace: str | Path | None = None,
    resource_policy: ResourcePolicy | None = None,
    progress_callback=None,
    control: RunControl | None = None,
    profile: ProfileBook | None = None,
    verbose: bool = False,
    inner_solver: str = 'auto',
    spmm_backend: str = 'auto',
    memory_mode: str = 'auto',
    inner_tol: float = 1e-4,
    inner_max_iter: int = 200,
    outer_tol: float = 1e-5,
    outer_max_iter: int = 100,
    acceptance_tol: float = 1e-6,
    acceptance_max_iter: int = 800,
    acceptance_rtol: float = 1e-8,
    acceptance_atol: float = 1e-10,
    discretization_backend: str = 'auto',
    discretization_block_rows: int = 262_144,
    discretization_max_iter: int = 20,
    discretization_tol: float = 1e-4,
    discretization_init_sample: int = 200_000,
    discretization_init_restarts: int = 5,
    discretization_init_local_iter: int = 8,
    labels_out: np.ndarray | None = None,
    faiss_gpu: bool = False,
    iteration_observer=None,
) -> tuple[np.ndarray, PipelineArtifacts]:
    specs0 = default_12_kernel_specs() if specs is None else list(specs)
    policy = (resource_policy or ResourcePolicy()).normalized()
    progress = ProgressSink(progress_callback, control)
    prof = profile or ProfileBook()
    ops, view = _build_view(
        X, 'view1', k, specs0,
        search=search, ann_backend=ann_backend, support_mode=support_mode,
        support_policy=support_policy, diffusion_power=diffusion_power,
        block_size=block_size, ef_search=ef_search, hnsw_m=hnsw_m,
        candidate_factor=candidate_factor, query_batch_size=query_batch_size,
        random_state=random_state, workspace=workspace, resource_policy=policy,
        progress=progress, profile=prof, spmm_backend=spmm_backend, memory_mode=memory_mode,
    )
    _fused_dir = (str(scratch_directory(workspace)) if resolve_memory_mode(memory_mode, next(iter(ops)).n) == 'laptop32' else None)
    model = SparseSimpleMKKM(ops, n_clusters=n_clusters, random_state=random_state, progress=progress, verbose=verbose, inner_solver=inner_solver, inner_tol=inner_tol, inner_max_iter=inner_max_iter, outer_tol=outer_tol, outer_max_iter=outer_max_iter, acceptance_tol=acceptance_tol, acceptance_max_iter=acceptance_max_iter, acceptance_rtol=acceptance_rtol, acceptance_atol=acceptance_atol, iteration_observer=iteration_observer, spmm_backend=spmm_backend, fused_buffer_dir=_fused_dir)
    with prof.stage('simplemkkm', n=int(X.shape[0]), c=int(n_clusters), kernels=len(ops)):
        labels = model.fit_predict(
            discretization_backend=discretization_backend,
            discretization_block_rows=discretization_block_rows,
            discretization_max_iter=discretization_max_iter,
            discretization_tol=discretization_tol,
            discretization_init_sample=discretization_init_sample,
            discretization_init_restarts=discretization_init_restarts,
            discretization_init_local_iter=discretization_init_local_iter,
            labels_out=labels_out,
            faiss_gpu=faiss_gpu,
        )
    return labels, PipelineArtifacts(view.graphs, view.specs, view.global_s2, model, view.prepared_path)


def _resolve_k(view_names: Sequence[str], k: int | Sequence[int] | Mapping[str, int]) -> dict[str, int]:
    if isinstance(k, Mapping):
        missing = [v for v in view_names if v not in k]
        if missing:
            raise ValueError(f'Missing k for views: {missing}')
        return {v: int(k[v]) for v in view_names}
    if np.isscalar(k):
        return {v: int(k) for v in view_names}
    ks = list(k)
    if len(ks) != len(view_names):
        raise ValueError(f'Need {len(view_names)} k values, got {len(ks)}')
    return {v: int(x) for v, x in zip(view_names, ks)}


def fit_sc_simplemkkm_multiview(
    views: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    n_clusters: int,
    k: int | Sequence[int] | Mapping[str, int] = 10,
    *,
    specs_by_view: Mapping[str, Sequence[KernelSpec]] | None = None,
    search: str = 'ann',
    ann_backend: str = 'auto',
    support_mode: str = 'directed',
    support_policy: str = 'ranking_grouped',
    diffusion_power: int = 1,
    block_size: int = 2048,
    ef_search: int = 128,
    hnsw_m: int = 32,
    candidate_factor: int = 4,
    query_batch_size: int = 4096,
    random_state: int = 0,
    workspace: str | Path | None = None,
    resource_policy: ResourcePolicy | None = None,
    progress_callback=None,
    control: RunControl | None = None,
    profile: ProfileBook | None = None,
    verbose: bool = False,
    inner_solver: str = 'auto',
    spmm_backend: str = 'auto',
    memory_mode: str = 'auto',
    inner_tol: float = 1e-4,
    inner_max_iter: int = 200,
    outer_tol: float = 1e-5,
    outer_max_iter: int = 100,
    acceptance_tol: float = 1e-6,
    acceptance_max_iter: int = 800,
    acceptance_rtol: float = 1e-8,
    acceptance_atol: float = 1e-10,
    discretization_backend: str = 'auto',
    discretization_block_rows: int = 262_144,
    discretization_max_iter: int = 20,
    discretization_tol: float = 1e-4,
    discretization_init_sample: int = 200_000,
    discretization_init_restarts: int = 5,
    discretization_init_local_iter: int = 8,
    labels_out: np.ndarray | None = None,
    faiss_gpu: bool = False,
    iteration_observer=None,
) -> tuple[np.ndarray, MultiViewPipelineArtifacts]:
    if isinstance(views, Mapping):
        view_map = dict(views)
    else:
        view_map = {f'view{i+1}': X for i, X in enumerate(views)}
    if not view_map:
        raise ValueError('At least one view is required')
    view_names = list(view_map)
    n0 = int(np.asarray(view_map[view_names[0]]).shape[0])
    for name, X in view_map.items():
        if np.asarray(X).ndim != 2:
            raise ValueError(f"View '{name}' must be a 2D array")
        if int(np.asarray(X).shape[0]) != n0:
            raise ValueError('All views must contain the same number of samples in the same order')
    if not (1 <= n_clusters < n0):
        raise ValueError('n_clusters must be in [1, n_samples-1]')

    policy = (resource_policy or ResourcePolicy()).normalized()
    progress = ProgressSink(progress_callback, control)
    prof = profile or ProfileBook()
    k_map = _resolve_k(view_names, k)
    all_ops: list[ImplicitCenteredNormalizedKernel] = []
    view_artifacts: dict[str, ViewArtifacts] = {}
    kernel_view: list[str] = []
    kernel_name: list[str] = []

    for iv, name in enumerate(view_names):
        progress.emit('view', iv / len(view_names), f'Preparing view {iv+1}/{len(view_names)}: {name}', view=name)
        specs = list(specs_by_view[name]) if specs_by_view is not None and name in specs_by_view else default_12_kernel_specs()
        ops, va = _build_view(
            view_map[name], name, k_map[name], specs,
            search=search, ann_backend=ann_backend, support_mode=support_mode,
            support_policy=support_policy, diffusion_power=diffusion_power,
            block_size=block_size, ef_search=ef_search, hnsw_m=hnsw_m,
            candidate_factor=candidate_factor, query_batch_size=query_batch_size,
            random_state=random_state + 104729 * iv, workspace=workspace,
            resource_policy=policy, progress=progress, profile=prof, spmm_backend=spmm_backend, memory_mode=memory_mode,
        )
        all_ops.extend(ops)
        view_artifacts[name] = va
        kernel_view.extend([name] * len(va.specs))
        kernel_name.extend([s.name for s in va.specs])
    progress.emit('view', 1.0, 'All views prepared')

    _fused_dir = (str(scratch_directory(workspace)) if resolve_memory_mode(memory_mode, n0) == 'laptop32' else None)
    model = SparseSimpleMKKM(all_ops, n_clusters=n_clusters, random_state=random_state, progress=progress, verbose=verbose, inner_solver=inner_solver, inner_tol=inner_tol, inner_max_iter=inner_max_iter, outer_tol=outer_tol, outer_max_iter=outer_max_iter, acceptance_tol=acceptance_tol, acceptance_max_iter=acceptance_max_iter, acceptance_rtol=acceptance_rtol, acceptance_atol=acceptance_atol, iteration_observer=iteration_observer, spmm_backend=spmm_backend, fused_buffer_dir=_fused_dir)
    with prof.stage('simplemkkm', n=n0, c=int(n_clusters), kernels=len(all_ops)):
        labels = model.fit_predict(
            discretization_backend=discretization_backend,
            discretization_block_rows=discretization_block_rows,
            discretization_max_iter=discretization_max_iter,
            discretization_tol=discretization_tol,
            discretization_init_sample=discretization_init_sample,
            discretization_init_restarts=discretization_init_restarts,
            discretization_init_local_iter=discretization_init_local_iter,
            labels_out=labels_out,
            faiss_gpu=faiss_gpu,
        )
    artifacts = MultiViewPipelineArtifacts(view_artifacts, model, kernel_view, kernel_name, profile=prof.as_dicts())
    return labels, artifacts
