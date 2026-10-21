from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np
from scipy import sparse

from .support import NeighborGraph
from .runtime import ProgressSink


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kind: str
    degree: int | None = None
    rbf_scale: float | None = None
    support_family: str | None = None
    local_scale_k: int | None = None

    def inferred_support_family(self) -> str:
        if self.support_family is not None:
            return self.support_family
        k = self.kind.lower()
        if k == 'poly' and self.degree is not None and self.degree % 2 == 0:
            return 'abs_cosine'
        return 'cosine'


def historical_12_kernel_specs() -> list[KernelSpec]:
    out = [
        KernelSpec('linear', 'linear', support_family='cosine'),
        KernelSpec('polyplus2', 'polyplus', degree=2, support_family='cosine'),
        KernelSpec('polyplus4', 'polyplus', degree=4, support_family='cosine'),
        KernelSpec('poly2', 'poly', degree=2, support_family='abs_cosine'),
        KernelSpec('poly4', 'poly', degree=4, support_family='abs_cosine'),
    ]
    for e in range(-3, 4):
        out.append(KernelSpec(f'rbf_2^{e}', 'rbf', rbf_scale=2.0**e, support_family='cosine'))
    return out


def default_12_kernel_specs() -> list[KernelSpec]:
    """Backward-compatible alias for the historical 12-kernel bank."""
    return historical_12_kernel_specs()


def rbf7_kernel_specs() -> list[KernelSpec]:
    """Paper-default seven RBF kernels centered at alpha=1.

    Each view has its own exact global mean squared pairwise distance s2_v, so
    alpha is dimensionless and the effective bandwidth is view-specific even
    when the same alpha bank is shared across views.
    """
    return [
        KernelSpec(f'rbf_2^{e}', 'rbf', rbf_scale=2.0**e, support_family='cosine')
        for e in range(-3, 4)
    ]


def rbf1_kernel_specs(scale: float = 1.0) -> list[KernelSpec]:
    """Single representative RBF kernel.

    The default alpha=1 is the center member of RBF7 and is the recommended
    no-label/no-tuning ablation.  The actual denominator is alpha * s2_v, where
    s2_v is computed separately for every view.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('single RBF scale must be positive and finite')
    if np.isclose(scale, 1.0):
        name = 'rbf_center'
    else:
        name = f'rbf_alpha_{scale:g}'
    return [KernelSpec(name, 'rbf', rbf_scale=scale, support_family='cosine')]


def rbf_self_tuned_kernel_specs(local_scale_k: int = 7) -> list[KernelSpec]:
    """Single locally scaled Gaussian affinity (Zelnik-Manor--Perona).

    For sample i, sigma_i is the distance to its ``local_scale_k``-th
    neighbor.  Retained-edge affinity is
        exp(-||x_i-x_j||^2 / (sigma_i sigma_j)).
    This removes the global Gaussian bandwidth search.  The historical
    self-tuning spectral-clustering default is K=7.  In SC-SimpleMKKM the
    resulting nonnegative affinity is converted to the same sparse PSD base
    operator used by every other kernel preset.
    """
    kk = int(local_scale_k)
    if kk < 1:
        raise ValueError('local_scale_k must be >= 1')
    return [KernelSpec('rbf_self_tuned', 'rbf_self_tuned', local_scale_k=kk, support_family='cosine')]


def kernel_preset_specs(name: str, *, single_rbf_scale: float = 1.0, local_scale_k: int = 7) -> list[KernelSpec]:
    """Resolve a compact kernel preset name to explicit KernelSpec objects.

    Supported names: historical12, rbf7, rbf1/rbf1_center, rbf_self_tuned.
    """
    key = str(name or '').strip().lower().replace('-', '_')
    if key in {'historical12', 'historical_12', 'default12', 'default_12'}:
        return historical_12_kernel_specs()
    if key in {'rbf7', 'rbf_7'}:
        return rbf7_kernel_specs()
    if key in {'rbf1', 'rbf_1', 'rbf1_center', 'rbf_center', 'single_rbf'}:
        return rbf1_kernel_specs(single_rbf_scale)
    if key in {'rbf_self_tuned', 'rbf_selftuned', 'self_tuned_rbf', 'selftuned_rbf', 'local_rbf', 'local_scale_rbf', 'rbf_local7'}:
        return rbf_self_tuned_kernel_specs(local_scale_k)
    raise ValueError(f"unknown kernel preset '{name}'; use historical12, rbf7, rbf1, or rbf_self_tuned")


def kernel_spec_to_dict(spec: KernelSpec) -> dict:
    return {
        'name': spec.name,
        'kind': spec.kind,
        'degree': spec.degree,
        'rbf_scale': spec.rbf_scale,
        'local_scale_k': spec.local_scale_k,
        'support_family': spec.support_family,
        'resolved_support_family': spec.inferred_support_family(),
    }


def kernel_spec_from_dict(obj: Mapping) -> KernelSpec:
    if not isinstance(obj, Mapping):
        raise TypeError('kernel spec must be a mapping')
    name = str(obj.get('name') or '').strip()
    kind = str(obj.get('kind') or '').strip().lower()
    if not name:
        raise ValueError('kernel name cannot be empty')
    if kind not in {'linear', 'polyplus', 'poly', 'rbf', 'rbf_self_tuned'}:
        raise ValueError(f"unsupported kernel kind '{kind}' for {name}")
    degree = obj.get('degree')
    rbf_scale = obj.get('rbf_scale')
    local_scale_k = obj.get('local_scale_k')
    family = obj.get('support_family')
    if family in ('', 'auto', None):
        family = None
    else:
        family = str(family)
        if family not in {'cosine', 'abs_cosine'}:
            raise ValueError(f"unsupported support family '{family}' for {name}")
    if kind in {'polyplus', 'poly'}:
        if degree is None:
            raise ValueError(f'{name}: polynomial kernel requires degree')
        degree = int(degree)
        if degree < 1:
            raise ValueError(f'{name}: degree must be >= 1')
    else:
        degree = None
    if kind == 'rbf':
        if rbf_scale is None:
            raise ValueError(f'{name}: RBF kernel requires rbf_scale')
        rbf_scale = float(rbf_scale)
        if not np.isfinite(rbf_scale) or rbf_scale <= 0:
            raise ValueError(f'{name}: rbf_scale must be positive and finite')
        local_scale_k = None
    elif kind == 'rbf_self_tuned':
        rbf_scale = None
        local_scale_k = 7 if local_scale_k is None else int(local_scale_k)
        if local_scale_k < 1:
            raise ValueError(f'{name}: local_scale_k must be >= 1')
    else:
        rbf_scale = None
        local_scale_k = None
    return KernelSpec(name=name, kind=kind, degree=degree, rbf_scale=rbf_scale, local_scale_k=local_scale_k, support_family=family)


def kernel_specs_from_rows(rows: Iterable[Mapping] | None) -> list[KernelSpec]:
    if rows is None:
        return default_12_kernel_specs()
    specs: list[KernelSpec] = []
    names: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping) and not bool(row.get('enabled', True)):
            continue
        spec = kernel_spec_from_dict(row)
        if spec.name in names:
            raise ValueError(f"duplicate kernel name '{spec.name}'")
        names.add(spec.name)
        specs.append(spec)
    if not specs:
        raise ValueError('kernel bank must contain at least one enabled kernel')
    return specs


def exact_global_mean_pairwise_sqdist(X: np.ndarray, block_rows: int = 65536) -> float:
    """Exact O(nd) identity, blockwise and memmap-safe."""
    X = np.asarray(X)
    n, d = X.shape
    sum_vec = np.zeros(d, dtype=np.float64)
    sum_norm2 = 0.0
    for s in range(0, n, block_rows):
        b = np.asarray(X[s:min(n, s + block_rows)], dtype=np.float64)
        sum_vec += np.sum(b, axis=0, dtype=np.float64)
        sum_norm2 += float(np.sum(np.einsum('ij,ij->i', b, b), dtype=np.float64))
    mu = sum_vec / max(n, 1)
    s2 = 2.0 * (sum_norm2 / max(n, 1) - float(mu @ mu))
    return max(s2, np.finfo(np.float32).eps)


def edge_values(graph: NeighborGraph, spec: KernelSpec, global_s2: float, block_rows: int = 262_144) -> np.ndarray:
    """Evaluate one kernel only on retained relations with bounded temporaries."""
    sim = graph.similarity.astype(np.float32, copy=False)
    kind = spec.kind.lower()
    if kind == 'linear':
        return sim.copy()
    if kind == 'polyplus':
        if spec.degree is None:
            raise ValueError('polyplus needs degree')
        vals = sim.copy()
        vals += np.float32(1.0)
        np.power(vals, spec.degree, out=vals)
        return vals
    if kind == 'poly':
        if spec.degree is None:
            raise ValueError('poly needs degree')
        vals = sim.copy()
        np.power(vals, spec.degree, out=vals)
        return vals
    if kind == 'rbf':
        if spec.rbf_scale is None or spec.rbf_scale <= 0:
            raise ValueError('rbf needs positive rbf_scale')
        # Reuse one nnz-sized float32 output buffer rather than materializing
        # graph.dist2 plus several exponent temporaries.
        vals = sim.copy()
        vals *= np.float32(-2.0)
        vals += np.float32(2.0)
        vals *= np.float32(-1.0 / float(spec.rbf_scale * global_s2))
        np.exp(vals, out=vals)
        return vals
    if kind == 'rbf_self_tuned':
        kk = int(spec.local_scale_k or 7)
        if graph.support_mode != 'directed':
            raise ValueError('rbf_self_tuned currently requires support_mode=directed so the K-th-neighbor scale is well-defined')
        if graph.k < kk:
            raise ValueError(f'rbf_self_tuned local_scale_k={kk} requires graph k >= {kk}; got k={graph.k}')
        counts = np.diff(graph.indptr)
        if counts.size != graph.n_samples or np.any(counts < kk):
            raise ValueError('directed support rows do not contain enough neighbors for self-tuned local scaling')
        kth_pos = graph.indptr[:-1] + (kk - 1)
        sigma2 = np.maximum(
            np.float64(2.0) - np.float64(2.0) * graph.similarity[kth_pos].astype(np.float64),
            np.finfo(np.float32).eps,
        )
        sigma = np.sqrt(sigma2)
        vals = np.empty(graph.nnz, dtype=np.float32)
        bs = max(1, int(block_rows))
        for srow in range(0, graph.n_samples, bs):
            erow = min(graph.n_samples, srow + bs)
            p0 = int(graph.indptr[srow]); p1 = int(graph.indptr[erow])
            if p1 <= p0:
                continue
            local_counts = np.diff(graph.indptr[srow:erow + 1])
            sigma_i = np.repeat(sigma[srow:erow], local_counts)
            sigma_j = sigma[graph.indices[p0:p1]]
            denom = np.maximum(sigma_i * sigma_j, np.finfo(np.float32).eps)
            d2 = np.float64(2.0) - np.float64(2.0) * graph.similarity[p0:p1].astype(np.float64)
            block = np.exp(-d2 / denom)
            vals[p0:p1] = block.astype(np.float32)
        return vals
    raise ValueError(f'Unknown kernel kind: {spec.kind}')


def _scale_csr_columns_inplace(A: sparse.csr_matrix, scale: np.ndarray, block_nnz: int = 8_000_000) -> None:
    nnz = int(A.nnz)
    step = max(1, int(block_nnz))
    for p in range(0, nnz, step):
        q = min(nnz, p + step)
        A.data[p:q] *= scale[A.indices[p:q]]


def _scale_csr_rows_inplace(A: sparse.csr_matrix, scale: np.ndarray, block_rows: int = 262_144) -> None:
    n = int(A.shape[0])
    bs = max(1, int(block_rows))
    for srow in range(0, n, bs):
        erow = min(n, srow + bs)
        p0 = int(A.indptr[srow]); p1 = int(A.indptr[erow])
        if p1 <= p0:
            continue
        counts = np.diff(A.indptr[srow:erow + 1])
        row_scale = np.repeat(scale[srow:erow], counts)
        A.data[p0:p1] *= row_scale


def build_sparse_psd_kernel(
    graph: NeighborGraph,
    values: np.ndarray,
    eps: float = 1e-8,
    *,
    block_rows: int = 262_144,
    block_nnz: int = 8_000_000,
) -> sparse.csr_matrix:
    """Graph -> PSD core with no sparse-diagonal materialization.

    This is algebraically the same production construction as v0.7.5:
    column-normalize W, symmetrize, symmetric-degree-normalize, and return
    P=(I+S)/2.  v0.7.6 performs scaling in place and injects the identity via
    ``setdiag`` so peak memory is governed by relation storage rather than
    temporary diagonal sparse matrices.
    """
    n = graph.n_samples
    values = np.asarray(values, dtype=np.float32)
    W = sparse.csr_matrix((values, graph.indices, graph.indptr), shape=(n, n), dtype=np.float32, copy=False)
    colsum = np.asarray(W.sum(axis=0)).ravel().astype(np.float32)
    inv_col = np.float32(1.0) / np.maximum(colsum, np.float32(eps))
    _scale_csr_columns_inplace(W, inv_col, block_nnz=block_nnz)

    A = (W + W.T).tocsr()
    A.data *= np.float32(0.5)
    A.sum_duplicates(); A.sort_indices(); A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    dinv = np.float32(1.0) / np.sqrt(np.maximum(deg, np.float32(eps)))
    _scale_csr_columns_inplace(A, dinv, block_nnz=block_nnz)
    _scale_csr_rows_inplace(A, dinv, block_rows=block_rows)

    # A is now S=D^-1/2 A D^-1/2.  Form P=(I+S)/2 without sparse.eye.
    A.data *= np.float32(0.5)
    diag = A.diagonal().astype(np.float32, copy=False)
    A.setdiag(diag + np.float32(0.5))
    A.sum_duplicates(); A.sort_indices(); A.eliminate_zeros()
    return A


def iter_kernel_bank_grouped(
    graphs: Mapping[str, NeighborGraph],
    specs: Iterable[KernelSpec],
    *,
    global_s2: float,
    progress: ProgressSink | None = None,
):
    """Yield one sparse PSD kernel at a time to bound construction peak RAM."""
    specs = list(specs)
    progress = progress or ProgressSink()
    for i, spec in enumerate(specs):
        family = spec.inferred_support_family()
        if family not in graphs:
            raise KeyError(f"Missing support family '{family}' for kernel '{spec.name}'")
        graph = graphs[family]
        vals = edge_values(graph, spec, global_s2)
        vmin = float(np.min(vals)) if vals.size else 0.0
        if vmin < -1e-7:
            raise ValueError(
                f"Kernel '{spec.name}' has negative retained edge weights (min={vmin:.3g}). "
                'The graph->PSD construction assumes nonnegative affinities. '
                'Use nonnegative local affinities or a different kernel bank for this dataset.'
            )
        K = build_sparse_psd_kernel(graph, vals)
        progress.emit('kernel_bank', (i + 1) / len(specs), f'Kernel {i+1}/{len(specs)}: {spec.name}', kernel=spec.name)
        yield K, spec


def build_kernel_bank_grouped(
    graphs: Mapping[str, NeighborGraph],
    X_for_scale: np.ndarray,
    specs: Iterable[KernelSpec] | None = None,
    *,
    global_s2: float | None = None,
    progress: ProgressSink | None = None,
) -> tuple[list[sparse.csr_matrix], list[KernelSpec], float]:
    """Compatibility API that materializes the full bank.

    The scalable pipeline itself uses ``iter_kernel_bank_grouped`` so raw
    kernels are converted to implicit operators one at a time.
    """
    specs = list(default_12_kernel_specs() if specs is None else specs)
    global_s2 = exact_global_mean_pairwise_sqdist(X_for_scale) if global_s2 is None else float(global_s2)
    kernels: list[sparse.csr_matrix] = []
    for K, _ in iter_kernel_bank_grouped(graphs, specs, global_s2=global_s2, progress=progress):
        kernels.append(K)
    return kernels, specs, global_s2

def build_kernel_bank(graph: NeighborGraph, X_for_scale: np.ndarray, specs: Iterable[KernelSpec] | None = None) -> tuple[list[sparse.csr_matrix], list[KernelSpec], float]:
    specs = list(default_12_kernel_specs() if specs is None else specs)
    graphs = {spec.inferred_support_family(): graph for spec in specs}
    return build_kernel_bank_grouped(graphs, X_for_scale, specs)
