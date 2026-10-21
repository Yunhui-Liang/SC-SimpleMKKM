from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from .runtime import ProgressSink


@dataclass
class DiscretizationResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int
    backend: str
    seconds: float


def _normalized_block(H: np.ndarray, s: int, e: int) -> np.ndarray:
    X = np.asarray(H[s:e], dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, np.float32(1e-12))


def _sample_normalized(H: np.ndarray, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    n = int(H.shape[0])
    m = min(n, max(1, int(sample_size)))
    if m == n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=m, replace=False)
    X = np.asarray(H[idx], dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, np.float32(1e-12))


def _kmeanspp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    if k > n:
        raise ValueError('k cannot exceed initialization sample size')
    centers = np.empty((k, d), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = X[first]
    dist2 = np.sum((X - centers[0]) ** 2, axis=1, dtype=np.float64)
    for j in range(1, k):
        total = float(np.sum(dist2, dtype=np.float64))
        if not np.isfinite(total) or total <= 1e-30:
            centers[j] = X[int(rng.integers(0, n))]
        else:
            r = float(rng.random()) * total
            idx = int(np.searchsorted(np.cumsum(dist2, dtype=np.float64), r, side='right'))
            idx = min(idx, n - 1)
            centers[j] = X[idx]
        dnew = np.sum((X - centers[j]) ** 2, axis=1, dtype=np.float64)
        dist2 = np.minimum(dist2, dnew)
    return centers


def _assign_block(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Squared distances without constructing a 3-D tensor.
    x2 = np.sum(X * X, axis=1, keepdims=True, dtype=np.float32)
    c2 = np.sum(centers * centers, axis=1, dtype=np.float32)[None, :]
    dist = x2 + c2 - 2.0 * (X @ centers.T)
    labels = np.argmin(dist, axis=1).astype(np.int32)
    best = dist[np.arange(X.shape[0]), labels].astype(np.float64)
    return labels, best


def _refine_sample_centers(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Small in-memory Lloyd refinement used only to select a robust start."""
    centers = np.asarray(centers, dtype=np.float32).copy()
    k, d = centers.shape
    prev = math.inf
    inertia = math.inf
    for _ in range(max(1, int(max_iter))):
        lab, best = _assign_block(X, centers)
        inertia = float(np.sum(best, dtype=np.float64))
        counts = np.bincount(lab, minlength=k).astype(np.int64)
        sums = np.zeros((k, d), dtype=np.float64)
        for j in np.flatnonzero(counts > 0):
            sums[j] = np.sum(X[lab == j], axis=0, dtype=np.float64)
        nz = counts > 0
        centers[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
        cn = np.linalg.norm(centers, axis=1, keepdims=True)
        centers /= np.maximum(cn, np.float32(1e-12))
        rel = abs(prev - inertia) / max(abs(prev), 1.0) if np.isfinite(prev) else math.inf
        if rel <= tol:
            break
        prev = inertia
    return centers, float(inertia)


def _robust_sample_init(
    X: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    restarts: int = 5,
    local_iter: int = 8,
) -> np.ndarray:
    """Choose the best of several k-means++ starts after sample-local Lloyd."""
    best_c = None
    best_inertia = math.inf
    for _ in range(max(1, int(restarts))):
        C0 = _kmeanspp(X, k, rng)
        C, inertia = _refine_sample_centers(X, C0, max_iter=local_iter)
        if inertia < best_inertia:
            best_inertia = inertia
            best_c = C.copy()
    assert best_c is not None
    return best_c


def batched_lloyd(
    H: np.ndarray,
    n_clusters: int,
    *,
    block_rows: int = 262_144,
    max_iter: int = 20,
    tol: float = 1e-4,
    init_sample: int = 200_000,
    init_restarts: int = 5,
    init_local_iter: int = 8,
    random_state: int = 0,
    labels_out: np.ndarray | None = None,
    progress: ProgressSink | None = None,
) -> DiscretizationResult:
    """Full-data Lloyd iterations using bounded-memory blocks.

    This is the default large-scale discretizer. It uses k-means++ only on a
    bounded initialization sample, then every Lloyd iteration scans *all* n
    samples. Hence the iterative assignment/update itself is not mini-batch
    k-means and introduces no sklearn dependency.
    """
    progress = progress or ProgressSink()
    n, d = map(int, H.shape)
    if not (1 <= n_clusters <= n):
        raise ValueError('invalid n_clusters')
    block_rows = max(1024, int(block_rows))
    rng = np.random.default_rng(random_state)
    init = _sample_normalized(H, init_sample, rng)
    centers = _robust_sample_init(init, n_clusters, rng, restarts=init_restarts, local_iter=init_local_iter)
    labels = labels_out if labels_out is not None else np.empty(n, dtype=np.int32)
    if labels.shape != (n,) or labels.dtype != np.int32:
        raise ValueError('labels_out must be int32 with shape (n,)')

    t0 = time.perf_counter()
    prev_inertia = math.inf
    final_inertia = math.inf
    used_iter = 0
    for it in range(1, max_iter + 1):
        sums = np.zeros((n_clusters, d), dtype=np.float64)
        counts = np.zeros(n_clusters, dtype=np.int64)
        inertia = 0.0
        n_blocks = (n + block_rows - 1) // block_rows
        for ib, s in enumerate(range(0, n, block_rows)):
            e = min(n, s + block_rows)
            X = _normalized_block(H, s, e)
            lab, best = _assign_block(X, centers)
            labels[s:e] = lab
            inertia += float(np.sum(best, dtype=np.float64))
            # c is small in clustering; looping over clusters avoids a large
            # one-hot allocation and is independent of n.
            for j in np.unique(lab):
                mask = lab == j
                counts[j] += int(np.sum(mask))
                sums[j] += np.sum(X[mask], axis=0, dtype=np.float64)
            progress.emit(
                'discretize:lloyd',
                ((it - 1) + e / n) / max_iter,
                f'Batched Lloyd iter {it}: block {ib+1}/{n_blocks}, rows {e:,}/{n:,}',
                block_kind='lloyd_assign', block_index=ib, blocks_total=n_blocks,
                rows_start=s, rows_end=e, rows_total=n, lloyd_iter=it,
            )
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            # Deterministic rescue: re-seed empty centers from the init sample.
            for off, j in enumerate(empty):
                centers[j] = init[(it * 104729 + off) % init.shape[0]]
        nonempty = counts > 0
        centers[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        cn = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(cn, np.float32(1e-12))
        used_iter = it
        final_inertia = inertia
        rel = abs(prev_inertia - inertia) / max(abs(prev_inertia), 1.0) if np.isfinite(prev_inertia) else math.inf
        progress.emit('discretize:lloyd_pass', it / max_iter, f'Batched Lloyd pass {it}: inertia={inertia:.8e}, rel={rel:.2e}', lloyd_iter=it, inertia=inertia, relative_change=rel)
        if rel <= tol:
            break
        prev_inertia = inertia

    if isinstance(labels, np.memmap):
        labels.flush()
    return DiscretizationResult(labels, centers, float(final_inertia), used_iter, 'batched_lloyd', time.perf_counter() - t0)


def faiss_sampled_kmeans(
    H: np.ndarray,
    n_clusters: int,
    *,
    sample_size: int = 1_000_000,
    niter: int = 30,
    nredo: int = 1,
    block_rows: int = 262_144,
    random_state: int = 0,
    labels_out: np.ndarray | None = None,
    gpu: bool = False,
    progress: ProgressSink | None = None,
) -> DiscretizationResult:
    """Optional FAISS discretizer: train centers on a bounded sample, assign all rows.

    This is an engineering alternative for very large n. The default paper path
    remains batched_lloyd because it performs full-data Lloyd updates.
    """
    import faiss  # type: ignore
    progress = progress or ProgressSink()
    n, d = map(int, H.shape)
    rng = np.random.default_rng(random_state)
    train = np.ascontiguousarray(_sample_normalized(H, sample_size, rng), dtype=np.float32)
    t0 = time.perf_counter()
    progress.emit('discretize:faiss_train', 0.0, f'FAISS k-means training on {train.shape[0]:,} samples')
    km = faiss.Kmeans(d, n_clusters, niter=niter, nredo=nredo, verbose=False, seed=int(random_state), gpu=bool(gpu))
    km.train(train)
    centers = np.asarray(km.centroids, dtype=np.float32).reshape(n_clusters, d)
    cn = np.linalg.norm(centers, axis=1, keepdims=True)
    centers /= np.maximum(cn, np.float32(1e-12))
    progress.emit('discretize:faiss_train', 1.0, 'FAISS k-means centers ready')

    labels = labels_out if labels_out is not None else np.empty(n, dtype=np.int32)
    inertia = 0.0
    n_blocks = (n + block_rows - 1) // block_rows
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows)
        X = _normalized_block(H, s, e)
        lab, best = _assign_block(X, centers)
        labels[s:e] = lab
        inertia += float(np.sum(best, dtype=np.float64))
        progress.emit('discretize:faiss_assign', e / n, f'FAISS-center assignment block {ib+1}/{n_blocks}', block_kind='faiss_assign', block_index=ib, blocks_total=n_blocks, rows_start=s, rows_end=e, rows_total=n)
    if isinstance(labels, np.memmap):
        labels.flush()
    return DiscretizationResult(labels, centers, inertia, niter, 'faiss_sampled_kmeans_gpu' if gpu else 'faiss_sampled_kmeans_cpu', time.perf_counter() - t0)


def discretize_embedding(
    H: np.ndarray,
    n_clusters: int,
    *,
    backend: str = 'auto',
    block_rows: int = 262_144,
    max_iter: int = 20,
    tol: float = 1e-4,
    init_sample: int = 200_000,
    init_restarts: int = 5,
    init_local_iter: int = 8,
    random_state: int = 0,
    labels_out: np.ndarray | None = None,
    faiss_gpu: bool = False,
    progress: ProgressSink | None = None,
) -> DiscretizationResult:
    backend = backend.lower()
    if backend == 'auto':
        backend = 'batched_lloyd'
    if backend in {'batched_lloyd', 'lloyd'}:
        return batched_lloyd(H, n_clusters, block_rows=block_rows, max_iter=max_iter, tol=tol, init_sample=init_sample, init_restarts=init_restarts, init_local_iter=init_local_iter, random_state=random_state, labels_out=labels_out, progress=progress)
    if backend in {'standard', 'standard_replicate_lloyd', 'official'}:
        return standard_replicate_lloyd(H, n_clusters, replicates=max(1, int(init_restarts)), block_rows=block_rows, max_iter=max_iter, tol=tol, random_state=random_state, labels_out=labels_out, progress=progress)
    if backend in {'faiss', 'faiss_sampled'}:
        return faiss_sampled_kmeans(H, n_clusters, sample_size=max(init_sample, 200_000), niter=max_iter, block_rows=block_rows, random_state=random_state, labels_out=labels_out, gpu=faiss_gpu, progress=progress)
    raise ValueError("discretization backend must be 'auto', 'batched_lloyd', 'standard_replicate_lloyd', or 'faiss'")


def standard_replicate_lloyd(
    H: np.ndarray,
    n_clusters: int,
    *,
    replicates: int = 10,
    block_rows: int = 262_144,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 0,
    labels_out: np.ndarray | None = None,
    progress: ProgressSink | None = None,
) -> DiscretizationResult:
    """Reference discretization: row-normalize H and keep best of full Lloyd replicates.

    The released SimpleMKKM evaluation calls ``litekmeans`` on the row-normalized
    spectral embedding with ``MaxIter=100`` and ``Replicates=10``.  This routine
    implements the same clustering *protocol* without a MATLAB/sklearn runtime:
    each replicate performs a full-data Lloyd run with k-means++ initialization;
    the replicate with the smallest embedding SSE is retained.

    The outer 20 repetitions used by ``myNMIACCV2`` are an *evaluation* protocol,
    not a different clustering model.  They are handled separately when requested.
    """
    progress = progress or ProgressSink()
    n = int(H.shape[0])
    reps = max(1, int(replicates))
    best: DiscretizationResult | None = None
    best_labels: np.ndarray | None = None
    t0 = time.perf_counter()
    for r in range(reps):
        progress.emit('discretize:standard_replicate', r / reps, f'Standard Lloyd replicate {r+1}/{reps}', replicate=r+1, replicates=reps)
        tmp = np.empty(n, dtype=np.int32)
        res = batched_lloyd(
            H, n_clusters,
            block_rows=block_rows,
            max_iter=max_iter,
            tol=tol,
            init_sample=n,            # reference path: initialize from all rows
            init_restarts=1,
            init_local_iter=min(8, max_iter),
            random_state=int(random_state) + 104729 * r,
            labels_out=tmp,
            progress=progress,
        )
        if best is None or res.inertia < best.inertia:
            best = res
            best_labels = np.asarray(tmp, dtype=np.int32).copy()
    assert best is not None and best_labels is not None
    labels = labels_out if labels_out is not None else np.empty(n, dtype=np.int32)
    if labels.shape != (n,) or labels.dtype != np.int32:
        raise ValueError('labels_out must be int32 with shape (n,)')
    labels[:] = best_labels
    if isinstance(labels, np.memmap):
        labels.flush()
    progress.emit('discretize:standard_replicate', 1.0, f'Standard replicated Lloyd complete; best inertia={best.inertia:.8e}', replicates=reps, inertia=best.inertia)
    return DiscretizationResult(
        labels=labels,
        centers=np.asarray(best.centers, dtype=np.float32),
        inertia=float(best.inertia),
        n_iter=int(best.n_iter),
        backend=f'standard_replicate_lloyd_r{reps}',
        seconds=float(time.perf_counter() - t0),
    )
