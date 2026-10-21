from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy import sparse

from .runtime import ProgressSink, ResourcePolicy


@dataclass
class NeighborGraph:
    """CSR sparse neighbor support with exact retained-edge similarities.

    ANN chooses candidates only. Similarities are recomputed from the original
    normalized float32 data on selected edges. dist2 is derived exactly as
    2-2*s for normalized data, so it is not stored separately.
    """

    n_samples: int
    k: int
    indptr: np.ndarray
    indices: np.ndarray
    similarity: np.ndarray
    backend: str
    support_mode: str
    score_mode: str
    normalized_input: bool = True

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    @property
    def cols(self) -> np.ndarray:
        return self.indices

    @property
    def rows(self) -> np.ndarray:
        # Compatibility helper; do not call at 10M scale unless necessary.
        return np.repeat(np.arange(self.n_samples, dtype=np.int32), np.diff(self.indptr))

    @property
    def dist2(self) -> np.ndarray:
        if not self.normalized_input:
            raise AttributeError('dist2 derivation requires normalized input')
        return np.maximum(np.float32(0.0), np.float32(2.0) - np.float32(2.0) * self.similarity)

    def adjacency(self) -> sparse.csr_matrix:
        data = np.ones(self.nnz, dtype=np.float32)
        return sparse.csr_matrix((data, self.indices, self.indptr), shape=(self.n_samples, self.n_samples))

    def bytes(self) -> int:
        return int(self.indptr.nbytes + self.indices.nbytes + self.similarity.nbytes)


def row_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order='C')
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, np.float32(eps))


def _topk_from_scores(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    val = np.take_along_axis(score, idx, axis=1)
    order = np.argsort(-val, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    val = np.take_along_axis(val, order, axis=1)
    return idx.astype(np.int32, copy=False), val.astype(np.float32, copy=False)


def exact_knn(
    Xn: np.ndarray,
    k: int,
    block_size: int = 2048,
    score_mode: str = 'cosine',
    progress: ProgressSink | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k support without n x n storage; still O(n^2 d) arithmetic."""
    n = int(Xn.shape[0])
    if not (1 <= k < n):
        raise ValueError(f'k must satisfy 1 <= k < n, got k={k}, n={n}')
    if score_mode not in {'cosine', 'abs_cosine'}:
        raise ValueError("score_mode must be 'cosine' or 'abs_cosine'")
    # An exact scan may be prohibitively slow, but must never allocate an
    # unbounded query-by-database block on a low-memory workstation.
    per_row = max(n * 4 * (2 if score_mode == 'abs_cosine' else 1), 1)
    block_size = max(1, min(int(block_size), (128 * 1024**2) // per_row or 1))
    out = np.empty((n, k), dtype=np.int32)
    sim_out = np.empty((n, k), dtype=np.float32)
    progress = progress or ProgressSink()
    XT = np.asarray(Xn, dtype=np.float32).T
    for s in range(0, n, block_size):
        e = min(n, s + block_size)
        q = np.asarray(Xn[s:e], dtype=np.float32)
        sims = q @ XT
        rr = np.arange(e - s)
        cc = np.arange(s, e)
        rank_score = sims if score_mode == 'cosine' else np.abs(sims)
        rank_score[rr, cc] = -np.inf
        idx, _ = _topk_from_scores(rank_score, k)
        true_sim = np.take_along_axis(sims, idx, axis=1)
        out[s:e] = idx
        sim_out[s:e] = true_sim.astype(np.float32)
        progress.emit('support:exact', e / n, f'Exact support {e:,}/{n:,}', rows_done=e, rows_total=n, rows_start=s, rows_end=e, block_kind='exact_support')
    return out, sim_out


def _rerank_batch(
    Xn: np.ndarray,
    rows: np.ndarray,
    candidates: np.ndarray,
    k: int,
    score_mode: str,
    max_temp_bytes: int = 32 * 1024 * 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact re-score ANN candidates in vectorized bounded-memory chunks.

    The ANN backend only proposes candidate ids. This routine recomputes exact
    retained-edge similarities from the normalized float32 input. It avoids a
    Python loop over query rows, which is essential at million-scale n.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cand0 = np.asarray(candidates, dtype=np.int64)
    if cand0.ndim != 2 or cand0.shape[0] != rows.size:
        raise ValueError('candidates must have shape (len(rows), breadth)')
    b, breadth = cand0.shape
    if breadth < k:
        raise RuntimeError(f'ANN returned breadth={breadth}; need k={k}')
    d = int(Xn.shape[1])
    out_i = np.empty((b, k), dtype=np.int32)
    out_s = np.empty((b, k), dtype=np.float32)

    # Gathered candidate tensor dominates temporary memory: chunk*breadth*d*4.
    # Include a conservative factor for ids/scores/order temporaries.
    per_row = max(1, breadth * d * 4 * 2 + breadth * 32)
    chunk_rows = max(1, min(b, int(max_temp_bytes // per_row)))

    for s in range(0, b, chunk_rows):
        e = min(b, s + chunk_rows)
        rr = rows[s:e]
        C = np.array(cand0[s:e], dtype=np.int64, copy=True)

        # Sort ids row-wise solely to mark duplicates cheaply in C/NumPy.
        order = np.argsort(C, axis=1, kind='stable')
        C = np.take_along_axis(C, order, axis=1)
        valid = (C >= 0) & (C != rr[:, None])
        if C.shape[1] > 1:
            valid[:, 1:] &= C[:, 1:] != C[:, :-1]
        safe = np.where(valid, C, 0)

        q = np.asarray(Xn[rr], dtype=np.float32)
        gathered = np.asarray(Xn[safe], dtype=np.float32)
        sims = np.einsum('bld,bd->bl', gathered, q, optimize=True).astype(np.float32, copy=False)
        score = np.abs(sims) if score_mode == 'abs_cosine' else sims
        score = np.array(score, copy=True)
        score[~valid] = -np.inf
        good = np.sum(np.isfinite(score), axis=1)
        if np.any(good < k):
            bad = int(np.flatnonzero(good < k)[0])
            raise RuntimeError(f'Only {int(good[bad])} unique ANN candidates for row {int(rr[bad])}; need {k}.')
        take = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        take_score = np.take_along_axis(score, take, axis=1)
        sort_top = np.argsort(-take_score, axis=1)
        take = np.take_along_axis(take, sort_top, axis=1)
        out_i[s:e] = np.take_along_axis(C, take, axis=1).astype(np.int32, copy=False)
        out_s[s:e] = np.take_along_axis(sims, take, axis=1).astype(np.float32, copy=False)
    return out_i, out_s


def _faiss_index(Xn: np.ndarray, ef_search: int, M: int, threads: int):
    import faiss  # type: ignore
    faiss.omp_set_num_threads(max(1, int(threads)))
    X32 = np.asarray(Xn, dtype=np.float32, order='C')
    d = X32.shape[1]
    try:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    except TypeError:
        index = faiss.IndexHNSWFlat(d, M)
        if hasattr(index, 'metric_type'):
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = max(ef_search, 2 * M)
    index.hnsw.efSearch = ef_search
    index.add(X32)
    return index, X32


def _ann_faiss(
    Xn: np.ndarray,
    k: int,
    ef_search: int,
    M: int,
    score_mode: str,
    candidate_factor: int,
    query_batch_size: int,
    threads: int,
    progress: ProgressSink,
) -> tuple[np.ndarray, np.ndarray]:
    progress.emit('ann:index', 0.0, 'Building FAISS HNSW index')
    index, X32 = _faiss_index(Xn, ef_search, M, threads)
    progress.emit('ann:index', 1.0, 'FAISS HNSW index built')
    n = X32.shape[0]
    breadth = min(n, max(k + 8, candidate_factor * k + 1))
    I = np.empty((n, k), dtype=np.int32)
    S = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, query_batch_size):
        e = min(n, s + query_batch_size)
        q = X32[s:e]
        _, ipos = index.search(q, breadth)
        if score_mode == 'abs_cosine':
            _, ineg = index.search(-q, breadth)
            cand = np.concatenate([ipos, ineg], axis=1)
        else:
            cand = ipos
        ii, ss = _rerank_batch(Xn, np.arange(s, e), cand, k, score_mode)
        I[s:e], S[s:e] = ii, ss
        progress.emit('ann:query', e / n, f'ANN query/rerank {e:,}/{n:,}', rows_done=e, rows_total=n, rows_start=s, rows_end=e, block_kind='ann_query_rerank')
    return I, S





def _low_memory_training_sample(X32: np.ndarray, size: int, random_state: int) -> np.ndarray:
    """Deterministic near-uniform training sample without O(n) choice workspace.

    ``Generator.choice(..., replace=False)`` may allocate state proportional to
    the full population on some NumPy versions.  At 10M scale the training set
    itself is modest, so select approximately uniform locations with a random
    phase and tiny bounded jitter.
    """
    n = int(X32.shape[0])
    size = int(min(max(1, size), n))
    if size >= n:
        return np.asarray(X32, dtype=np.float32, order='C')
    rng = np.random.default_rng(random_state)
    step = n / float(size)
    phase = float(rng.random()) * step
    idx = np.floor(phase + step * np.arange(size, dtype=np.float64)).astype(np.int64)
    np.minimum(idx, n - 1, out=idx)
    return np.asarray(X32[idx], dtype=np.float32, order='C')

def _ann_faiss_ivf(
    Xn: np.ndarray,
    k: int,
    score_mode: str,
    candidate_factor: int,
    query_batch_size: int,
    threads: int,
    random_state: int,
    nlist: int,
    nprobe: int,
    train_size: int,
    progress: ProgressSink,
) -> tuple[np.ndarray, np.ndarray]:
    import faiss  # type: ignore
    faiss.omp_set_num_threads(max(1, int(threads)))
    X32 = np.asarray(Xn, dtype=np.float32, order='C')
    n, d = X32.shape
    if nlist <= 0:
        nlist = int(min(65536, max(256, round(4.0 * np.sqrt(n)))))
    nlist = min(nlist, max(1, n // 20))
    nprobe = int(min(max(1, nprobe), nlist))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ts = int(min(n, max(train_size, 40 * nlist)))
    train = _low_memory_training_sample(X32, ts, random_state)
    progress.emit('ann:index', 0.05, f'Training FAISS IVF-Flat (nlist={nlist}, train={ts:,})', nlist=nlist, nprobe=nprobe)
    index.train(train)
    add_batch = max(query_batch_size, min(65536, max(1024, (32 * 1024**2) // max(d * 4, 1))))
    for s in range(0, n, add_batch):
        e = min(n, s + add_batch)
        index.add(np.asarray(X32[s:e], dtype=np.float32, order='C'))
        progress.emit('ann:index', 0.05 + 0.95 * e / n, f'Adding to IVF index {e:,}/{n:,}', nlist=nlist, nprobe=nprobe, rows_start=s, rows_end=e, rows_total=n, block_kind='ivf_add')
    index.nprobe = nprobe
    breadth = min(n, max(k + 8, candidate_factor * k + 1))
    I = np.empty((n, k), dtype=np.int32)
    S = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, query_batch_size):
        e = min(n, s + query_batch_size)
        q = X32[s:e]
        _, ipos = index.search(q, breadth)
        if score_mode == 'abs_cosine':
            _, ineg = index.search(-q, breadth)
            cand = np.concatenate([ipos, ineg], axis=1)
        else:
            cand = ipos
        ii, ss = _rerank_batch(Xn, np.arange(s, e), cand, k, score_mode)
        I[s:e], S[s:e] = ii, ss
        progress.emit('ann:query', e / n, f'IVF query/rerank {e:,}/{n:,}', rows_done=e, rows_total=n, rows_start=s, rows_end=e, block_kind='ivf_query_rerank', nlist=nlist, nprobe=nprobe)
    return I, S

def _ann_faiss_ivfsq(
    Xn: np.ndarray,
    k: int,
    score_mode: str,
    candidate_factor: int,
    query_batch_size: int,
    threads: int,
    random_state: int,
    nlist: int,
    nprobe: int,
    train_size: int,
    progress: ProgressSink,
) -> tuple[np.ndarray, np.ndarray]:
    """Memory-lean FAISS IVF scalar-quantized candidate discovery.

    The ANN index stores approximately one byte per feature dimension rather
    than a duplicate float32 vector bank.  Candidate edges are still rescored
    exactly from Xn, so quantization affects candidate discovery only.
    """
    import faiss  # type: ignore
    faiss.omp_set_num_threads(max(1, int(threads)))
    X32 = np.asarray(Xn, dtype=np.float32, order='C')
    n, d = X32.shape
    if nlist <= 0:
        nlist = int(min(65536, max(256, round(4.0 * np.sqrt(n)))))
    nlist = min(nlist, max(1, n // 20))
    nprobe = int(min(max(1, nprobe), nlist))
    quantizer = faiss.IndexFlatIP(d)
    try:
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
    except TypeError:  # older FAISS bindings expose by_residual explicitly
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT, True,
        )
    ts = int(min(n, max(train_size, 40 * nlist)))
    train = _low_memory_training_sample(X32, ts, random_state)
    progress.emit('ann:index', 0.05, f'Training FAISS IVF-SQ8 (nlist={nlist}, train={ts:,})', nlist=nlist, nprobe=nprobe)
    index.train(train)
    add_batch = max(query_batch_size, min(65536, max(1024, (32 * 1024**2) // max(d * 4, 1))))
    for s in range(0, n, add_batch):
        e = min(n, s + add_batch)
        index.add(np.asarray(X32[s:e], dtype=np.float32, order='C'))
        progress.emit('ann:index', 0.05 + 0.95 * e / n, f'Adding to IVF-SQ8 index {e:,}/{n:,}', nlist=nlist, nprobe=nprobe, rows_start=s, rows_end=e, rows_total=n, block_kind='ivfsq_add')
    index.nprobe = nprobe
    breadth = min(n, max(k + 8, candidate_factor * k + 1))
    I = np.empty((n, k), dtype=np.int32)
    S = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, query_batch_size):
        e = min(n, s + query_batch_size)
        q = X32[s:e]
        _, ipos = index.search(q, breadth)
        if score_mode == 'abs_cosine':
            _, ineg = index.search(-q, breadth)
            cand = np.concatenate([ipos, ineg], axis=1)
        else:
            cand = ipos
        ii, ss = _rerank_batch(Xn, np.arange(s, e), cand, k, score_mode)
        I[s:e], S[s:e] = ii, ss
        progress.emit('ann:query', e / n, f'IVF-SQ8 query/rerank {e:,}/{n:,}', rows_done=e, rows_total=n, rows_start=s, rows_end=e, block_kind='ivfsq_query_rerank', nlist=nlist, nprobe=nprobe)
    return I, S


def _ann_hnswlib(
    Xn: np.ndarray,
    k: int,
    ef_search: int,
    M: int,
    random_state: int,
    score_mode: str,
    candidate_factor: int,
    query_batch_size: int,
    threads: int,
    progress: ProgressSink,
) -> tuple[np.ndarray, np.ndarray]:
    import hnswlib  # type: ignore
    n, d = Xn.shape
    X32 = np.asarray(Xn, dtype=np.float32, order='C')
    progress.emit('ann:index', 0.0, 'Building hnswlib index')
    index = hnswlib.Index(space='cosine', dim=d)
    index.init_index(max_elements=n, ef_construction=max(ef_search, 2 * M), M=M, random_seed=random_state)
    index.add_items(X32, np.arange(n), num_threads=max(1, threads))
    index.set_ef(max(ef_search, 64))
    progress.emit('ann:index', 1.0, 'hnswlib index built')
    breadth = min(n, max(k + 8, candidate_factor * k + 1))
    I = np.empty((n, k), dtype=np.int32)
    S = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, query_batch_size):
        e = min(n, s + query_batch_size)
        q = X32[s:e]
        ipos, _ = index.knn_query(q, k=breadth, num_threads=max(1, threads))
        if score_mode == 'abs_cosine':
            ineg, _ = index.knn_query(-q, k=breadth, num_threads=max(1, threads))
            cand = np.concatenate([ipos, ineg], axis=1)
        else:
            cand = ipos
        ii, ss = _rerank_batch(Xn, np.arange(s, e), cand, k, score_mode)
        I[s:e], S[s:e] = ii, ss
        progress.emit('ann:query', e / n, f'ANN query/rerank {e:,}/{n:,}', rows_done=e, rows_total=n, rows_start=s, rows_end=e, block_kind='ann_query_rerank')
    return I, S


def _ann_pynndescent(Xn: np.ndarray, k: int, random_state: int, progress: ProgressSink) -> tuple[np.ndarray, np.ndarray]:
    from pynndescent import NNDescent  # type: ignore
    n = Xn.shape[0]
    progress.emit('ann:index', 0.0, 'Building PyNNDescent graph')
    index = NNDescent(np.asarray(Xn, dtype=np.float32), n_neighbors=min(n - 1, max(k + 8, 2 * k)), metric='cosine', random_state=random_state, low_memory=True)
    cand, _ = index.neighbor_graph
    progress.emit('ann:index', 1.0, 'PyNNDescent graph built')
    return _rerank_batch(Xn, np.arange(n), cand, k, 'cosine')


def _sklearn_knn(Xn: np.ndarray, k: int, n_jobs: int, progress: ProgressSink) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors
    n = Xn.shape[0]
    progress.emit('ann:index', 0.0, 'Building sklearn brute-force neighbor search')
    nn = NearestNeighbors(n_neighbors=min(n, k + 1), metric='cosine', algorithm='brute', n_jobs=n_jobs)
    nn.fit(Xn)
    cand = nn.kneighbors(Xn, return_distance=False)
    progress.emit('ann:index', 1.0, 'sklearn brute-force search complete')
    return _rerank_batch(Xn, np.arange(n), cand, k, 'cosine')


def ann_knn(
    Xn: np.ndarray,
    k: int,
    backend: str = 'auto',
    ef_search: int = 128,
    hnsw_m: int = 32,
    random_state: int = 0,
    n_jobs: int = -1,
    score_mode: str = 'cosine',
    candidate_factor: int = 4,
    exact_block_size: int = 2048,
    query_batch_size: int = 4096,
    threads: int | None = None,
    progress: ProgressSink | None = None,
    ivf_nlist: int = 0,
    ivf_nprobe: int = 32,
    ivf_train_size: int = 200000,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Approximate support discovery followed by exact retained-edge reranking."""
    backend = backend.lower()
    progress = progress or ProgressSink()
    threads = max(1, int(threads or (n_jobs if n_jobs > 0 else 1)))
    if backend != 'auto':
        candidates = [backend]
    else:
        feature_bytes = int(Xn.shape[0]) * int(Xn.shape[1]) * 4
        if Xn.shape[0] >= 1_000_000 or feature_bytes >= 2 * 1024**3:
            # Preflight for auto at this scale budgets the SQ8 index.
            # Never silently switch to a much larger full-vector index.
            candidates = ['faiss_ivfsq']
        elif Xn.shape[0] >= 500000:
            candidates = ['faiss_ivfsq', 'faiss_ivf']
        else:
            candidates = ['faiss', 'hnswlib', 'faiss_ivf', 'faiss_ivfsq', 'pynndescent', 'exact']
    last_error = None
    for name in candidates:
        try:
            if name in {'faiss_ivfsq', 'faiss_ivf_sq', 'faiss_ivf_sq8'}:
                I, S = _ann_faiss_ivfsq(Xn, k, score_mode, candidate_factor, query_batch_size, threads, random_state, ivf_nlist, ivf_nprobe, ivf_train_size, progress)
                return I, S, f'faiss-ivf-sq8:{score_mode}'
            if name == 'faiss_ivf':
                I, S = _ann_faiss_ivf(Xn, k, score_mode, candidate_factor, query_batch_size, threads, random_state, ivf_nlist, ivf_nprobe, ivf_train_size, progress)
                return I, S, f'faiss-ivf-flat:{score_mode}'
            if name == 'faiss':
                I, S = _ann_faiss(Xn, k, ef_search, hnsw_m, score_mode, candidate_factor, query_batch_size, threads, progress)
                return I, S, f'faiss-hnsw:{score_mode}'
            if name == 'hnswlib':
                I, S = _ann_hnswlib(Xn, k, ef_search, hnsw_m, random_state, score_mode, candidate_factor, query_batch_size, threads, progress)
                return I, S, f'hnswlib:{score_mode}'
            if name == 'pynndescent':
                if score_mode != 'cosine':
                    raise ImportError('PyNNDescent path is enabled only for cosine support')
                I, S = _ann_pynndescent(Xn, k, random_state, progress)
                return I, S, 'pynndescent:cosine'
            if name == 'sklearn':
                I, S = _sklearn_knn(Xn, k, n_jobs=threads, progress=progress)
                return I, S, 'sklearn-brute:cosine'
            if name == 'exact':
                warnings.warn('No ANN package is available; falling back to exact blockwise support search. Do not report this as ANN scaling.', RuntimeWarning)
                I, S = exact_knn(Xn, k, block_size=exact_block_size, score_mode=score_mode, progress=progress)
                return I, S, f'exact-blockwise:{score_mode}'
            raise ValueError(f'Unknown ANN backend: {name}')
        except (ImportError, ModuleNotFoundError) as e:
            last_error = e
            continue
    if backend == 'auto' and Xn.shape[0] >= 500_000:
        raise RuntimeError(
            'Large-scale ANN requires FAISS IVF-SQ8/IVF-Flat; automatic all-pairs/HNSW fallback '
            'is disabled to protect workstation RAM. Install faiss-cpu or select a backend explicitly. '
            f'Last backend error: {last_error}'
        )
    raise RuntimeError(f'No usable ANN backend found. Last error: {last_error}')


def _csr_from_dense_knn(indices: np.ndarray, similarity: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = int(indices.shape[1])
    indptr = np.arange(0, n * k + 1, k, dtype=np.int64)
    return indptr, indices.reshape(-1).astype(np.int32, copy=False), similarity.reshape(-1).astype(np.float32, copy=False)


def _mutual_filter(indices: np.ndarray, similarity: np.ndarray, n: int, rescue_min_degree: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = indices.shape[1]
    indptr0 = np.arange(0, n * k + 1, k, dtype=np.int64)
    B = sparse.csr_matrix((np.ones(n * k, dtype=np.uint8), indices.reshape(-1), indptr0), shape=(n, n))
    M = B.multiply(B.T).tocsr()
    if rescue_min_degree > 0:
        deg = np.diff(M.indptr)
        # Build rescue directed edges only for deficient rows, then union.
        rr: list[int] = []
        cc: list[int] = []
        for i in np.flatnonzero(deg < rescue_min_degree):
            existing = set(M.indices[M.indptr[i]:M.indptr[i+1]].tolist())
            need = rescue_min_degree - len(existing)
            for j in indices[i]:
                jj = int(j)
                if jj not in existing:
                    rr.append(int(i)); cc.append(jj); existing.add(jj); need -= 1
                    if need <= 0:
                        break
        if rr:
            R = sparse.csr_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)), shape=(n, n))
            M = (M + R).astype(bool).astype(np.uint8).tocsr()
    # Recompute exact similarity for retained edges from the original kNN table via a map.
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(M.indptr))
    cols = M.indices.astype(np.int32, copy=False)
    lookup = sparse.csr_matrix((similarity.reshape(-1), indices.reshape(-1), indptr0), shape=(n, n))
    vals = np.asarray(lookup[rows, cols]).ravel().astype(np.float32)
    return M.indptr.astype(np.int64, copy=False), cols, vals


def build_neighbor_graph(
    X: np.ndarray,
    k: int = 10,
    search: str = 'ann',
    ann_backend: str = 'auto',
    support_mode: str = 'directed',
    score_mode: str = 'cosine',
    block_size: int = 2048,
    ef_search: int = 128,
    hnsw_m: int = 32,
    candidate_factor: int = 4,
    rescue_min_degree: int = 0,
    normalize: bool = True,
    random_state: int = 0,
    n_jobs: int = -1,
    query_batch_size: int = 4096,
    resource_policy: ResourcePolicy | None = None,
    progress: ProgressSink | None = None,
    ivf_nlist: int = 0,
    ivf_nprobe: int = 32,
    ivf_train_size: int = 200000,
) -> tuple[NeighborGraph, np.ndarray]:
    """Build one sparse support family.

    At scale the caller should pass an already normalized float32 X and
    normalize=False, avoiding a hidden full-memory copy.
    """
    progress = progress or ProgressSink()
    X0 = np.asarray(X)
    Xn = row_l2_normalize(X0) if normalize else np.asarray(X0, dtype=np.float32, order='C')
    n = int(Xn.shape[0])
    policy = (resource_policy or ResourcePolicy()).normalized()
    threads = policy.recommended_threads()
    if search == 'exact':
        indices, sims = exact_knn(Xn, k, block_size=block_size, score_mode=score_mode, progress=progress)
        backend_used = f'exact-blockwise:{score_mode}'
    elif search == 'ann':
        indices, sims, backend_used = ann_knn(
            Xn, k, backend=ann_backend, ef_search=ef_search, hnsw_m=hnsw_m,
            random_state=random_state, n_jobs=threads, score_mode=score_mode,
            candidate_factor=candidate_factor, exact_block_size=block_size,
            query_batch_size=query_batch_size, threads=threads, progress=progress,
            ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe, ivf_train_size=ivf_train_size,
        )
    else:
        raise ValueError("search must be 'ann' or 'exact'")

    if support_mode == 'directed':
        indptr, cols, vals = _csr_from_dense_knn(indices, sims, n)
    elif support_mode == 'mutual':
        indptr, cols, vals = _mutual_filter(indices, sims, n, rescue_min_degree=0)
    elif support_mode == 'mutual_rescue':
        indptr, cols, vals = _mutual_filter(indices, sims, n, rescue_min_degree=max(1, rescue_min_degree or min(2, k)))
    else:
        raise ValueError("support_mode must be directed, mutual, or mutual_rescue")

    graph = NeighborGraph(
        n_samples=n, k=int(k), indptr=indptr, indices=cols, similarity=vals,
        backend=backend_used, support_mode=support_mode, score_mode=score_mode,
        normalized_input=True,
    )
    progress.emit(f'support:{score_mode}', 1.0, f'{score_mode} support ready', nnz=graph.nnz, bytes=graph.bytes(), backend=backend_used)
    return graph, Xn


def graph_diagnostics(graph: NeighborGraph, full_threshold: int = 1_000_000) -> dict:
    """Support diagnostics with an automatic scale-safe path.

    Full undirected connectivity diagnostics require constructing an additional
    symmetrized CSR graph.  That is useful on small/medium data but can create a
    multi-GB post-fit spike at 10M scale.  Above ``full_threshold`` we report
    exact directed-row statistics and explicitly mark connectivity as skipped.
    """
    n = int(graph.n_samples)
    if n > int(full_threshold):
        deg = np.diff(graph.indptr)
        return {
            'n_samples': n,
            'k': int(graph.k),
            'nnz_directed': int(graph.nnz),
            'support_bytes': int(graph.bytes()),
            'backend': graph.backend,
            'support_mode': graph.support_mode,
            'score_mode': graph.score_mode,
            'mean_degree': float(np.mean(deg)) if deg.size else 0.0,
            'min_degree': int(np.min(deg)) if deg.size else 0,
            'max_degree': int(np.max(deg)) if deg.size else 0,
            'isolated': int(np.sum(deg == 0)),
            'connected_components': -1,
            'diagnostic_mode': 'lightweight_directed',
            'connectivity_skipped': True,
        }
    A = graph.adjacency()
    U = (A + A.T).astype(bool).astype(np.uint8).tocsr()
    deg = np.diff(U.indptr)
    try:
        ncomp = int(sparse.csgraph.connected_components(U, directed=False, return_labels=False))
    except Exception:
        ncomp = -1
    return {
        'n_samples': n,
        'k': int(graph.k),
        'nnz_directed': int(graph.nnz),
        'support_bytes': int(graph.bytes()),
        'backend': graph.backend,
        'support_mode': graph.support_mode,
        'score_mode': graph.score_mode,
        'mean_degree': float(np.mean(deg)) if deg.size else 0.0,
        'min_degree': int(np.min(deg)) if deg.size else 0,
        'max_degree': int(np.max(deg)) if deg.size else 0,
        'isolated': int(np.sum(deg == 0)),
        'connected_components': ncomp,
        'diagnostic_mode': 'full_undirected',
        'connectivity_skipped': False,
    }


def neighbor_recall(approx: NeighborGraph, exact: NeighborGraph) -> float:
    if approx.n_samples != exact.n_samples:
        raise ValueError('graph sizes differ')
    n = approx.n_samples
    hit = 0
    denom = 0
    for i in range(n):
        a = set(map(int, approx.indices[approx.indptr[i]:approx.indptr[i+1]]))
        e = set(map(int, exact.indices[exact.indptr[i]:exact.indptr[i+1]]))
        hit += len(a & e)
        denom += len(e)
    return float(hit / max(denom, 1))
