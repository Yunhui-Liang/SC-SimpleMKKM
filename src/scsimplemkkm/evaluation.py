from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .runtime import ProgressSink


def _comb2(x: np.ndarray | float) -> np.ndarray | float:
    return x * (x - 1.0) / 2.0


def contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray, block_rows: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape or yt.ndim != 1:
        raise ValueError('labels must be 1D arrays with identical shape')
    true_vals = np.unique(yt)
    pred_vals = np.unique(yp)
    # c is normally small, so mapping through sorted unique values is cheap.
    C = np.zeros((true_vals.size, pred_vals.size), dtype=np.int64)
    for s in range(0, yt.size, block_rows):
        e = min(yt.size, s + block_rows)
        ti = np.searchsorted(true_vals, yt[s:e])
        pi = np.searchsorted(pred_vals, yp[s:e])
        flat = ti.astype(np.int64) * pred_vals.size + pi.astype(np.int64)
        C += np.bincount(flat, minlength=true_vals.size * pred_vals.size).reshape(C.shape)
    return C, true_vals, pred_vals


def external_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    C, true_vals, pred_vals = contingency_matrix(y_true, y_pred)
    n = int(C.sum())
    if n <= 0:
        return {}
    a = C.sum(axis=1).astype(np.float64)
    b = C.sum(axis=0).astype(np.float64)
    # ACC through optimal one-to-one assignment. Rectangular contingency is OK.
    rr, cc = linear_sum_assignment(-C)
    acc = float(C[rr, cc].sum() / n)
    purity = float(np.max(C, axis=0).sum() / n)

    nz = C > 0
    ii, jj = np.nonzero(nz)
    vals = C[ii, jj].astype(np.float64)
    mi = float(np.sum((vals / n) * np.log((vals * n) / (a[ii] * b[jj]))))
    pa = a / n; pb = b / n
    ha = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    nmi = float(mi / max(math.sqrt(max(ha * hb, 0.0)), np.finfo(float).eps))

    sum_comb = float(np.sum(_comb2(C.astype(np.float64))))
    sum_a = float(np.sum(_comb2(a)))
    sum_b = float(np.sum(_comb2(b)))
    total_comb = float(_comb2(float(n)))
    expected = sum_a * sum_b / max(total_comb, np.finfo(float).eps)
    max_index = 0.5 * (sum_a + sum_b)
    ari = float((sum_comb - expected) / max(max_index - expected, np.finfo(float).eps))

    # Macro F1 after the same Hungarian label alignment.
    f1s = []
    for r, c in zip(rr, cc):
        tp = float(C[r, c]); fp = float(b[c] - tp); fn = float(a[r] - tp)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else 2 * tp / denom)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {
        'n': n, 'true_clusters': int(C.shape[0]), 'pred_clusters': int(C.shape[1]),
        'ACC': acc, 'NMI': nmi, 'ARI': ari, 'Purity': purity, 'MacroF1_mapped': macro_f1,
    }


def internal_embedding_metrics(
    H: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray | None = None,
    *,
    block_rows: int = 262_144,
    progress: ProgressSink | None = None,
) -> dict:
    """O(nc)-ish streaming internal metrics on the row-normalized embedding.

    Reports full-data SSE/inertia, Calinski-Harabasz, Davies-Bouldin and cluster
    size diagnostics. No O(n^2) silhouette computation is attempted at scale.
    """
    progress = progress or ProgressSink()
    y = np.asarray(labels, dtype=np.int32)
    n, d = map(int, H.shape)
    if y.shape != (n,):
        raise ValueError('labels shape mismatch')
    c = int(y.max()) + 1 if y.size else 0
    counts = np.zeros(c, dtype=np.int64)
    sums = np.zeros((c, d), dtype=np.float64)
    global_sum = np.zeros(d, dtype=np.float64)
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows)
        X = np.asarray(H[s:e], dtype=np.float32)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.float32(1e-12))
        lab = y[s:e]
        global_sum += np.sum(X, axis=0, dtype=np.float64)
        for j in np.unique(lab):
            m = lab == j
            counts[j] += int(np.sum(m)); sums[j] += np.sum(X[m], axis=0, dtype=np.float64)
        progress.emit('evaluate:internal_centers', e / n, f'Internal metrics centers block {ib+1}', block_kind='eval_centers', rows_start=s, rows_end=e, rows_total=n)
    ctr = np.zeros_like(sums)
    nonempty = counts > 0
    ctr[nonempty] = sums[nonempty] / counts[nonempty, None]
    if centers is not None and centers.shape == ctr.shape:
        ctr_assign = np.asarray(centers, dtype=np.float64)
    else:
        ctr_assign = ctr
    within_sq = 0.0
    within_dist_sum = np.zeros(c, dtype=np.float64)
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows)
        X = np.asarray(H[s:e], dtype=np.float32)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.float32(1e-12))
        lab = y[s:e]
        diff = X.astype(np.float64) - ctr_assign[lab]
        d2 = np.sum(diff * diff, axis=1)
        within_sq += float(np.sum(d2, dtype=np.float64))
        dist = np.sqrt(np.maximum(d2, 0.0))
        for j in np.unique(lab):
            within_dist_sum[j] += float(np.sum(dist[lab == j], dtype=np.float64))
        progress.emit('evaluate:internal_dispersion', e / n, f'Internal metrics dispersion block {ib+1}', block_kind='eval_dispersion', rows_start=s, rows_end=e, rows_total=n)
    g = global_sum / max(n, 1)
    between = float(np.sum(counts[:, None] * (ctr - g[None, :]) ** 2, dtype=np.float64))
    active = int(np.sum(nonempty))
    ch = float((between / max(active - 1, 1)) / max(within_sq / max(n - active, 1), np.finfo(float).eps))
    scat = np.zeros(c, dtype=np.float64)
    scat[nonempty] = within_dist_sum[nonempty] / counts[nonempty]
    active_idx = np.flatnonzero(nonempty)
    db_terms = []
    for i in active_idx:
        others = active_idx[active_idx != i]
        if others.size == 0:
            continue
        sep = np.linalg.norm(ctr[i] - ctr[others], axis=1)
        ratio = (scat[i] + scat[others]) / np.maximum(sep, np.finfo(float).eps)
        db_terms.append(float(np.max(ratio)))
    db = float(np.mean(db_terms)) if db_terms else 0.0
    p = counts[nonempty].astype(np.float64) / max(n, 1)
    ent = float(-np.sum(p * np.log(np.maximum(p, np.finfo(float).eps))) / max(math.log(max(active, 2)), np.finfo(float).eps))
    return {
        'n': n, 'clusters_active': active, 'inertia_sse': float(within_sq),
        'CalinskiHarabasz': ch, 'DaviesBouldin': db,
        'cluster_size_min': int(counts[nonempty].min()) if active else 0,
        'cluster_size_max': int(counts[nonempty].max()) if active else 0,
        'cluster_size_entropy_norm': ent,
    }


def evaluate_clustering(H: np.ndarray, labels: np.ndarray, *, centers: np.ndarray | None = None, y_true: np.ndarray | None = None, block_rows: int = 262_144, progress: ProgressSink | None = None) -> dict:
    out = {'internal': internal_embedding_metrics(H, labels, centers, block_rows=block_rows, progress=progress)}
    if y_true is not None:
        out['external'] = external_clustering_metrics(y_true, labels)
    return out


def official_simplemkkm_external_protocol(
    H: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    *,
    rounds: int = 20,
    replicates: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 0,
    block_rows: int = 262_144,
) -> dict:
    """Reproduce the released ``myNMIACCV2`` evaluation structure.

    Each round clusters the row-normalized embedding using a k-means call with
    ``Replicates=10`` semantics, then ACC/NMI/Purity/ARI are computed.  The
    released MATLAB code repeats this 20 times and reports mean/std.

    This is intentionally separate from the algorithm's canonical ``labels.npy``
    because the published evaluation protocol is stochastic and returns summary
    statistics rather than one authoritative partition.
    """
    from .discretize import standard_replicate_lloyd

    yt = np.asarray(y_true).reshape(-1)
    if yt.shape[0] != int(H.shape[0]):
        raise ValueError('ground truth length mismatch')
    rows = []
    for r in range(max(1, int(rounds))):
        labels = np.empty(yt.shape[0], dtype=np.int32)
        disc = standard_replicate_lloyd(
            H, int(n_clusters),
            replicates=max(1, int(replicates)),
            block_rows=block_rows,
            max_iter=max_iter,
            tol=tol,
            random_state=int(random_state) + 1_000_003 * r,
            labels_out=labels,
            progress=ProgressSink(),
        )
        ext = external_clustering_metrics(yt, labels)
        rows.append({
            'round': r + 1,
            'ACC': float(ext['ACC']),
            'NMI': float(ext['NMI']),
            'Purity': float(ext['Purity']),
            'ARI': float(ext['ARI']),
            'inertia': float(disc.inertia),
        })
    summary = {}
    for k in ('ACC', 'NMI', 'Purity', 'ARI'):
        a = np.asarray([x[k] for x in rows], dtype=np.float64)
        summary[k + '_mean'] = float(np.mean(a))
        summary[k + '_std'] = float(np.std(a, ddof=1 if a.size > 1 else 0))
    return {
        'protocol': 'released_myNMIACCV2_structure',
        'rounds': int(rounds),
        'replicates_per_round': int(replicates),
        'max_iter': int(max_iter),
        'summary': summary,
        'round_metrics': rows,
    }
