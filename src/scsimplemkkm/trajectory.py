from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .discretize import standard_replicate_lloyd, discretize_embedding
from .evaluation import external_clustering_metrics
from .runtime import ProgressSink


@dataclass
class TrajectoryConfig:
    enabled: bool = True
    full_threshold: int = 100_000
    probe_size: int = 100_000
    replicates: int = 3
    max_iter: int = 30
    tol: float = 1e-4
    block_rows: int = 262_144
    seed: int = 0
    backend: str = 'standard_replicate_lloyd'
    init_sample: int = 200_000
    init_restarts: int = 5
    init_local_iter: int = 8


class IterationTrajectoryObserver:
    """Low-overhead diagnostic clustering on each *accepted* outer iterate.

    For n <= ``full_threshold`` metrics are computed on all samples.  For larger
    runs a single frozen probe set is selected once and reused for every outer
    iterate.  This makes ACC/NMI/ARI trajectories comparable across iterations
    without turning monitoring into the dominant cost at 1M--10M scale.

    The discretization seed is intentionally *fixed across accepted iterates*.
    Otherwise changes in ACC/NMI/ARI could be caused by a different k-means
    initialization rather than by a change in the spectral embedding itself.
    """

    def __init__(self, n_samples: int, n_clusters: int, *, y_true: np.ndarray | None, config: TrajectoryConfig):
        self.n = int(n_samples)
        self.c = int(n_clusters)
        self.y_true = None if y_true is None else np.asarray(y_true).reshape(-1)
        if self.y_true is not None and self.y_true.shape[0] != self.n:
            raise ValueError('trajectory ground truth length mismatch')
        self.cfg = config
        if self.n <= int(config.full_threshold):
            self.indices = None
            self.scope = 'FULL'
            self.metric_n = self.n
        else:
            m = min(self.n, max(self.c * 20, int(config.probe_size)))
            rng = np.random.default_rng(int(config.seed) + 99173)
            self.indices = np.sort(rng.choice(self.n, size=m, replace=False)).astype(np.int64)
            self.scope = 'PROBE'
            self.metric_n = int(m)

    def __call__(self, accepted_iter: int, H: np.ndarray, gamma: np.ndarray, objective: float, sinfo) -> dict:
        if not self.cfg.enabled:
            return {'metric_scope': 'OFF', 'metric_n': 0}
        t0 = time.perf_counter()
        Hs = np.asarray(H if self.indices is None else H[self.indices], dtype=np.float32)
        labels = np.empty(Hs.shape[0], dtype=np.int32)
        backend = str(self.cfg.backend or 'standard_replicate_lloyd').lower()
        if backend in {'standard', 'standard_replicate_lloyd', 'official'}:
            res = standard_replicate_lloyd(
                Hs,
                self.c,
                replicates=max(1, int(self.cfg.replicates)),
                block_rows=min(max(1024, int(self.cfg.block_rows)), max(1024, Hs.shape[0])),
                max_iter=max(1, int(self.cfg.max_iter)),
                tol=float(self.cfg.tol),
                random_state=int(self.cfg.seed),
                labels_out=labels,
                progress=ProgressSink(),  # do not pollute the main execution timeline
            )
        else:
            res = discretize_embedding(
                Hs,
                self.c,
                backend=backend,
                block_rows=min(max(1024, int(self.cfg.block_rows)), max(1024, Hs.shape[0])),
                max_iter=max(1, int(self.cfg.max_iter)),
                tol=float(self.cfg.tol),
                init_sample=min(int(self.cfg.init_sample), Hs.shape[0]),
                init_restarts=max(1, int(self.cfg.init_restarts)),
                init_local_iter=max(1, int(self.cfg.init_local_iter)),
                random_state=int(self.cfg.seed),
                labels_out=labels,
                progress=ProgressSink(),
            )
        row = {
            'metric_scope': self.scope,
            'metric_n': int(Hs.shape[0]),
            'diagnostic_inertia': float(res.inertia),
            'diagnostic_seconds': float(time.perf_counter() - t0),
            'diagnostic_backend': res.backend,
        }
        if self.y_true is not None:
            yt = self.y_true if self.indices is None else self.y_true[self.indices]
            ext = external_clustering_metrics(yt, labels)
            for k in ('ACC', 'NMI', 'ARI', 'Purity', 'MacroF1_mapped', 'pred_clusters', 'true_clusters'):
                if k in ext:
                    row[k] = ext[k]
        return row
