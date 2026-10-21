from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np


PRESETS = {
    '1m': 1_000_000,
    '2m': 2_000_000,
    '5m': 5_000_000,
    '10m': 10_000_000,
}


def _parse_dims(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def generate_multiview_gaussian(
    out_dir: str | Path,
    n: int,
    *,
    dims: list[int] = [32, 24],
    n_clusters: int = 10,
    noise: float = 0.30,
    seed: int = 20261003,
    block_rows: int = 100_000,
) -> dict:
    """Generate ready-to-run float32 L2-normalized multi-view HPC benchmarks.

    Files are written directly as .npy memmaps; no full nxd temporary is held.
    Ground-truth int32 labels are included solely for external evaluation.
    """
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    centers = []
    for d in dims:
        C = rng.standard_normal((n_clusters, d), dtype=np.float32)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), np.float32(1e-12))
        centers.append(C)
    ys = np.lib.format.open_memmap(root / 'labels.npy', mode='w+', dtype=np.int32, shape=(n,))
    views = [np.lib.format.open_memmap(root / f'view{i+1}.npy', mode='w+', dtype=np.float32, shape=(n, d)) for i, d in enumerate(dims)]
    counts = np.zeros(n_clusters, dtype=np.int64)
    t0 = time.perf_counter()
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows); b = e - s
        # Near-balanced labels, shuffled per block, makes all large presets have
        # stable class counts without constructing a global permutation.
        lab = (np.arange(s, e, dtype=np.int64) % n_clusters).astype(np.int32)
        rng.shuffle(lab)
        ys[s:e] = lab
        counts += np.bincount(lab, minlength=n_clusters)
        for iv, (d, C, out) in enumerate(zip(dims, centers, views)):
            X = C[lab] + np.float32(noise * (1.0 + 0.12 * iv)) * rng.standard_normal((b, d), dtype=np.float32)
            # View-specific structured nuisance: a weak common direction whose
            # amplitude varies by sample; it preserves labels but prevents views
            # from being literal copies.
            nuisance = rng.standard_normal((1, d), dtype=np.float32)
            nuisance /= np.maximum(np.linalg.norm(nuisance), np.float32(1e-12))
            amp = rng.standard_normal((b, 1), dtype=np.float32) * np.float32(0.10 + 0.03 * iv)
            X += amp * nuisance
            X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.float32(1e-12))
            out[s:e] = X
        if (ib + 1) % 10 == 0 or e == n:
            print(f'generated {e:,}/{n:,} rows')
    ys.flush(); [v.flush() for v in views]
    manifest = {
        'name': f'sc_hpc_{n}', 'dataset': f'ScaleSynth-MVC-{n}', 'n_samples': int(n), 'n_clusters': int(n_clusters), 'classes': int(n_clusters),
        'views': [{'name': f'view{i+1}', 'path': str((root / f"view{i+1}.npy").resolve()), 'shape': [int(n), int(d)], 'dtype': 'float32', 'l2_normalized': True} for i, d in enumerate(dims)],
        'labels': str((root / 'labels.npy').resolve()), 'label_dtype': 'int32',
        'dims': list(map(int, dims)), 'view_dims': list(map(int, dims)), 'view_count': len(dims), 'noise': float(noise), 'seed': int(seed),
        'cluster_counts': counts.tolist(), 'generation_seconds': time.perf_counter() - t0,
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10, 'recommended_kernel_preset': 'rbf_self_tuned',
        'purpose': 'single-node HPC scaling; controlled multi-view ground truth; not a real-world scientific benchmark',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    config = {
        'clusters': int(n_clusters), 'k': 10,
        'views': [{'name': v['name'], 'path': v['path'], 'k': 10} for v in manifest['views']],
        'kernel_bank': {'mode': 'shared', 'preset': 'rbf_self_tuned', 'local_scale_k': 7},
        'ground_truth': manifest['labels'], 'search': 'ann', 'ann_backend': 'auto',
        'support_mode': 'directed', 'support_policy': 'ranking_grouped', 'diffusion_power': 1,
        'solver': {'inner_solver': 'auto', 'inner_tol': 1e-4, 'inner_max_iter': 200, 'outer_tol': 1e-5, 'outer_max_iter': 100},
        'discretization': {'backend': 'batched_lloyd', 'block_rows': 262144, 'max_iter': 20, 'tol': 1e-4, 'init_sample': 200000, 'init_restarts': 5, 'init_local_iter': 8},
        'evaluation': {'enabled': True, 'block_rows': 262144},
        'out': str((root / 'run' / 'result.npz').resolve()),
        'workspace': str((root / 'run' / 'workspace').resolve()),
        'run_dir': str((root / 'run').resolve()),
    }
    (root / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate processed SC-SimpleMKKM HPC benchmark data')
    ap.add_argument('--out-root', default='benchdata')
    ap.add_argument('--sizes', default='1m,2m,5m,10m', help='comma separated: 1m,2m,5m,10m or integer sample counts')
    ap.add_argument('--dims', default='32,24')
    ap.add_argument('--clusters', type=int, default=10)
    ap.add_argument('--noise', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--block-rows', type=int, default=100000)
    args = ap.parse_args()
    sizes=[]
    for tok in args.sizes.split(','):
        tok=tok.strip().lower(); sizes.append(PRESETS[tok] if tok in PRESETS else int(tok))
    dims=_parse_dims(args.dims)
    root=Path(args.out_root); root.mkdir(parents=True, exist_ok=True)
    suite=[]
    for n in sizes:
        out=root / (f'{n//1_000_000}m' if n % 1_000_000 == 0 else str(n))
        suite.append(generate_multiview_gaussian(out,n,dims=dims,n_clusters=args.clusters,noise=args.noise,seed=args.seed+n,block_rows=args.block_rows))
    (root/'suite_manifest.json').write_text(json.dumps(suite,indent=2,ensure_ascii=False),encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
