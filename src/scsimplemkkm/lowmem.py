"""Low-memory execution controls for consumer workstations.

Disk-backed CSR VALUES are an optional execution representation.  The entire
mathematical operator and float32 values remain unchanged.  File-backed memory
permits the OS to reclaim clean pages; it is not a guarantee that every workload
will fit or run quickly with insufficient physical RAM.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from .runtime import human_bytes, system_info


def resolve_memory_mode(mode: str, n: int) -> str:
    key = str(mode or 'auto').strip().lower()
    if key not in {'auto', 'standard', 'laptop32'}:
        raise ValueError("memory_mode must be auto, standard or laptop32")
    if key != 'auto':
        return key
    total = int(system_info().get('ram_total_bytes', 0))
    return 'laptop32' if total > 0 and total <= 40 * 1024**3 and int(n) >= 100_000 else 'standard'


def scratch_directory(workspace: str | Path | None) -> Path:
    import tempfile
    root = Path(workspace or tempfile.gettempdir()) / 'scsimplemkkm_cache' / ('scratch_' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=False)
    return root


def check_disk_capacity(root: Path, next_bytes: int, reserve_bytes: int = 2 * 1024**3) -> None:
    free = int(shutil.disk_usage(root).free)
    required = int(next_bytes) + int(reserve_bytes)
    if free < required:
        raise OSError(f'Low-memory mode requires NVMe workspace: only {human_bytes(free)} free on {root}, '
                      f'need at least {human_bytes(required)} for this stage. Free disk space or move workspace.')


def disk_back_csr_values(op, path: Path) -> Path:
    """Persist already-normalized sparse values; index layout and operators unchanged.

    Do not change a CSR core until the full file has been written and flushed.
    """
    S = op.sparse_core
    if S is None:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    check_disk_capacity(path.parent, S.data.nbytes)
    mm = np.lib.format.open_memmap(path, mode='w+', dtype=S.data.dtype, shape=S.data.shape)
    block = 2_000_000
    for s in range(0, len(S.data), block):
        e = min(len(S.data), s + block)
        mm[s:e] = S.data[s:e]
    mm.flush()
    S.data = mm
    return path


def estimate_scratch_bytes(shapes, ks, ms, gs, sym_factor: float = 1.8) -> int:
    """Approximate total persistent scratch for laptop32 mode, not RAM RSS."""
    total = 0
    for (n, d), k, m, g in zip(shapes, ks, ms, gs):
        n = int(n); d = int(d); k = int(k); m = int(m); g = int(g)
        nnz_sym = int(sym_factor * n * k + n)
        total += 4*n*d + 4*nnz_sym*(m+g) + 128*(m+g)
    return int(total * 1.15 + 2*1024**3)


def disk_plan(workspace, needed_bytes: int) -> dict:
    import tempfile
    root = Path(workspace or tempfile.gettempdir())
    # The target workspace may not exist until the first run.
    probe = root
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError(root)
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    safe = int(usage.free) >= int(needed_bytes)
    return {'workspace': str(root), 'scratch_estimate_bytes': int(needed_bytes),
            'free_bytes': int(usage.free), 'safe': bool(safe),
            'message': 'scratch disk headroom available' if safe else
                f'Only {human_bytes(usage.free)} free at {root}; need about {human_bytes(needed_bytes)}. '
                'Select another workspace drive or reduce n/k/kernel count.'}
