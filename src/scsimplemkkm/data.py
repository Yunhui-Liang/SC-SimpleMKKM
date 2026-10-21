from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid
import numpy as np

from .runtime import ProgressSink


@dataclass
class PreparedArray:
    array: np.ndarray
    path: str | None
    normalized: bool
    dtype: str
    shape: tuple[int, int]
    mean_sq_norm: float
    mean_vector_norm_sq: float

    @property
    def global_mean_pairwise_sqdist(self) -> float:
        return max(2.0 * (self.mean_sq_norm - self.mean_vector_norm_sq), np.finfo(np.float32).eps)


def array_signature(X: np.ndarray, name: str = 'view', source: str | None = None) -> str:
    src = source or getattr(X, 'filename', None)
    payload = {
        'name': name,
        'shape': list(X.shape),
        'dtype': str(X.dtype),
        'source': os.path.abspath(str(src)) if src else None,
    }
    if src:
        try:
            st = os.stat(src)
            payload['size'] = st.st_size
            payload['mtime_ns'] = st.st_mtime_ns
        except OSError:
            pass
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def prepare_float32_l2(
    X: np.ndarray,
    *,
    name: str = 'view',
    workspace: str | Path | None = None,
    block_rows: int = 65536,
    memory_threshold_bytes: int = 768 * 1024**2,
    progress: ProgressSink | None = None,
    reuse_cache: bool = True,
) -> PreparedArray:
    """Create an L2-normalized float32 C-order view without a large RAM spike.

    Large arrays are written to an .npy memmap in workspace. This intentionally
    makes normalization an explicit profiled stage rather than relying on a
    hidden temporary allocation inside NumPy/ANN libraries.
    """
    source = getattr(X, 'filename', None)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f'{name} must be 2D')
    n, d = map(int, X.shape)
    bytes_out = n * d * 4
    block_rows = max(1, int(block_rows))
    progress = progress or ProgressSink()

    out_path: Path | None = None
    temp_path: Path | None = None
    if bytes_out > memory_threshold_bytes or isinstance(X, np.memmap) or source is not None and bytes_out > 64*1024**2:
        root = Path(workspace or tempfile.gettempdir()) / 'scsimplemkkm_cache'
        root.mkdir(parents=True, exist_ok=True)
        sig = array_signature(X, name, None if source is None else str(source))
        from .lowmem import check_disk_capacity
        check_disk_capacity(root, bytes_out)
        out_path = root / f'{name}.{sig}.v077.l2f32.npy'
        # Without a source filename/signature, two distinct in-memory arrays
        # can have the same name+shape. Never reuse such a cache.
        reuse_cache = bool(reuse_cache and source is not None)
        marker = out_path.with_suffix(out_path.suffix + '.complete')
        if reuse_cache and out_path.exists() and marker.exists() and out_path.stat().st_size >= bytes_out:
            arr = np.load(out_path, mmap_mode='r+')
            if tuple(arr.shape) == (n, d) and arr.dtype == np.float32:
                # Compute summary statistics in blocks; rows should already be normalized.
                sum_vec = np.zeros(d, dtype=np.float64)
                sum_norm2 = 0.0
                for s in range(0, n, block_rows):
                    b = np.asarray(arr[s:min(n, s + block_rows)], dtype=np.float32)
                    sum_vec += np.sum(b, axis=0, dtype=np.float64)
                    sum_norm2 += float(np.sum(np.einsum('ij,ij->i', b, b), dtype=np.float64))
                mu = sum_vec / max(n, 1)
                return PreparedArray(arr, str(out_path), True, 'float32', (n, d), sum_norm2 / n, float(mu @ mu))
        # Write a unique partial file and publish only after all blocks have
        # completed. Interrupted normalization cannot silently poison a rerun.
        temp_path = out_path.with_name(out_path.name + '.partial_' + uuid.uuid4().hex[:12])
        arr = np.lib.format.open_memmap(temp_path, mode='w+', dtype=np.float32, shape=(n, d))
    else:
        arr = np.empty((n, d), dtype=np.float32, order='C')

    sum_vec = np.zeros(d, dtype=np.float64)
    sum_norm2 = 0.0
    for ib, s in enumerate(range(0, n, block_rows)):
        e = min(n, s + block_rows)
        b64 = np.asarray(X[s:e], dtype=np.float64)
        norms = np.linalg.norm(b64, axis=1)
        norms = np.maximum(norms, np.finfo(np.float64).eps)
        b = (b64 / norms[:, None]).astype(np.float32)
        arr[s:e] = b
        sum_vec += np.sum(b, axis=0, dtype=np.float64)
        sum_norm2 += float(np.sum(np.einsum('ij,ij->i', b, b), dtype=np.float64))
        progress.emit(
            f'normalize:{name}',
            e / n,
            f'Normalizing {name}: {e:,}/{n:,}',
            rows_done=e, rows_total=n,
            rows_start=s, rows_end=e, block_index=ib, block_kind='normalize',
        )
    if isinstance(arr, np.memmap):
        arr.flush()
    if temp_path is not None:
        del arr
        import os
        os.replace(temp_path, out_path)
        # Atomic marker publish: interrupted jobs never look 'complete'.
        marker.write_text('l2f32-v077\n', encoding='ascii')
        arr = np.load(out_path, mmap_mode='r+')
    mu = sum_vec / max(n, 1)
    return PreparedArray(arr, None if out_path is None else str(out_path), True, 'float32', (n, d), sum_norm2 / n, float(mu @ mu))
