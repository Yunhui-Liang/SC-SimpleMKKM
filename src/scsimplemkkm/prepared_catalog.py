"""Validated, relocatable prepared multi-view dataset registry.

Read-only diagnostics: 'ready' requires *all* view NPY headers and label NPY
header to agree. File relocation is resolved before any data is run; inputs
are never silently replaced or regenerated.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np


def _candidate_path(base: Path, stored: Any) -> Path | None:
    if stored in (None, ''):
        return None
    raw = str(stored)
    p = Path(raw).expanduser()
    # Prefer a local copy, even if an old v0.8.2 absolute path still exists.
    # pathlib.PosixPath('E:\\...').name on Linux is not the file basename;
    # handle old Windows manifests independent of current host OS.
    filename = raw.replace('\\', '/').rstrip('/').split('/')[-1]
    local = base / filename
    if local.is_file():
        return local.resolve()
    if not p.is_absolute():
        p = base / p
    return p.resolve() if p.is_file() else None


def inspect_manifest(manifest_path: str | Path, *, expected_id: str | None = None) -> dict:
    mp = Path(manifest_path).expanduser().resolve()
    base = mp.parent
    obj = {'manifest': str(mp), 'dataset_id': expected_id or base.name,
           'status': 'missing_manifest', 'ready': False, 'message': '', 'views': [], 'labels': None}
    if not mp.is_file():
        obj['message'] = 'manifest.json absent; dataset not prepared'
        return obj
    try:
        m = json.loads(mp.read_text(encoding='utf-8'))
        if not isinstance(m, dict):
            raise ValueError('manifest must be JSON object')
    except (ValueError, OSError) as e:
        obj.update(status='corrupt_manifest', message=f'{type(e).__name__}: {e}')
        return obj
    ident = str(m.get('id') or base.name)
    obj['dataset_id'] = ident
    if expected_id and ident.lower() != expected_id.lower():
        obj.update(status='id_mismatch', message=f'expected recipe {expected_id}, manifest contains {ident}')
        return obj
    views = m.get('views')
    if not isinstance(views, list) or len(views) < 2:
        obj.update(status='invalid_views', message='expected at least 2 aligned views')
        return obj
    lab = _candidate_path(base, m.get('labels'))
    if lab is None:
        obj.update(status='missing_labels', message=f'labels NPY missing or old absolute path: {m.get("labels")}')
        return obj
    try:
        y = np.load(str(lab), mmap_mode='r', allow_pickle=False)
        if y.ndim not in (1, 2) or (y.ndim == 2 and min(y.shape) != 1):
            raise ValueError(f'labels must be a vector, found shape={y.shape}')
        n = int(y.size)
        if n < 2:
            raise ValueError('fewer than 2 labels')
        for i, v in enumerate(views):
            if not isinstance(v, dict):
                v = {'path': v, 'name': f'view{i+1}'}
            path = _candidate_path(base, v.get('path') or v.get('view'))
            if path is None:
                obj.update(status='missing_view', message=f'view[{i+1}] file missing: {v.get("path") or v.get("view")}')
                return obj
            x = np.load(str(path), mmap_mode='r', allow_pickle=False)
            if x.ndim != 2 or int(x.shape[0]) != n or int(x.shape[1]) < 1:
                raise ValueError(f'view[{i+1}] rows/shape {x.shape} != labels n={n}')
            if x.dtype.kind not in 'fiu':
                raise ValueError(f'view[{i+1}] unsupported dtype {x.dtype}')
            if 'd' in v and v['d'] is not None and int(v['d']) != int(x.shape[1]):
                raise ValueError(f'view[{i+1}] declared d={v["d"]}, file d={x.shape[1]}')
            obj['views'].append({'name': str(v.get('name') or f'view{i+1}'),
                                 'path': str(path), 'n': n, 'd': int(x.shape[1]),
                                 'k': int(v.get('k') or m.get('recommended_k') or 10),
                                 'dtype': str(x.dtype)})
            del x
        if m.get('n') and int(m['n']) != n:
            raise ValueError(f'manifest n={m["n"]} differs from labels n={n}')
        if m.get('view_count') and int(m['view_count']) != len(views):
            raise ValueError(f'manifest view_count={m["view_count"]} differs from number of views={len(views)}')
        obj.update(status='ready', ready=True, message='validated NPY headers and aligned row counts',
                   labels=str(lab), n=n, classes=int(m.get('classes') or 0),
                   dataset=m.get('dataset') or ident,
                   benchmark_role=m.get('benchmark_role') or 'unknown',
                   recommended_engine=m.get('recommended_engine') or 'scalable_sparse',
                   recommended_kernel_preset=m.get('recommended_kernel_preset') or 'rbf_self_tuned',
                   recommended_k=int(m.get('recommended_k') or 10),
                   relocated=any(str(v.get('path') if isinstance(v, dict) else v) != d['path'] for v,d in zip(views,obj['views'])))
    except Exception as e:
        obj.update(status='invalid_data', message=f'{type(e).__name__}: {e}', ready=False)
        obj['views'] = []
    return obj


def status_for_recipe(data_root: str | Path, recipe_id: str) -> dict:
    folder = Path(data_root).expanduser().resolve() / recipe_id
    record = inspect_manifest(folder / 'manifest.json', expected_id=recipe_id)
    if record['ready']:
        return record
    # Include previous preparation exceptions and partial downloads in the UI.
    error_file = folder / '_prepare_error.json'
    if error_file.is_file():
        try:
            err = json.loads(error_file.read_text(encoding='utf-8'))
            record['last_failure'] = err
            if record['status'] == 'missing_manifest':
                record.update(status='prepare_failed', message=str(err.get('error') or 'unknown preparation error'))
        except Exception:
            pass
    if record['status'] == 'missing_manifest' and folder.exists():
        names = [p.name for p in folder.iterdir() if p.is_file()]
        if any(x.endswith('.part') or x.endswith('.tmp') for x in names):
            record.update(status='partial_download', message='interrupted download or temporary file found')
        elif names:
            record.update(status='downloaded_only', message='files found, but prepared manifest is absent')
    if record.get('last_failure'):
        record['message'] += f"; last failure [{recipe_id}]: {record['last_failure'].get('error', 'unknown')}"
    return record


def prepared_rows(data_root: str | Path) -> list[dict]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        return []
    out = []
    for mp in sorted(root.glob('*/manifest.json')) + sorted(root.glob('*/*/manifest.json')):
        check = inspect_manifest(mp)
        if check['ready']:
            out.append(check)
    return out
