"""One persistent workspace contract for both local browser consoles.

All user paths are resolved *before* writing to settings.  No data is copied
or deleted when selecting a legacy cache directory.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

FIELDS = ("data_root", "dataset_root", "mv_root", "runs_root", "plans_root")


def app_root() -> Path:
    return Path(os.environ.get('SCSIMPLEMKKM_APP_ROOT') or os.getcwd()).expanduser().resolve()


def settings_path() -> Path:
    return app_root() / 'workspace_paths.json'


def _default() -> dict:
    root = app_root()
    return {
        'data_root': str(root / 'paper_workspace' / 'data'),
        'dataset_root': str(root / 'datasets'),
        'mv_root': str(root / 'dataset_cache'),
        'runs_root': str(root / 'paper_workspace' / 'runs_interactive'),
        'plans_root': str(root / 'paper_workspace' / 'plans'),
    }


def workspace() -> dict:
    result = _default()
    p = settings_path()
    # Existing downloads copied from 0.8.2 may have been put directly into
    # root/data, root/paper_data, root/dataset_cache, or root itself.  Prefer the
    # directory containing the most prepared manifests unless the user has
    # explicitly saved their chosen directories.
    if not p.is_file():
        home = app_root()
        candidates = [home / x for x in ('paper_workspace/data','paper_data','data','dataset_cache','datasets','.')]
        counts = [len(list(c.glob('*/manifest.json'))) if c.is_dir() else 0 for c in candidates]
        if max(counts, default=0):
            result['data_root'] = str(candidates[counts.index(max(counts))].resolve())
    if p.is_file():
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            for key in FIELDS:
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = str(Path(value).expanduser().resolve())
        except (ValueError, OSError):
            pass
    return {'schema': 'scsimplemkkm.workspace.v1', 'settings_path': str(p), **result}


def update_workspace(changes: dict) -> dict:
    # Independent properties are updated atomically; never silently reset
    # unrelated root paths when the other UI saves one field.
    if not isinstance(changes, dict):
        raise ValueError('workspace changes must be an object')
    unknown = set(changes) - set(FIELDS)
    if unknown:
        raise ValueError(f'unknown workspace fields: {sorted(unknown)}')
    out = workspace()
    for k, value in changes.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{k}: select a non-empty directory path')
        out[k] = str(Path(value).expanduser().resolve())
    p = settings_path()
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({k: out[k] for k in FIELDS}, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)
    return workspace()


def legacy_data_roots(extra=None) -> list[Path]:
    """Shallow legacy layout search: no recursive scan through TB of data."""
    cfg = workspace()
    roots = [Path(cfg['data_root']), Path(cfg['mv_root']), Path(cfg['dataset_root'])]
    home = app_root()
    roots.extend((home / name for name in (
        'paper_workspace/data', 'paper_data', 'data', 'dataset_cache',
        'datasets', 'paper_workspace', '.',
    )))
    if extra is not None:
        roots.insert(0, Path(extra).expanduser().resolve())
    uniq = []
    for p in roots:
        p = p.resolve()
        if p not in uniq:
            uniq.append(p)
    return uniq
