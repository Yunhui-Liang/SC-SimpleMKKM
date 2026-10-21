from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import secrets
from typing import Any


def _safe_slug(text: str, fallback: str = 'run') -> str:
    text = str(text or '').strip()
    if not text:
        return fallback
    # Keep CJK letters/digits, ASCII letters/digits, dash/underscore. Windows-safe.
    text = re.sub(r'[\\/:*?"<>|\s]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-._ ')
    return (text[:48] or fallback)


def make_run_id(task_name: str | None = None) -> str:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    slug = _safe_slug(task_name or 'run')
    return f'{ts}-{slug}-{secrets.token_hex(2)}'


def managed_config(config: dict) -> tuple[dict, str, Path]:
    """Return a copy with collision-free managed output paths.

    Web runs are immutable task directories. CLI callers can still pass explicit
    out/run_dir directly to run_multiview_config; the web layer calls this helper.
    """
    cfg = json.loads(json.dumps(config, ensure_ascii=False))
    run_root = Path(str(cfg.get('runs_root') or 'runs'))
    run_id = str(cfg.get('run_id') or make_run_id(cfg.get('task_name')))
    run_dir = run_root / run_id
    # If a manually supplied id collides, never overwrite it.
    if run_dir.exists():
        run_id = make_run_id(cfg.get('task_name'))
        run_dir = run_root / run_id
    cfg['run_id'] = run_id
    cfg['run_dir'] = str(run_dir)
    cfg['out'] = str(run_dir / 'result.npz')
    cfg['workspace'] = str(run_dir / 'workspace')
    cfg['runs_root'] = str(run_root)
    return cfg, run_id, run_dir


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def summarize_run_dir(run_dir: Path) -> dict[str, Any] | None:
    result = _read_json(run_dir / 'result.json')
    handoff = _read_json(run_dir / 'handoff.json')
    config = _read_json(run_dir / 'config_resolved.json')
    if result is None and handoff is None and config is None:
        return None
    cfg = (result or {}).get('config') or (handoff or {}).get('config') or config or {}
    run_id = str((result or {}).get('run_id') or cfg.get('run_id') or run_dir.name)
    status = str((result or {}).get('simplemkkm_status') or (handoff or {}).get('status') or ('done' if result else 'unknown'))
    evaluation = (result or {}).get('evaluation') or {}
    ext = evaluation.get('external') or {}
    inte = evaluation.get('internal') or {}
    shapes = (result or {}).get('shapes') or {}
    views = list(shapes) if isinstance(shapes, dict) else []
    n = (result or {}).get('labels_count')
    if n is None and shapes:
        try:
            n = next(iter(shapes.values()))[0]
        except Exception:
            pass
    started = (handoff or {}).get('started_at') or (handoff or {}).get('created_at')
    finished = (handoff or {}).get('finished_at')
    return {
        'run_id': run_id,
        'task_name': cfg.get('task_name') or run_dir.name,
        'engine': (result or {}).get('engine') or cfg.get('engine') or 'scalable_sparse',
        'status': status,
        'run_dir': str(run_dir),
        'started_at': started,
        'finished_at': finished,
        'n': n,
        'views': len(views) if views else len(cfg.get('views') or []),
        'clusters': (result or {}).get('clusters') or cfg.get('clusters'),
        'kernels': len((result or {}).get('gamma') or []),
        'total_seconds': (result or {}).get('total_seconds'),
        'peak_rss_bytes': (result or {}).get('peak_rss_bytes'),
        'ACC': ext.get('ACC'), 'NMI': ext.get('NMI'), 'ARI': ext.get('ARI'),
        'inertia_sse': inte.get('inertia_sse'),
        'result_json': str(run_dir / 'result.json') if (run_dir / 'result.json').exists() else None,
        'run_report_html': str(run_dir / 'RUN_REPORT.html') if (run_dir / 'RUN_REPORT.html').exists() else None,
    }


def list_runs(run_root: str | Path = 'runs', limit: int = 200) -> list[dict[str, Any]]:
    root = Path(run_root)
    if not root.exists():
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        s = summarize_run_dir(p)
        if s is None:
            continue
        try:
            mt = max((p / 'result.json').stat().st_mtime if (p / 'result.json').exists() else 0, p.stat().st_mtime)
        except Exception:
            mt = 0
        rows.append((mt, s))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in rows[: max(1, int(limit))]]


def load_run(run_id: str, run_root: str | Path = 'runs') -> dict[str, Any]:
    rid = Path(str(run_id)).name
    root = Path(run_root)
    p = root / rid
    result = _read_json(p / 'result.json')
    if result is None:
        raise FileNotFoundError(f'run {rid!r} has no result.json')
    return result


def report_path(run_id: str, run_root: str | Path = 'runs') -> Path:
    rid = Path(str(run_id)).name
    p = Path(run_root) / rid / 'RUN_REPORT.html'
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p
