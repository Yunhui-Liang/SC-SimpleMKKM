from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


def _git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=None if cwd is None else str(cwd), stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
    except Exception:
        return None


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


@dataclass
class RunJournal:
    run_dir: Path
    run_id: str
    config: dict
    system: dict
    plan: dict | None = None

    @classmethod
    def create(cls, run_dir: str | Path, config: dict, system: dict, plan: dict | None = None) -> 'RunJournal':
        root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
        rid = time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}'
        j = cls(root, rid, dict(config), dict(system), None if plan is None else dict(plan))
        j._lock = threading.Lock()
        j._events = (root / 'events.jsonl').open('a', encoding='utf-8', buffering=1)
        j._resources = (root / 'resources.jsonl').open('a', encoding='utf-8', buffering=1)
        j.state = {
            'run_id': rid, 'status': 'running', 'started_at': time.time(), 'finished_at': None,
            'config': config, 'system': system, 'plan': plan, 'git_commit': _git_commit(),
            'python': sys.version, 'platform': platform.platform(), 'last_event': None,
            'warnings': [], 'outputs': {}, 'error': None,
        }
        j._write_handoff()
        return j

    def progress(self, ev: dict) -> None:
        rec = {'kind': 'progress', **ev}
        with self._lock:
            self._events.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            self.state['last_event'] = rec
        # Keep the handoff current without rewriting it for every tight block.
        now = time.monotonic()
        if now - getattr(self, '_last_handoff_write', 0.0) >= 5.0:
            self._last_handoff_write = now
            self._write_handoff()

    def resource(self, sample: dict) -> None:
        rec = {'kind': 'resource', **sample}
        with self._lock:
            self._resources.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

    def warning(self, message: str) -> None:
        with self._lock:
            self.state['warnings'].append({'time': time.time(), 'message': str(message)})
        self._write_handoff()

    def finish(self, *, status: str, outputs: dict | None = None, result_summary: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            self.state['status'] = status
            self.state['finished_at'] = time.time()
            if outputs:
                self.state['outputs'] = outputs
            if result_summary:
                self.state['result_summary'] = result_summary
            self.state['error'] = error
        self._write_handoff()
        try: self._events.close()
        except Exception: pass
        try: self._resources.close()
        except Exception: pass

    def fail_from_exception(self, exc: BaseException) -> None:
        self.finish(status='failed', error=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def _write_handoff(self) -> None:
        with self._lock:
            state = dict(self.state)
        _atomic_json(self.run_dir / 'handoff.json', state)
        last = state.get('last_event') or {}
        d = last.get('detail') or {}
        outputs = state.get('outputs') or {}
        lines = [
            '# SC-SimpleMKKM Run Handoff', '',
            f"- run_id: `{state.get('run_id')}`",
            f"- status: **{state.get('status')}**",
            f"- started_at_epoch: `{state.get('started_at')}`",
            f"- finished_at_epoch: `{state.get('finished_at')}`",
            f"- git_commit: `{state.get('git_commit')}`",
            '', '## Last known execution point',
            f"- stage: `{last.get('stage')}`",
            f"- progress: `{last.get('progress')}`",
            f"- message: {last.get('message')}",
        ]
        if d:
            lines += ['- detail: `' + json.dumps(d, ensure_ascii=False, default=str) + '`']
        lines += ['', '## Outputs']
        for k, v in outputs.items(): lines.append(f'- {k}: `{v}`')
        lines += ['', '## Files to send back for diagnosis',
                  '- `handoff.json` (authoritative run state)',
                  '- `events.jsonl` (stage/block events)',
                  '- `resources.jsonl` (CPU/RAM samples)',
                  '- `*.profile.csv` / `*.profile.json` (stage profile)',
                  '- result metadata JSON and any error traceback.',
                  '', '## Handoff note',
                  'When continuing this run in a later conversation, attach this run directory or at least the files above. The diagnostics can be reconstructed without relying on chat memory.']
        if state.get('warnings'):
            lines += ['', '## Warnings'] + [f"- {x['message']}" for x in state['warnings']]
        if state.get('error'):
            lines += ['', '## Error', '```text', str(state['error'])[-12000:], '```']
        (self.run_dir / 'HANDOFF.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
