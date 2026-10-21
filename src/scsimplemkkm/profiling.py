from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import threading
import time
from typing import Callable

from .runtime import _rss_bytes, _read_meminfo, _read_proc_io


@dataclass
class StageRecord:
    stage: str
    seconds: float
    cpu_seconds: float
    cpu_core_equiv: float
    read_bytes: int
    write_bytes: int
    rss_start_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int
    ram_available_start_bytes: int
    ram_available_end_bytes: int
    ram_available_min_bytes: int
    detail: dict


class ProfileBook:
    """Stage profiler with low-rate external resource samples.

    Wall time uses perf_counter. Peak RSS is updated from RuntimeMonitor samples
    and from start/end snapshots, so no tracing profiler is inserted in hot code.
    """

    def __init__(self, callback: Callable[[dict], None] | None = None) -> None:
        self.records: list[StageRecord] = []
        self.callback = callback
        self._lock = threading.Lock()
        self._active_stage: str | None = None
        self._active_peak_rss = 0
        self._active_min_available = 0

    def observe(self, sample: dict) -> None:
        with self._lock:
            if self._active_stage is None:
                return
            rss = int(sample.get('process_rss_bytes') or 0)
            av = int(sample.get('ram_available_bytes') or 0)
            self._active_peak_rss = max(self._active_peak_rss, rss)
            if av > 0:
                self._active_min_available = av if self._active_min_available <= 0 else min(self._active_min_available, av)

    @contextmanager
    def stage(self, name: str, **detail):
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        io0 = _read_proc_io()
        rss0 = _rss_bytes()
        av0 = int(_read_meminfo().get('MemAvailable', 0))
        with self._lock:
            self._active_stage = name
            self._active_peak_rss = rss0
            self._active_min_available = av0
        if self.callback:
            self.callback({'type': 'profile_stage_start', 'stage': name, 'time': time.time(), 'detail': detail})
        try:
            yield
        finally:
            sec = time.perf_counter() - t0
            cpu_sec = time.process_time() - cpu0
            io1 = _read_proc_io()
            rss1 = _rss_bytes()
            av1 = int(_read_meminfo().get('MemAvailable', 0))
            with self._lock:
                peak = max(self._active_peak_rss, rss0, rss1)
                min_av = self._active_min_available
                if av1 > 0:
                    min_av = av1 if min_av <= 0 else min(min_av, av1)
                self._active_stage = None
            rec = StageRecord(
                stage=name,
                seconds=float(sec),
                cpu_seconds=float(cpu_sec),
                cpu_core_equiv=float(cpu_sec / max(sec, 1e-12)),
                read_bytes=int(max(0, io1.get('read_bytes',0)-io0.get('read_bytes',0))),
                write_bytes=int(max(0, io1.get('write_bytes',0)-io0.get('write_bytes',0))),
                rss_start_bytes=int(rss0),
                rss_end_bytes=int(rss1),
                rss_peak_bytes=int(peak),
                ram_available_start_bytes=av0,
                ram_available_end_bytes=av1,
                ram_available_min_bytes=int(min_av),
                detail=dict(detail),
            )
            self.records.append(rec)
            if self.callback:
                self.callback({'type': 'profile_stage_end', 'time': time.time(), **asdict(rec)})

    def as_dicts(self) -> list[dict]:
        return [asdict(x) for x in self.records]

    def write_json(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dicts(), indent=2, ensure_ascii=False), encoding='utf-8')
        return p

    def write_csv(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        fields = ['stage','seconds','cpu_seconds','cpu_core_equiv','read_bytes','write_bytes','rss_start_bytes','rss_end_bytes','rss_peak_bytes','ram_available_start_bytes','ram_available_end_bytes','ram_available_min_bytes','detail']
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for rec in self.records:
                row = asdict(rec); row['detail'] = json.dumps(row['detail'], ensure_ascii=False, sort_keys=True); w.writerow(row)
        return p
