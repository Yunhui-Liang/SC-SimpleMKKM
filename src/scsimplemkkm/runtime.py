from __future__ import annotations

from dataclasses import dataclass, asdict
import contextlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Callable, Iterable

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    psutil = None


GB = 1024 ** 3
MB = 1024 ** 2


def _read_meminfo() -> dict[str, int]:
    # Use psutil first so stage-level profiling works on Windows/macOS as well
    # as Linux.  Keep Linux /proc parsing as the zero-dependency fallback.
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return {
                'MemTotal': int(vm.total),
                'MemAvailable': int(vm.available),
                'MemFree': int(getattr(vm, 'free', vm.available)),
            }
        except Exception:
            pass
    out: dict[str, int] = {}
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                key, rest = line.split(':', 1)
                val = rest.strip().split()[0]
                out[key] = int(val) * 1024
    except OSError:
        pass
    return out


def _rss_bytes() -> int:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    # Portable fallback. Linux ru_maxrss is KiB; macOS is bytes.
    try:
        import resource
        x = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return x if platform.system() == 'Darwin' else x * 1024
    except Exception:
        return 0


def _cpu_topology() -> tuple[int, int]:
    logical = os.cpu_count() or 1
    physical = 0
    try:
        sockets: set[tuple[str, str]] = set()
        phys_id = None
        core_id = None
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if phys_id is not None and core_id is not None:
                        sockets.add((phys_id, core_id))
                    phys_id = core_id = None
                    continue
                if line.startswith('physical id'):
                    phys_id = line.split(':', 1)[1].strip()
                elif line.startswith('core id'):
                    core_id = line.split(':', 1)[1].strip()
        if phys_id is not None and core_id is not None:
            sockets.add((phys_id, core_id))
        physical = len(sockets)
    except OSError:
        pass
    if physical <= 0:
        physical = logical
    return physical, logical


def _cpu_model() -> str:
    try:
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.lower().startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or 'unknown'


def system_info() -> dict:
    if psutil is not None:
        vm = psutil.virtual_memory()
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        physical = psutil.cpu_count(logical=False) or logical
        model = _cpu_model()
        if platform.system() == 'Windows':
            model = os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor() or model
        return {
            'platform': platform.platform(), 'os_name': platform.system(),
            'cpu_model': model, 'physical_cores': int(physical), 'logical_cores': int(logical),
            'ram_total_bytes': int(vm.total), 'ram_available_bytes': int(vm.available),
        }
    mem = _read_meminfo(); physical, logical = _cpu_topology()
    return {'platform': platform.platform(), 'os_name': platform.system(), 'cpu_model': _cpu_model(),
            'physical_cores': physical, 'logical_cores': logical,
            'ram_total_bytes': int(mem.get('MemTotal', 0)),
            'ram_available_bytes': int(mem.get('MemAvailable', mem.get('MemFree', 0)))}


@dataclass
class ResourcePolicy:
    """Conservative single-node resource policy.

    max_ram_fraction is applied to total RAM, then reserve_gb is subtracted as a
    hard OS/user safety margin. CPU threads default to a fraction of physical
    cores while leaving leave_cores idle. These are execution controls, not
    algorithm hyperparameters.
    """

    max_ram_fraction: float = 0.80
    reserve_gb: float = 4.0
    max_cpu_fraction: float = 0.85
    leave_cores: int = 2
    max_threads: int | None = None
    monitor_interval: float = 1.0
    allow_oom_risk: bool = False

    def normalized(self) -> 'ResourcePolicy':
        return ResourcePolicy(
            max_ram_fraction=float(min(max(self.max_ram_fraction, 0.20), 0.98)),
            reserve_gb=float(max(self.reserve_gb, 0.0)),
            max_cpu_fraction=float(min(max(self.max_cpu_fraction, 0.10), 1.0)),
            leave_cores=int(max(self.leave_cores, 0)),
            max_threads=None if self.max_threads is None else int(max(self.max_threads, 1)),
            monitor_interval=float(max(self.monitor_interval, 0.25)),
            allow_oom_risk=bool(self.allow_oom_risk),
        )

    def recommended_threads(self) -> int:
        info = system_info(); p, l = int(info['physical_cores']), int(info['logical_cores'])
        base = max(1, p - self.leave_cores)
        frac = max(1, int(math.floor(p * self.max_cpu_fraction)))
        t = min(base, frac)
        if self.max_threads is not None:
            t = min(t, self.max_threads)
        return max(1, min(t, l))

    def memory_budget_bytes(self) -> int:
        total = int(system_info().get('ram_total_bytes', 0))
        if total <= 0:
            return 0
        return max(0, int(total * self.max_ram_fraction - self.reserve_gb * GB))

    def as_dict(self) -> dict:
        d = asdict(self)
        d['recommended_threads'] = self.recommended_threads()
        d['memory_budget_bytes'] = self.memory_budget_bytes()
        return d


def configure_thread_limits(policy: ResourcePolicy) -> dict:
    """Best-effort BLAS/OpenMP/FAISS thread limiting.

    Environment variables are useful for subprocesses and libraries initialized
    later. threadpoolctl adjusts already loaded BLAS/OpenMP libraries when it is
    available through scikit-learn.
    """
    policy = policy.normalized()
    n = policy.recommended_threads()
    env_names = [
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
    ]
    for name in env_names:
        os.environ[name] = str(n)
    applied = {'threads': n, 'env': {x: os.environ.get(x) for x in env_names}}
    try:
        from threadpoolctl import threadpool_limits
        controller = threadpool_limits(limits=n)
        # Keep the controller alive for process lifetime; threadpoolctl restores
        # limits on __exit__, so deliberately retain it.
        applied['_threadpool_controller'] = controller
    except Exception:
        pass
    try:
        import faiss  # type: ignore
        faiss.omp_set_num_threads(n)
        applied['faiss_threads'] = n
    except Exception:
        pass
    try:
        from .spmm import set_numba_threads
        nt = set_numba_threads(n)
        if nt is not None:
            applied['numba_threads'] = int(nt)
    except Exception:
        pass
    return applied


def _read_proc_cpu() -> tuple[int, int] | None:
    try:
        with open('/proc/stat', 'r', encoding='utf-8') as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return total, idle
    except Exception:
        return None



def _read_proc_cpu_all() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    try:
        with open('/proc/stat', 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('cpu'):
                    break
                parts = line.split()
                name = parts[0]
                vals = [int(x) for x in parts[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                out[name] = (sum(vals), idle)
    except Exception:
        pass
    return out


def _read_proc_io() -> dict[str, int]:
    if psutil is not None:
        try:
            io = psutil.Process(os.getpid()).io_counters()
            return {'read_bytes': int(io.read_bytes), 'write_bytes': int(io.write_bytes)}
        except Exception:
            pass
    out: dict[str, int] = {}
    try:
        with open('/proc/self/io', 'r', encoding='utf-8') as f:
            for line in f:
                k, v = line.split(':', 1)
                out[k.strip()] = int(v.strip())
    except Exception:
        pass
    return out

def _read_proc_self_cpu() -> int | None:
    try:
        with open('/proc/self/stat', 'r', encoding='utf-8') as f:
            vals = f.read().split()
        # utime, stime are fields 14 and 15 (1-indexed)
        return int(vals[13]) + int(vals[14])
    except Exception:
        return None


class SystemSampler:
    def __init__(self) -> None:
        self.logical = os.cpu_count() or 1
        self.prev_t = time.monotonic()
        if psutil is not None:
            self.proc = psutil.Process(os.getpid())
            self.proc.cpu_percent(None); psutil.cpu_percent(None, percpu=True)
            try: self.prev_io_ps = self.proc.io_counters()
            except Exception: self.prev_io_ps = None
            return
        self.prev_sys = _read_proc_cpu_all(); self.prev_proc = _read_proc_self_cpu(); self.prev_io = _read_proc_io()
        try: self.clk_tck = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
        except Exception: self.clk_tck = 100

    def sample(self) -> dict:
        now=time.monotonic(); wall=max(now-self.prev_t,1e-9); self.prev_t=now
        if psutil is not None:
            per=[float(x) for x in psutil.cpu_percent(None,percpu=True)]
            sys_pct=float(sum(per)/max(len(per),1)); raw=float(self.proc.cpu_percent(None))
            proc_cores=max(0.0,raw/100.0); proc_capacity=100.0*proc_cores/max(self.logical,1)
            vm=psutil.virtual_memory(); rss=int(self.proc.memory_info().rss)
            rr=ww=None
            try:
                io_now=self.proc.io_counters()
                if self.prev_io_ps is not None:
                    rr=max(0.0,(io_now.read_bytes-self.prev_io_ps.read_bytes)/wall); ww=max(0.0,(io_now.write_bytes-self.prev_io_ps.write_bytes)/wall)
                self.prev_io_ps=io_now
            except Exception: pass
            return {'time':time.time(),'system_cpu_percent':sys_pct,'process_cpu_percent':proc_capacity,'process_cpu_cores':proc_cores,
                    'per_core_cpu_percent':per,'process_rss_bytes':rss,'ram_total_bytes':int(vm.total),'ram_available_bytes':int(vm.available),
                    'disk_read_bytes_per_sec':rr,'disk_write_bytes_per_sec':ww}
        sys_now=_read_proc_cpu_all(); proc_now=_read_proc_self_cpu(); io_now=_read_proc_io(); per_core=[]; sys_pct=None
        if 'cpu' in self.prev_sys and 'cpu' in sys_now:
            a,b=self.prev_sys['cpu'],sys_now['cpu']; dtot,didle=b[0]-a[0],b[1]-a[1]
            if dtot>0: sys_pct=100.0*(dtot-didle)/dtot
        for i in range(self.logical):
            key=f'cpu{i}'
            if key in self.prev_sys and key in sys_now:
                a,b=self.prev_sys[key],sys_now[key]; dtot,didle=b[0]-a[0],b[1]-a[1]
                per_core.append(float(max(0.0,min(100.0,100.0*(dtot-didle)/dtot))) if dtot>0 else 0.0)
        proc_capacity=proc_cores=None
        if self.prev_proc is not None and proc_now is not None:
            cpu_seconds=(proc_now-self.prev_proc)/float(self.clk_tck); proc_cores=max(0.0,cpu_seconds/wall); proc_capacity=100.0*proc_cores/self.logical
        rr=ww=None
        if self.prev_io and io_now:
            rr=max(0.0,(io_now.get('read_bytes',0)-self.prev_io.get('read_bytes',0))/wall); ww=max(0.0,(io_now.get('write_bytes',0)-self.prev_io.get('write_bytes',0))/wall)
        self.prev_sys,self.prev_proc,self.prev_io=sys_now,proc_now,io_now; mem=_read_meminfo()
        return {'time':time.time(),'system_cpu_percent':sys_pct,'process_cpu_percent':proc_capacity,'process_cpu_cores':proc_cores,'per_core_cpu_percent':per_core,
                'process_rss_bytes':int(_rss_bytes()),'ram_total_bytes':int(mem.get('MemTotal',0)),'ram_available_bytes':int(mem.get('MemAvailable',mem.get('MemFree',0))),
                'disk_read_bytes_per_sec':rr,'disk_write_bytes_per_sec':ww}


class CancelledError(RuntimeError):
    pass


class RunControl:
    def __init__(self) -> None:
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError('Run cancelled by user')


class ProgressSink:
    """Small callback adapter used throughout the pipeline."""

    def __init__(self, callback: Callable[[dict], None] | None = None, control: RunControl | None = None):
        self.callback = callback
        self.control = control

    def emit(self, stage: str, progress: float | None = None, message: str | None = None, **detail) -> None:
        if self.control is not None:
            self.control.check()
        if self.callback is None:
            return
        event = {
            'time': time.time(),
            'stage': str(stage),
            'progress': None if progress is None else float(min(max(progress, 0.0), 1.0)),
            'message': message,
            'detail': detail,
        }
        self.callback(event)


class RuntimeMonitor:
    """Low-overhead daemon sampler. No profiling hooks are injected in hot loops."""

    def __init__(
        self,
        callback: Callable[[dict], None] | None = None,
        interval: float = 1.0,
        control: RunControl | None = None,
        reserve_bytes: int = 0,
    ) -> None:
        self.callback = callback
        self.interval = max(float(interval), 0.25)
        self.control = control
        self.reserve_bytes = max(int(reserve_bytes), 0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampler = SystemSampler()
        self.peak_rss_bytes = 0

    def start(self) -> 'RuntimeMonitor':
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name='scsimplemkkm-monitor', daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            s = self._sampler.sample()
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(s['process_rss_bytes']))
            if (
                self.control is not None
                and self.reserve_bytes > 0
                and s['ram_available_bytes'] > 0
                and s['ram_available_bytes'] < self.reserve_bytes
            ):
                self.control.cancel_event.set()
                s['resource_warning'] = 'Available RAM fell below the configured reserve; cancellation requested.'
            if self.callback is not None:
                self.callback(s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 2 * self.interval))
            self._thread = None

    def __enter__(self) -> 'RuntimeMonitor':
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def human_bytes(n: int | float) -> str:
    x = float(n)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(x) < 1024.0 or unit == 'TiB':
            return f'{x:.2f} {unit}'
        x /= 1024.0
    return f'{x:.2f} TiB'


@dataclass
class MemoryEstimate:
    normalized_views_bytes: int
    supports_bytes: int
    kernel_cores_bytes: int
    embedding_work_bytes: int
    ann_index_bytes: int
    ann_workspace_bytes: int
    misc_bytes: int

    @property
    def total_bytes(self) -> int:
        return int(sum(asdict(self).values()))

    def as_dict(self) -> dict:
        d = asdict(self)
        d['total_bytes'] = self.total_bytes
        d['total_human'] = human_bytes(self.total_bytes)
        return d


def estimate_peak_memory(
    view_shapes: Iterable[tuple[int, int]],
    k_by_view: Iterable[int],
    kernels_per_view: Iterable[int],
    families_per_view: Iterable[int],
    n_clusters: int,
    candidate_factor: int = 4,
    sym_factor: float = 1.8,
    ann_backend: str = 'auto',
    hnsw_m: int = 32,
    memory_mode: str = 'standard',
) -> MemoryEstimate:
    """Conservative planning model for the current float32/CSR implementation.

    It intentionally errs high. Kernel sparse cores are estimated as float32
    values + int32 column indices + int32 row pointers. CSR structure sharing
    inside support families is not credited here, so real use can be lower.
    """
    shapes = list(view_shapes)
    from .lowmem import resolve_memory_mode
    resolved_mode = resolve_memory_mode(memory_mode, max((int(n) for n, _ in shapes), default=0))
    ks = list(k_by_view)
    ms = list(kernels_per_view)
    gs = list(families_per_view)
    if not (len(shapes) == len(ks) == len(ms) == len(gs)):
        raise ValueError('view_shapes, k_by_view, kernels_per_view and families_per_view must match')
    normalized = 0
    supports = 0
    kernels = 0
    ann_index = 0
    ann_ws = 0
    n_ref = 0
    for (n, d), k, m, g in zip(shapes, ks, ms, gs):
        n_ref = max(n_ref, int(n))
        normalized += int(n) * int(d) * 4
        # support: CSR ptr + int32 indices + float32 similarities
        supports += int(g) * (8 * int(n) * int(k) + 8 * (int(n) + 1))
        nnz_sym = int(math.ceil(sym_factor * int(n) * int(k) + int(n)))
        if resolved_mode == 'laptop32':
            # CSR values are backed by disk. Resident planning allows roughly
            # one active core plus small per-kernel stats and shared indices.
            kernels += (4 * nnz_sym * int(min(g, m))
                        + 4 * (int(n) + 1) * int(min(g, m))
                        + 12 * int(n) * int(m))
        else:
            kernels += int(m) * (8 * nnz_sym + 4 * (int(n) + 1))
        # ANN indexes generally own their own vector representation. IVF-SQ8
        # stores approximately one byte per feature dimension plus ids/coarse
        # metadata, while IVF-Flat stores float32 vectors.  current auto mode prefers
        # IVF-SQ8 for million-scale or multi-GB feature banks when FAISS exists.
        vec = int(n) * int(d) * 4
        backend = ann_backend.lower().replace('-', '_')
        if backend in {'faiss_ivfsq', 'faiss_ivf_sq', 'faiss_ivf_sq8'}:
            ann_index += int(n) * int(d) + int(n) * 12
        elif backend in {'faiss_ivf', 'ivf'}:
            ann_index += vec + int(n) * 8
        elif backend == 'auto' and (int(n) >= 1_000_000 or vec >= 2 * GB):
            # Reflect the preferred current low-memory path. Runtime still reports the
            # actual backend used, so a fallback can be identified explicitly.
            ann_index += int(n) * int(d) + int(n) * 12
        elif backend in {'sklearn', 'exact'}:
            ann_index += 0
        else:
            ann_index += vec + int(n) * int(max(hnsw_m, 8)) * 8
        # one ANN query batch / candidates and implementation overhead proxy
        ann_ws += int(min(n, 65536)) * max(int(candidate_factor) * int(k), int(k) + 8) * 8
    # Search uses float32, but strict accepted-state solves use float64 and
    # iterative eigensolvers may hold several n*c slabs. Reserve eight float64
    # slabs so the preflight remains conservative for 1M-10M runs.
    embed = 8 * n_ref * int(n_clusters) * 8
    misc = int(0.20 * (normalized + supports + kernels + embed + ann_index)) + 768 * MB
    return MemoryEstimate(normalized, supports, kernels, embed, ann_index, ann_ws, misc)


def ensure_memory_safe(estimate: MemoryEstimate, policy: ResourcePolicy) -> dict:
    p = policy.normalized()
    budget = p.memory_budget_bytes()
    info = system_info()
    available = int(info.get('ram_available_bytes', 0))
    safe = True
    reasons: list[str] = []
    if budget > 0 and estimate.total_bytes > budget:
        safe = False
        reasons.append(
            f'estimated peak {human_bytes(estimate.total_bytes)} exceeds configured budget {human_bytes(budget)}'
        )
    if available > 0 and estimate.total_bytes > max(0, available - int(p.reserve_gb * GB)):
        safe = False
        reasons.append(
            f'estimated peak {human_bytes(estimate.total_bytes)} exceeds currently available RAM after reserve'
        )
    if not safe and not p.allow_oom_risk:
        raise MemoryError('; '.join(reasons))
    return {'safe': safe, 'reasons': reasons, 'estimate': estimate.as_dict(), 'policy': p.as_dict(), 'system': info}


def estimate_dense_reference_peak_memory(
    view_shapes: Iterable[tuple[int, int]],
    kernels_per_view: Iterable[int],
    n_clusters: int,
) -> MemoryEstimate:
    """Conservative RAM plan for the dense reference SimpleMKKM engine.

    Stores every base kernel as float64 plus one fused dense float64 matrix,
    normalized feature views, and several n*c work arrays.  It deliberately
    ignores possible allocator reuse so that Windows preflight rejects risky
    dense runs rather than relying on the OOM killer / pagefile.
    """
    shapes = list(view_shapes)
    ms = list(kernels_per_view)
    if len(shapes) != len(ms):
        raise ValueError('view_shapes and kernels_per_view must match')
    normalized = 0
    kernels = 0
    n_ref = 0
    for (n, d), m in zip(shapes, ms):
        n = int(n); d = int(d); m = int(m)
        n_ref = max(n_ref, n)
        normalized += n * d * 4
        kernels += m * n * n * 8
    # One similarity/all-pairs scratch and one fused dense kernel in float64.
    dense_scratch = 2 * n_ref * n_ref * 8
    # eigsh / embedding work.  Keep a generous c-dependent allowance.
    embed = 10 * n_ref * max(int(n_clusters), 1) * 8
    misc = dense_scratch + int(0.12 * (normalized + kernels + embed)) + 512 * MB
    return MemoryEstimate(normalized, 0, kernels, embed, 0, 0, misc)
