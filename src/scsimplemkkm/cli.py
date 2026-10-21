from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .pipeline import fit_sc_simplemkkm, fit_sc_simplemkkm_multiview
from .dense_reference import fit_dense_reference_multiview
from .support import graph_diagnostics
from .kernels import default_12_kernel_specs, kernel_specs_from_rows, kernel_spec_to_dict, kernel_preset_specs
from .runtime import (
    ResourcePolicy, RunControl, RuntimeMonitor, ProgressSink, configure_thread_limits,
    estimate_peak_memory, estimate_dense_reference_peak_memory, ensure_memory_safe, system_info, human_bytes,
)
from .profiling import ProfileBook
from .evaluation import evaluate_clustering, official_simplemkkm_external_protocol
from .handoff import RunJournal
from .reporting import write_run_report, write_paper_summary, write_tabular_artifacts, write_artifact_manifest
from .trajectory import TrajectoryConfig, IterationTrajectoryObserver
from .diagnostics import ground_truth_diagnostic, graph_diagnostic_summary, spectral_boundary_diagnostic, aggregate_run_warnings
from . import __version__ as SOFTWARE_VERSION


def load_array(path: str, key: str | None = None, mmap: bool = True) -> np.ndarray:
    p = Path(path)
    if p.suffix == '.npy':
        return np.load(p, mmap_mode='r' if mmap else None)
    if p.suffix == '.npz':
        z = np.load(p, mmap_mode='r' if mmap else None)
        if key is None:
            if len(z.files) != 1:
                raise ValueError(f'NPZ contains {z.files}; pass a key')
            key = z.files[0]
        return z[key]
    if p.suffix == '.mat':
        from scipy.io import loadmat
        if key is None:
            key = 'X'
        obj = loadmat(p)
        if key not in obj:
            raise KeyError(f"'{key}' not found in MAT file; keys={list(obj)}")
        return obj[key]
    raise ValueError('Supported inputs: .npy, .npz, .mat (non-v7.3)')


def _package_status(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def doctor_report() -> dict:
    info = system_info()
    return {
        'software_version': SOFTWARE_VERSION,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'numpy': np.__version__,
        'faiss': _package_status('faiss'),
        'hnswlib': _package_status('hnswlib'),
        'pynndescent': _package_status('pynndescent'),
        'threadpoolctl': _package_status('threadpoolctl'),
        'psutil': _package_status('psutil'),
        **info,
    }


def doctor() -> int:
    print(json.dumps(doctor_report(), indent=2, ensure_ascii=False))
    return 0


def resource_policy_from_config(config: dict) -> ResourcePolicy:
    r = config.get('resources', {}) or {}
    return ResourcePolicy(
        max_ram_fraction=float(r.get('max_ram_fraction', 0.80)),
        reserve_gb=float(r.get('reserve_gb', 4.0)),
        max_cpu_fraction=float(r.get('max_cpu_fraction', 0.85)),
        leave_cores=int(r.get('leave_cores', 2)),
        max_threads=None if r.get('max_threads') in (None, '', 0) else int(r.get('max_threads')),
        monitor_interval=float(r.get('monitor_interval', 1.0)),
        allow_oom_risk=bool(r.get('allow_oom_risk', False)),
    ).normalized()


def _view_headers(config: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    views: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    mmap = bool(config.get('mmap', True))
    common_k = int(config.get('k', 10))
    for i, vc in enumerate(config.get('views', [])):
        name = str(vc.get('name') or f'view{i+1}')
        X = load_array(str(vc['path']), vc.get('key'), mmap=mmap)
        if X.ndim != 2:
            raise ValueError(f'{name} must be 2D')
        views[name] = X
        rows.append({'name': name, 'shape': [int(X.shape[0]), int(X.shape[1])], 'dtype': str(X.dtype), 'k': int(vc.get('k', common_k)), 'path': os.path.abspath(str(vc['path']))})
    return rows, views


def _kernel_specs_from_entry(entry, *, fallback_preset: str = 'historical12', fallback_scale: float = 1.0, fallback_local_scale_k: int = 7) -> list:
    """Resolve either legacy explicit rows or the compact v0.7.3 preset form."""
    if entry is None:
        return kernel_preset_specs(fallback_preset, single_rbf_scale=fallback_scale, local_scale_k=fallback_local_scale_k)
    if isinstance(entry, (list, tuple)):
        return kernel_specs_from_rows(entry)
    if isinstance(entry, dict):
        if entry.get('preset'):
            return kernel_preset_specs(
                str(entry['preset']),
                single_rbf_scale=float(entry.get('single_rbf_scale', fallback_scale)),
                local_scale_k=int(entry.get('local_scale_k', fallback_local_scale_k)),
            )
        rows = entry.get('rows')
        if rows is None:
            rows = entry.get('kernels')
        if rows is not None:
            return kernel_specs_from_rows(rows)
    raise TypeError('kernel bank entry must be a preset mapping or a list of explicit kernel rows')


def _resolved_kernel_specs(config: dict, view_names: list[str]) -> dict[str, list]:
    kb = config.get('kernel_bank') or {}
    mode = str(kb.get('mode', 'shared')).lower()
    if mode not in {'shared', 'per_view'}:
        raise ValueError("kernel_bank.mode must be 'shared' or 'per_view'")

    fallback_preset = str(kb.get('preset') or 'historical12')
    fallback_scale = float(kb.get('single_rbf_scale', 1.0))
    fallback_local_scale_k = int(kb.get('local_scale_k', 7))
    shared_entry = kb.get('shared')
    if shared_entry is None:
        shared_entry = {'preset': fallback_preset, 'single_rbf_scale': fallback_scale, 'local_scale_k': fallback_local_scale_k}

    if mode == 'shared':
        specs = _kernel_specs_from_entry(shared_entry, fallback_preset=fallback_preset, fallback_scale=fallback_scale, fallback_local_scale_k=fallback_local_scale_k)
        return {name: list(specs) for name in view_names}

    by_view = kb.get('by_view') or {}
    out = {}
    for name in view_names:
        entry = by_view.get(name, shared_entry)
        out[name] = _kernel_specs_from_entry(entry, fallback_preset=fallback_preset, fallback_scale=fallback_scale, fallback_local_scale_k=fallback_local_scale_k)
    return out


def _kernel_plan(specs_by_view: dict[str, list], support_policy: str) -> dict:
    out = {}
    for view, specs in specs_by_view.items():
        fams = ['cosine'] if support_policy == 'shared_geometry' else sorted({s.inferred_support_family() for s in specs})
        out[view] = {
            'kernels': len(specs),
            'families': fams,
            'specs': [kernel_spec_to_dict(s) for s in specs],
        }
    return out


def _effective_parameter_plan(engine: str, rows: list[dict], specs_by_view: dict[str, list]) -> dict:
    """Return an explicit, non-ambiguous view of the two neighbor parameters.

    ``support_k`` controls how many graph neighbors are retained by the sparse
    engine.  ``local_scale_k`` is the Zelnik-Manor/Perona self-tuned bandwidth
    rank and is independent of ``support_k``.  Historically both were displayed
    as "k" in parts of the UI, which was easy to misread.  This structure is
    persisted with every run so paper tables can state the effective values.
    """
    support_k = {str(r['name']): int(r['k']) for r in rows}
    local_scale = {}
    for view, specs in specs_by_view.items():
        vals = []
        for spec in specs:
            if getattr(spec, 'kind', None) == 'rbf_self_tuned':
                vals.append({'kernel': str(spec.name), 'local_scale_k': int(spec.local_scale_k or 7)})
        if vals:
            local_scale[str(view)] = vals
    active_support = str(engine).lower() == 'scalable_sparse'
    violations = []
    if active_support:
        for view, entries in local_scale.items():
            sk = int(support_k[view])
            for item in entries:
                if int(item['local_scale_k']) > sk:
                    violations.append(
                        f"{view}:{item['kernel']} uses local-scale K={item['local_scale_k']} but support k={sk}; require K_scale <= k_support"
                    )
    return {
        'support_k_active': bool(active_support),
        'support_k_by_view': support_k if active_support else None,
        'support_k_note': 'retained directed neighbors per sample' if active_support else 'inactive for Standard Dense engine',
        'self_tuned_local_scale_by_view': local_scale,
        'local_scale_note': 'sigma_i = distance to the K_scale-th non-self neighbor; independent of support k',
        'constraints_ok': not violations,
        'violations': violations,
    }



def ground_truth_plan(config: dict, expected_n: int | None = None) -> dict:
    """Cheap label-file inspection used by the UI/preflight.

    Ground-truth is never required by the clustering algorithm; this helper only
    prevents benchmark configuration slips (e.g. Letter c=10 although labels
    contain 26 classes).  ``.npy`` labels are memory-mapped and the unique count
    is computed once during preflight.
    """
    gt = config.get('ground_truth')
    if not gt:
        return {'available': False, 'requested_c': int(config.get('clusters', 0) or 0), 'warnings': []}
    try:
        if isinstance(gt, dict):
            y = np.asarray(load_array(str(gt['path']), gt.get('key'), mmap=True)).reshape(-1)
            path = str(gt['path'])
        else:
            y = np.asarray(load_array(str(gt), None, mmap=True)).reshape(-1)
            path = str(gt)
        tc = int(np.unique(y).size)
        req = int(config.get('clusters', 0) or 0)
        warnings = []
        if expected_n is not None and int(y.size) != int(expected_n):
            warnings.append(f'ground_truth length {int(y.size)} != n={int(expected_n)}')
        if req and tc != req:
            warnings.append(f'ground_truth has {tc} classes but requested c={req}')
        return {
            'available': True, 'path': os.path.abspath(path), 'n': int(y.size),
            'true_clusters': tc, 'requested_c': req,
            'matches_requested_c': bool(req == tc), 'warnings': warnings,
        }
    except Exception as e:
        return {'available': False, 'error': f'{type(e).__name__}: {e}', 'requested_c': int(config.get('clusters', 0) or 0), 'warnings': [f'Could not inspect ground truth: {e}']}


def autofill_config(config: dict) -> dict:
    """Resolve a prepared multi-view manifest and benchmark-safe defaults.

    A v0.7.3 config may contain only ``dataset_manifest`` plus algorithmic
    settings. Explicit ``views``/``ground_truth``/``clusters`` fields still win.
    """
    import copy
    cfg = copy.deepcopy(config)

    manifest_path = cfg.get('dataset_manifest')
    if manifest_path:
        mp = Path(str(manifest_path)).expanduser().resolve()
        m = json.loads(mp.read_text(encoding='utf-8'))
        from .prepared_catalog import inspect_manifest
        check = inspect_manifest(mp)
        if not check['ready']:
            raise ValueError(f"dataset {mp.parent.name} not ready ({check['status']}): {check['message']}")
        # Resolve old absolute paths into the copied dataset directory.
        # Do not modify the archived manifest; the config records effective paths.
        m['views'] = check['views']
        m['labels'] = check['labels']
        if not cfg.get('views'):
            rows = []
            kval = int(cfg.get('k', m.get('recommended_k', 10)))
            for i, v in enumerate(m.get('views') or []):
                if isinstance(v, dict):
                    rows.append({
                        'name': str(v.get('name') or f'view{i+1}'),
                        'path': str(v.get('path') or v.get('view')),
                        'k': int(v.get('k', kval)),
                    })
                else:
                    rows.append({'name': f'view{i+1}', 'path': str(v), 'k': kval})
            cfg['views'] = rows
        if not cfg.get('ground_truth') and m.get('labels'):
            cfg['ground_truth'] = str(m['labels'])
        if not cfg.get('task_name'):
            cfg['task_name'] = str(m.get('dataset') or m.get('name') or mp.stem)
        if 'k' not in cfg and m.get('recommended_k') is not None:
            cfg['k'] = int(m['recommended_k'])
        if ('clusters' not in cfg or not int(cfg.get('clusters') or 0)) and int(m.get('classes') or 0) >= 2:
            cfg['clusters'] = int(m['classes'])

    if bool(cfg.get('auto_clusters_from_ground_truth', False)) and cfg.get('ground_truth'):
        g = ground_truth_plan(cfg)
        if g.get('available') and int(g.get('true_clusters', 0)) >= 2:
            cfg['clusters'] = int(g['true_clusters'])
    return cfg


def plan_multiview_config(config: dict) -> dict:
    config = autofill_config(config)
    rows, _ = _view_headers(config)
    if not rows:
        raise ValueError('config.views must contain at least one view')
    nset = {r['shape'][0] for r in rows}
    if len(nset) != 1:
        raise ValueError('All views must have the same number of samples')
    engine = str(config.get('engine', 'scalable_sparse')).lower()
    if engine not in {'scalable_sparse', 'reference_dense'}:
        raise ValueError("engine must be 'scalable_sparse' or 'reference_dense'")
    support_policy = config.get('support_policy', 'ranking_grouped')
    view_names = [r['name'] for r in rows]
    specs_by_view = _resolved_kernel_specs(config, view_names)
    kplan = _kernel_plan(specs_by_view, support_policy)
    effective_parameters = _effective_parameter_plan(engine, rows, specs_by_view)
    kernel_counts = [kplan[v]['kernels'] for v in view_names]
    family_counts = [len(kplan[v]['families']) for v in view_names]
    if engine == 'reference_dense':
        est = estimate_dense_reference_peak_memory(
            [tuple(r['shape']) for r in rows], kernel_counts, int(config['clusters'])
        )
    else:
        est = estimate_peak_memory(
            [tuple(r['shape']) for r in rows],
            [r['k'] for r in rows],
            kernel_counts,
            family_counts,
            int(config['clusters']),
            candidate_factor=int(config.get('candidate_factor', 4)),
            ann_backend=str(config.get('ann_backend', 'auto')),
            hnsw_m=int(config.get('hnsw_m', 32)),
            memory_mode=str(config.get('memory_mode', 'auto')),
        )
    policy = resource_policy_from_config(config)
    try:
        safe = ensure_memory_safe(est, policy)
        allowed = True
    except MemoryError as e:
        safe = {'safe': False, 'reasons': [str(e)], 'estimate': est.as_dict(), 'policy': policy.as_dict(), 'system': system_info()}
        allowed = False
    from .lowmem import resolve_memory_mode, estimate_scratch_bytes, disk_plan
    effective_mode = resolve_memory_mode(str(config.get('memory_mode', 'auto')), rows[0]['shape'][0])
    disk = None
    if engine == 'scalable_sparse' and effective_mode == 'laptop32':
        disk = disk_plan(
            config.get('workspace') or 'runs/workspace',
            estimate_scratch_bytes(
                [tuple(r['shape']) for r in rows], [r['k'] for r in rows],
                kernel_counts, family_counts,
            ),
        )
        if not disk['safe']:
            allowed = False
            safe['safe'] = False
            safe.setdefault('reasons', []).append(disk['message'])
    gt_plan = ground_truth_plan(config, expected_n=rows[0]['shape'][0])
    parameter_warnings = list(effective_parameters.get('violations') or [])
    if parameter_warnings:
        allowed = False
        safe['safe'] = False
        safe.setdefault('reasons', []).extend(parameter_warnings)
    return {
        'engine': engine,
        'views': rows,
        'clusters': int(config['clusters']),
        'ground_truth': gt_plan,
        'warnings': list(gt_plan.get('warnings') or []) + parameter_warnings,
        'effective_parameters': effective_parameters,
        'support_policy': support_policy,
        'kernel_plan': kplan,
        'kernels_per_view': {v: kplan[v]['kernels'] for v in view_names},
        'families_per_view': {v: len(kplan[v]['families']) for v in view_names},
        'memory': safe,
        'memory_mode_resolved': effective_mode,
        'workspace_disk': disk,
        'allowed': allowed and (disk is None or disk['safe']) or (policy.allow_oom_risk and (disk is None or disk['safe'])),
        'recommended_threads': policy.recommended_threads(),
        'system': doctor_report(),
        'protocol_note': (
            'Dense all-pairs kernels + released kcenter/knorm + official SimpleMKKM outer optimizer + dense partial eigensolve.'
            if engine == 'reference_dense' else
            'Support-first sparse operators + implicit center/knorm + same official SimpleMKKM outer optimizer + iterative top-c solver.'
        ),
    }


def _settings(config: dict, policy: ResourcePolicy) -> dict:
    disc = config.get('discretization', {}) or {}
    solver = config.get('solver', {}) or {}
    return dict(
        search=config.get('search', 'ann'),
        ann_backend=config.get('ann_backend', 'auto'),
        support_mode=config.get('support_mode', 'directed'),
        support_policy=config.get('support_policy', 'ranking_grouped'),
        diffusion_power=int(config.get('diffusion_power', 1)),
        block_size=int(config.get('block_size', 2048)),
        ef_search=int(config.get('ef_search', 128)),
        hnsw_m=int(config.get('hnsw_m', 32)),
        candidate_factor=int(config.get('candidate_factor', 4)),
        query_batch_size=int(config.get('query_batch_size', 4096)),
        random_state=int(config.get('seed', 0)),
        workspace=config.get('workspace') or 'runs/workspace',
        resource_policy=policy,
        verbose=bool(config.get('verbose', False)),
        inner_solver=str(solver.get('inner_solver', 'auto')),
        spmm_backend=str(solver.get('spmm_backend', config.get('spmm_backend', 'auto'))),
        memory_mode=str(config.get('memory_mode', 'auto')),
        inner_tol=float(solver.get('inner_tol', 1e-4)),
        inner_max_iter=int(solver.get('inner_max_iter', 200)),
        outer_tol=float(solver.get('outer_tol', 1e-5)),
        outer_max_iter=int(solver.get('outer_max_iter', 100)),
        acceptance_tol=float(solver.get('acceptance_tol', 1e-6)),
        acceptance_max_iter=int(solver.get('acceptance_max_iter', 800)),
        acceptance_rtol=float(solver.get('acceptance_rtol', 1e-8)),
        acceptance_atol=float(solver.get('acceptance_atol', 1e-10)),
        discretization_backend=str(disc.get('backend', 'auto')),
        discretization_block_rows=int(disc.get('block_rows', 262144)),
        discretization_max_iter=int(disc.get('max_iter', 20)),
        discretization_tol=float(disc.get('tol', 1e-4)),
        discretization_init_sample=int(disc.get('init_sample', 200000)),
        discretization_init_restarts=int(disc.get('init_restarts', 5)),
        discretization_init_local_iter=int(disc.get('init_local_iter', 8)),
        faiss_gpu=bool(disc.get('faiss_gpu', False)),
    )


def run_multiview_config(
    config: dict,
    *,
    progress_callback=None,
    control: RunControl | None = None,
    monitor_callback=None,
) -> tuple[Path, dict]:
    config = autofill_config(config)
    t0 = time.perf_counter()
    view_cfgs = config.get('views', [])
    if not view_cfgs:
        raise ValueError('config.views must contain at least one view')
    engine = str(config.get('engine', 'scalable_sparse')).lower()
    if engine not in {'scalable_sparse', 'reference_dense'}:
        raise ValueError("engine must be 'scalable_sparse' or 'reference_dense'")
    clusters = int(config['clusters'])
    common_k = int(config.get('k', 10))
    mmap = bool(config.get('mmap', True))
    policy = resource_policy_from_config(config)
    thread_info = configure_thread_limits(policy)

    plan = plan_multiview_config(config)
    if not plan['allowed']:
        raise MemoryError('; '.join(plan['memory'].get('reasons') or ['memory plan rejected']))

    out = Path(config.get('out', 'runs/result_multiview.npz'))
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(config.get('run_dir') or (out.parent / (out.stem + '.run')))
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = RunJournal.create(run_dir, config, doctor_report(), plan)

    def emit(ev: dict) -> None:
        journal.progress(ev)
        if progress_callback is not None:
            progress_callback(ev)

    views: dict[str, np.ndarray] = {}
    kmap: dict[str, int] = {}
    shapes: dict[str, list[int]] = {}
    inputs: dict[str, str] = {}
    t_load0 = time.perf_counter()
    try:
        for i, vc in enumerate(view_cfgs):
            name = str(vc.get('name') or f'view{i+1}')
            if name in views:
                raise ValueError(f'Duplicate view name: {name}')
            path = str(vc['path'])
            X = load_array(path, vc.get('key'), mmap=mmap)
            if X.ndim != 2:
                raise ValueError(f'{name} must be 2D')
            views[name] = X
            kmap[name] = int(vc.get('k', common_k))
            shapes[name] = list(X.shape)
            inputs[name] = os.path.abspath(path)
        t_load = time.perf_counter() - t_load0

        n0 = int(next(iter(views.values())).shape[0])
        if any(int(X.shape[0]) != n0 for X in views.values()):
            raise ValueError('All views must have the same number of samples')

        # Ground truth is loaded before fitting so accepted-iterate trajectory
        # metrics can be recorded.  Final evaluation still happens after fit.
        y_true = None
        gt = config.get('ground_truth')
        if gt:
            if isinstance(gt, dict):
                y_true = np.asarray(load_array(str(gt['path']), gt.get('key'), mmap=True)).reshape(-1)
            else:
                y_true = np.asarray(load_array(str(gt), None, mmap=True)).reshape(-1)
            if y_true.shape[0] != n0:
                raise ValueError('ground_truth length does not match sample count')

        gt_diag = ground_truth_diagnostic(y_true, clusters)
        for warning in gt_diag.get('warnings') or []:
            emit({'stage': 'diagnostic:ground_truth', 'progress': None, 'message': warning, 'detail': {'warning': warning}})

        labels_path = run_dir / 'labels.npy'
        labels_mm = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int32, shape=(n0,))

        profile = ProfileBook(callback=emit)
        settings = _settings(config, policy)
        specs_by_view = _resolved_kernel_specs(config, list(views))
        settings['labels_out'] = labels_mm

        traj_raw = config.get('trajectory', {}) or {}
        disc_for_traj = config.get('discretization', {}) or {}
        traj_backend = str(traj_raw.get('backend') or ('standard_replicate_lloyd' if engine == 'reference_dense' else (disc_for_traj.get('backend') or 'batched_lloyd')))
        if traj_backend == 'auto':
            traj_backend = 'batched_lloyd'
        traj_cfg = TrajectoryConfig(
            enabled=bool(traj_raw.get('enabled', True)),
            full_threshold=int(traj_raw.get('full_threshold', 100000)),
            probe_size=int(traj_raw.get('probe_size', 100000)),
            replicates=int(traj_raw.get('replicates', 3)),
            max_iter=int(traj_raw.get('max_iter', 30)),
            tol=float(traj_raw.get('tol', 1e-4)),
            block_rows=int(traj_raw.get('block_rows', 262144)),
            seed=int(config.get('seed', 0)),
            backend=traj_backend,
            init_sample=int(disc_for_traj.get('init_sample', 200000)),
            init_restarts=int(disc_for_traj.get('init_restarts', 5)),
            init_local_iter=int(disc_for_traj.get('init_local_iter', 8)),
        )
        iteration_observer = IterationTrajectoryObserver(
            n0, clusters, y_true=y_true, config=traj_cfg
        ) if traj_cfg.enabled else None

        def _monitor(sample: dict) -> None:
            profile.observe(sample)
            journal.resource(sample)
            if monitor_callback is not None:
                monitor_callback(sample)

        mon = RuntimeMonitor(
            callback=_monitor,
            interval=policy.monitor_interval,
            control=control,
            reserve_bytes=int(policy.reserve_gb * 1024**3),
        )
        evaluation = None
        official_reference_eval = None
        with mon:
            t_fit0 = time.perf_counter()
            if engine == 'reference_dense':
                disc_cfg = config.get('discretization', {}) or {}
                ref_cfg = config.get('reference_protocol', {}) or {}
                disc_backend = str(disc_cfg.get('backend') or 'standard_replicate_lloyd')
                if disc_backend == 'auto':
                    disc_backend = 'standard_replicate_lloyd'
                labels, art = fit_dense_reference_multiview(
                    views,
                    n_clusters=clusters,
                    specs_by_view=specs_by_view,
                    random_state=int(config.get('seed', 0)),
                    workspace=config.get('workspace') or str(run_dir / 'workspace'),
                    resource_policy=policy,
                    progress_callback=emit,
                    control=control,
                    profile=profile,
                    verbose=bool(config.get('verbose', False)),
                    dense_block_rows=int(ref_cfg.get('dense_block_rows', 4096)),
                    outer_tol=float((config.get('solver') or {}).get('outer_tol', 1e-5)),
                    outer_max_iter=int((config.get('solver') or {}).get('outer_max_iter', 100)),
                    reference_eig_tol=float(ref_cfg.get('eig_tol', 1e-10)),
                    discretization_backend=disc_backend,
                    discretization_block_rows=int(disc_cfg.get('block_rows', 262144)),
                    discretization_max_iter=int(disc_cfg.get('max_iter', 100)),
                    discretization_tol=float(disc_cfg.get('tol', 1e-4)),
                    discretization_init_restarts=int(disc_cfg.get('init_restarts', 10)),
                    labels_out=labels_mm,
                    iteration_observer=iteration_observer,
                )
            else:
                labels, art = fit_sc_simplemkkm_multiview(
                    views,
                    n_clusters=clusters,
                    k=kmap,
                    specs_by_view=specs_by_view,
                    progress_callback=emit,
                    control=control,
                    profile=profile,
                    iteration_observer=iteration_observer,
                    **settings,
                )
            t_fit = time.perf_counter() - t_fit0
            labels_mm.flush()

            evcfg = config.get('evaluation', {}) or {}
            if bool(evcfg.get('enabled', True)):
                with profile.stage('evaluation', n=int(n0), external=bool(y_true is not None)):
                    evaluation = evaluate_clustering(
                        art.model.embedding_, labels_mm,
                        centers=None if art.model.discretization_result_ is None else art.model.discretization_result_.centers,
                        y_true=y_true,
                        block_rows=int(evcfg.get('block_rows', 262144)),
                        progress=ProgressSink(emit, control),
                    )
                # Optional reproduction of the released myNMIACCV2 stochastic
                # reporting protocol.  Kept off by default because 20 x 10
                # full k-means replicates can dominate runtime even on 20K.
                ref_cfg = config.get('reference_protocol', {}) or {}
                if engine == 'reference_dense' and y_true is not None and bool(ref_cfg.get('official_eval_enabled', False)):
                    with profile.stage('reference_official_evaluation', n=int(n0)):
                        official_reference_eval = official_simplemkkm_external_protocol(
                            art.model.embedding_, y_true, clusters,
                            rounds=int(ref_cfg.get('official_eval_rounds', 20)),
                            replicates=int(ref_cfg.get('official_eval_replicates', 10)),
                            max_iter=int(ref_cfg.get('official_eval_max_iter', 100)),
                            random_state=int(config.get('seed', 0)),
                            block_rows=int(evcfg.get('block_rows', 262144)),
                        )

        gamma = np.asarray(art.model.gamma_, dtype=np.float64)
        trajectory = list(art.model.history_.iteration_metrics)
        # The accepted-state trajectory is the authoritative optimization curve.
        objective = np.asarray([float(r['objective']) for r in trajectory], dtype=np.float64)
        if trajectory:
            g0 = np.asarray(trajectory[0].get('gamma') or gamma, dtype=np.float64)
            gf = np.asarray(trajectory[-1].get('gamma') or gamma, dtype=np.float64)
            odiff = np.diff(objective) if objective.size > 1 else np.asarray([], dtype=np.float64)
            optimization_summary = {
                'accepted_states': int(len(trajectory)),
                'accepted_updates': int(max(0, len(trajectory) - 1)),
                'objective_initial': float(objective[0]),
                'objective_final': float(objective[-1]),
                'objective_change': float(objective[-1] - objective[0]),
                'max_abs_gamma_change': float(np.max(np.abs(gf - g0))) if g0.size and gf.size == g0.size else None,
                'accepted_objective_monotone': bool(np.all(odiff <= 0.0)) if odiff.size else True,
                'max_positive_objective_step': float(max(0.0, float(np.max(odiff)))) if odiff.size else 0.0,
                'acceptance_tol': float(getattr(art.model, 'acceptance_tol', np.nan)),
                'acceptance_rtol': float(getattr(art.model, 'acceptance_rtol', np.nan)),
                'acceptance_atol': float(getattr(art.model, 'acceptance_atol', np.nan)),
            }
        else:
            optimization_summary = {'accepted_states': 0, 'accepted_updates': 0}
        centers = None if art.model.discretization_result_ is None else np.asarray(art.model.discretization_result_.centers, dtype=np.float32)
        np.savez_compressed(out, gamma=gamma, objective=objective, **({} if centers is None else {'centers': centers}))

        profile_json = run_dir / 'profile.json'
        profile_csv = run_dir / 'profile.csv'
        profile.write_json(profile_json); profile.write_csv(profile_csv)
        eval_path = None
        if evaluation is not None:
            eval_path = run_dir / 'evaluation.json'
            eval_payload = dict(evaluation)
            if official_reference_eval is not None:
                eval_payload['official_reference_protocol'] = official_reference_eval
            eval_path.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding='utf-8')

        disc = art.model.discretization_result_
        if engine == 'scalable_sparse':
            graphs_meta = {view: {fam: graph_diagnostics(g) for fam, g in va.graphs.items()} for view, va in art.views.items()}
        else:
            graphs_meta = {}
        graph_diag = graph_diagnostic_summary(graphs_meta, clusters)
        spec_cfg = config.get('spectral_diagnostics', {}) or {}
        spectral_diag = {'status': 'disabled'}
        if bool(spec_cfg.get('enabled', True)):
            with profile.stage('spectral_diagnostic', n=int(n0), c=int(clusters)):
                spectral_diag = spectral_boundary_diagnostic(
                    art.model,
                    max_n=int(spec_cfg.get('max_n', 200000)),
                    tol=float(spec_cfg.get('tol', 1e-5)),
                    maxiter=int(spec_cfg.get('maxiter', 400)),
                )
        solver_diag = {
            'fit_status': art.model.fit_status_,
            'termination_reason': art.model.termination_reason_,
            'final_inner_residual': float(art.model.history_.iteration_metrics[-1].get('inner_residual', np.nan)) if art.model.history_.iteration_metrics else None,
            'inner_tol': float(getattr(art.model, 'inner_tol', np.nan)),
            'acceptance_tol': float(getattr(art.model, 'acceptance_tol', np.nan)),
            'warnings': [],
        }
        if art.model.fit_status_ in {'numerically_ambiguous', 'converged_inner_not_certified'}:
            solver_diag['warnings'].append(f'SimpleMKKM stopped with status {art.model.fit_status_}; inspect residuals and line_search.csv.')
        run_warnings = aggregate_run_warnings(gt_diag, graph_diag, spectral_diag, solver_diag)
        prepared_paths = {view: va.prepared_path for view, va in art.views.items()}
        meta = {
            'mode': 'multi_view', 'engine': engine,
            'engine_label': 'Standard Dense SimpleMKKM' if engine == 'reference_dense' else 'SC-SimpleMKKM',
            'run_id': config.get('run_id'), 'task_name': config.get('task_name'), 'run_dir': str(run_dir), 'config': config,
            'inputs': inputs, 'shapes': shapes, 'clusters': clusters,
            'k_by_view': kmap if engine == 'scalable_sparse' else None,
            'effective_parameters': _effective_parameter_plan(
                engine,
                [{'name': name, 'k': kmap[name]} for name in views],
                specs_by_view,
            ),
            'kernel_bank_resolved': {view: [kernel_spec_to_dict(s) for s in specs] for view, specs in specs_by_view.items()},
            'protocol': {
                'feature_preprocess': 'row L2 normalization',
                'kernel_representation': 'dense all-pairs base kernels' if engine == 'reference_dense' else 'support-first sparse graph-induced PSD operators',
                'kernel_preprocess': 'released kcenter + knorm (explicit dense)' if engine == 'reference_dense' else 'mathematically exact implicit center + diagonal norm',
                'outer_optimizer': 'released SimpleMKKM reduced gradient + projected step + golden-section search',
                'acceptance_rule': 'cheap search + common strict accepted-state recheck; public J trajectory contains strict states only',
                'inner_solver': 'dense ARPACK eigsh largest-algebraic' if engine == 'reference_dense' else str((config.get('solver') or {}).get('inner_solver', 'auto')),
                'discretization': None if disc is None else disc.backend,
            },
            'load_seconds': t_load, 'fit_seconds': t_fit, 'total_seconds': time.perf_counter() - t0,
            'peak_rss_bytes': int(mon.peak_rss_bytes),
            'labels_path': str(labels_path), 'labels_dtype': 'int32', 'labels_count': int(n0),
            'gamma': gamma.tolist(), 'gamma_table': art.gamma_table(), 'view_weight_table': art.view_weight_table(),
            'objective': objective.tolist(),
            'optimization_summary': optimization_summary,
            'iteration_metrics': trajectory,
            'trajectory_config': traj_cfg.__dict__,
            'inner_iterations': list(map(int, art.model.history_.inner_iterations)),
            'inner_residual': list(map(float, art.model.history_.inner_residual)),
            'inner_seconds': list(map(float, art.model.history_.inner_seconds)),
            'line_search_trials': list(map(int, art.model.history_.line_search_trials)),
            'line_search': list(art.model.history_.line_search),
            'acceptance_checks': list(getattr(art.model.history_, 'acceptance_checks', [])),
            'simplemkkm_status': art.model.fit_status_, 'simplemkkm_termination': art.model.termination_reason_,
            'postprocess_backend': art.model.postprocess_backend_,
            'discretization': None if disc is None else {'backend': disc.backend, 'inertia': disc.inertia, 'n_iter': disc.n_iter, 'seconds': disc.seconds},
            'evaluation': evaluation,
            'official_reference_evaluation': official_reference_eval,
            'graphs': graphs_meta,
            'diagnostics': {
                'ground_truth': gt_diag,
                'graph_summary': graph_diag,
                'spectral_boundary': spectral_diag,
                'solver': solver_diag,
                'warnings': run_warnings,
            },
            'warnings': run_warnings,
            'prepared_paths': prepared_paths,
            'settings': {k: (v.as_dict() if isinstance(v, ResourcePolicy) else v) for k, v in settings.items() if k != 'labels_out'},
            'resource_plan': plan,
            'thread_limits': {k: v for k, v in thread_info.items() if not k.startswith('_')},
            'profile': profile.as_dicts(), 'profile_json': str(profile_json), 'profile_csv': str(profile_csv),
            'evaluation_json': None if eval_path is None else str(eval_path),
            'events_jsonl': str(run_dir / 'events.jsonl'), 'resources_jsonl': str(run_dir / 'resources.jsonl'),
            'handoff_json': str(run_dir / 'handoff.json'), 'handoff_md': str(run_dir / 'HANDOFF.md'),
            **doctor_report(),
        }
        meta_path = run_dir / 'result.json'
        meta['result_json'] = str(meta_path); meta['result_npz'] = str(out)
        paper_csv, paper_json = write_paper_summary(run_dir, meta)
        meta['paper_summary_csv'] = str(paper_csv); meta['paper_summary_json'] = str(paper_json)
        (run_dir/'config_resolved.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        (run_dir/'effective_parameters.json').write_text(json.dumps(meta['effective_parameters'], indent=2, ensure_ascii=False), encoding='utf-8')
        tabular = write_tabular_artifacts(run_dir, meta)
        tabular['config_resolved.json']=str(run_dir/'config_resolved.json')
        tabular['effective_parameters.json']=str(run_dir/'effective_parameters.json')
        # Save a bounded embedding sample for teaching/visualization, never full H by default.
        sample_n = int((config.get('artifacts') or {}).get('embedding_sample', 20000))
        if sample_n > 0 and art.model.embedding_ is not None:
            ns = min(int(n0), sample_n); idx = np.linspace(0, int(n0)-1, ns, dtype=np.int64)
            emb = np.asarray(art.model.embedding_[idx], dtype=np.float32)
            sample_payload = {'index': idx, 'embedding': emb, 'labels': np.asarray(labels_mm[idx],dtype=np.int32)}
            if y_true is not None:
                sample_payload['ground_truth'] = np.asarray(y_true[idx])
            np.savez_compressed(run_dir/'embedding_sample.npz', **sample_payload)
            tabular['embedding_sample.npz'] = str(run_dir/'embedding_sample.npz')
        report_html = write_run_report(run_dir, meta)
        meta['run_report_html'] = str(report_html)
        artifact_manifest = write_artifact_manifest(run_dir, meta, tabular)
        meta['artifact_manifest'] = str(artifact_manifest)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        write_run_report(run_dir, meta)
        out.with_suffix(out.suffix + '.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        outputs = {
            'result_npz': str(out), 'labels_npy': str(labels_path), 'result_json': str(meta_path),
            'profile_csv': str(profile_csv), 'profile_json': str(profile_json),
            'evaluation_json': None if eval_path is None else str(eval_path),
            'events_jsonl': str(run_dir / 'events.jsonl'), 'resources_jsonl': str(run_dir / 'resources.jsonl'),
            'run_report_html': str(report_html), 'paper_summary_csv': str(paper_csv), 'paper_summary_json': str(paper_json), 'artifact_manifest': str(artifact_manifest),
        }
        journal.finish(status='done', outputs=outputs, result_summary={'engine': engine, 'total_seconds': meta['total_seconds'], 'peak_rss_bytes': meta['peak_rss_bytes'], 'postprocess_backend': meta['postprocess_backend'], 'evaluation': evaluation})
        return out, meta
    except Exception as exc:
        journal.fail_from_exception(exc)
        raise


def run_mv(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = autofill_config(json.loads(config_path.read_text(encoding='utf-8')))
    if args.out is not None:
        config['out'] = args.out
    control = RunControl()
    last_stage = {'stage': ''}
    def progress(ev: dict) -> None:
        if ev.get('type', '').startswith('profile_'):
            return
        st = ev.get('stage', '')
        msg = ev.get('message')
        if msg and st != last_stage['stage']:
            print(f'[{st}] {msg}', flush=True)
            last_stage['stage'] = st
    out, meta = run_multiview_config(config, progress_callback=progress, control=control)
    print(json.dumps({'result': str(out), 'meta': str(out) + '.json', **meta}, indent=2, ensure_ascii=False))
    return 0


def plan_cmd(args: argparse.Namespace) -> int:
    config = autofill_config(json.loads(Path(args.config).read_text(encoding='utf-8')))
    print(json.dumps(plan_multiview_config(config), indent=2, ensure_ascii=False))
    return 0


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='scsimplemkkm', description='SC-SimpleMKKM runner')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor', help='show environment, CPU/RAM, optional ANN backends')
    plan = sub.add_parser('plan', help='resource preflight for a multi-view config')
    plan.add_argument('--config', required=True)
    mv = sub.add_parser('run-mv', help='run multi-view config JSON')
    mv.add_argument('--config', required=True)
    mv.add_argument('--out', default=None)
    return p


def main() -> int:
    p = make_parser(); args = p.parse_args()
    if args.cmd == 'doctor': return doctor()
    if args.cmd == 'plan': return plan_cmd(args)
    if args.cmd == 'run-mv': return run_mv(args)
    raise AssertionError(args.cmd)


if __name__ == '__main__':
    raise SystemExit(main())
