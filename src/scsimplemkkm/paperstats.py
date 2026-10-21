from __future__ import annotations

"""Publication-oriented aggregation of immutable SC-SimpleMKKM run folders.

The per-run artifacts remain the source of truth.  This module only reads them
and produces flat CSV/JSON summaries.  It intentionally distinguishes final
``evaluation.json`` metrics from accepted-iterate diagnostic trajectories.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _finite(x) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _stats(vals: Iterable[float | None]) -> dict[str, float | int | None]:
    x = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not x:
        return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None, 'ci95_halfwidth': None}
    n = len(x)
    sd = stdev(x) if n >= 2 else 0.0
    # Normal approximation is deliberately labelled as an interval summary, not
    # an inferential guarantee.  For small n, the JSON warnings request repeats.
    ci = 1.96 * sd / math.sqrt(n) if n >= 2 else None
    return {'n': n, 'mean': mean(x), 'std': sd, 'min': min(x), 'max': max(x), 'ci95_halfwidth': ci}


def _flatten_eval(ev: dict) -> dict:
    ev = ev or {}
    ext = ev.get('external') or {}
    inte = ev.get('internal') or {}
    keys = ['ACC', 'NMI', 'ARI', 'Purity', 'MacroF1_mapped']
    out = {k: _finite(ext.get(k)) for k in keys}
    for k in ['inertia_sse', 'CalinskiHarabasz', 'DaviesBouldin', 'cluster_size_entropy_norm']:
        out[k] = _finite(inte.get(k))
    out['pred_clusters'] = ext.get('pred_clusters', inte.get('clusters_active'))
    out['true_clusters'] = ext.get('true_clusters')
    return out


def _support_and_scale(config: dict, result: dict) -> tuple[str, str]:
    views = config.get('views') or []
    sk = []
    if str(config.get('engine') or result.get('engine')) == 'scalable_sparse':
        for v in views:
            try:
                sk.append(int(v.get('k', config.get('k', 10))))
            except Exception:
                pass
    specs = result.get('kernel_bank_resolved') or {}
    scales = []
    if isinstance(specs, dict):
        for view, rows in specs.items():
            for s in rows or []:
                if s.get('kind') == 'rbf_self_tuned' and s.get('local_scale_k') is not None:
                    scales.append(int(s['local_scale_k']))
    if not scales:
        kb = config.get('kernel_bank') or {}
        candidates = kb.get('shared') or []
        if isinstance(candidates, dict):
            candidates = [candidates]
        for s in candidates:
            if isinstance(s, dict) and s.get('kind') == 'rbf_self_tuned':
                scales.append(int(s.get('local_scale_k', 7)))
    return '/'.join(map(str, sorted(set(sk)))) if sk else 'inactive', '/'.join(map(str, sorted(set(scales)))) if scales else 'unused'


def _validate_run(run_dir: Path, result: dict, config: dict, evaluation: dict) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    opt = result.get('optimization_summary') or {}
    checks = result.get('acceptance_checks') or []
    gamma = result.get('gamma') or []
    ev = _flatten_eval(evaluation)

    mono = bool(opt.get('accepted_objective_monotone', True))
    if not mono:
        warnings.append('accepted objective is not monotone')
    certified = all(bool(x.get('certified', False)) for x in checks if x.get('accepted')) if checks else True
    if not certified:
        warnings.append('one or more accepted states are not certified')
    acceptance_tol = _finite(opt.get('acceptance_tol'))
    accepted_res = [_finite(x.get('inner_residual')) for x in checks if x.get('accepted')]
    accepted_res = [x for x in accepted_res if x is not None]
    max_res = max(accepted_res) if accepted_res else _finite(max(result.get('inner_residual') or [float('nan')]))
    residual_ok = True if acceptance_tol is None or max_res is None else max_res <= acceptance_tol * (1 + 1e-8)
    if not residual_ok:
        warnings.append(f'accepted residual {max_res:g} exceeds acceptance_tol {acceptance_tol:g}')

    gamma_sum = float(np.sum(np.asarray(gamma, dtype=float))) if gamma else None
    gamma_nonnegative = bool(np.all(np.asarray(gamma, dtype=float) >= -1e-12)) if gamma else True
    gamma_ok = gamma_nonnegative and (gamma_sum is None or abs(gamma_sum - 1.0) <= 1e-7)
    if not gamma_ok:
        warnings.append(f'invalid gamma simplex: sum={gamma_sum}')

    labels_count = result.get('labels_count')
    n = result.get('labels_count')
    if n is None:
        shapes = result.get('shapes') or {}
        try:
            n = next(iter(shapes.values()))[0]
        except Exception:
            n = None
    labels_ok = labels_count is None or n is None or int(labels_count) == int(n)
    if not labels_ok:
        warnings.append('labels_count does not match n')

    requested_c = int(result.get('clusters') or config.get('clusters') or 0)
    pred_c = ev.get('pred_clusters')
    cluster_count_ok = pred_c is None or requested_c <= 0 or int(pred_c) == requested_c
    if not cluster_count_ok:
        warnings.append(f'predicted clusters={pred_c}, requested={requested_c}')

    metric_ok = all(v is None or (math.isfinite(v) and (-1.000001 <= v <= 1.000001 if k in {'ACC','NMI','ARI','Purity','MacroF1_mapped'} else True))
                    for k, v in ev.items() if k not in {'pred_clusters','true_clusters'})
    if not metric_ok:
        warnings.append('one or more final evaluation metrics are non-finite/out of range')

    return {
        'objective_monotone': mono,
        'accepted_states_certified': certified,
        'max_accepted_residual': max_res,
        'acceptance_tol': acceptance_tol,
        'residual_ok': residual_ok,
        'gamma_sum': gamma_sum,
        'gamma_simplex_ok': gamma_ok,
        'labels_ok': labels_ok,
        'cluster_count_ok': cluster_count_ok,
        'metrics_ok': metric_ok,
        'valid_for_aggregation': not warnings,
    }, warnings


def scan_runs(run_root: str | Path) -> tuple[list[dict], list[str]]:
    root = Path(run_root)
    rows: list[dict] = []
    global_warnings: list[str] = []
    if not root.exists():
        return [], [f'run root does not exist: {root}']
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        result = _read_json(d / 'result.json')
        config = _read_json(d / 'config_resolved.json') or {}
        if not isinstance(result, dict):
            continue
        evaluation = _read_json(d / 'evaluation.json') or result.get('evaluation') or {}
        summary = _read_json(d / 'paper_summary.json') or {}
        valid, warnings = _validate_run(d, result, config, evaluation)
        ev = _flatten_eval(evaluation)
        shapes = result.get('shapes') or {}
        try:
            n = int(result.get('labels_count') or next(iter(shapes.values()))[0])
        except Exception:
            n = int(summary.get('n') or 0)
        support_k, scale_k = _support_and_scale(config, result)
        profile = _read_json(d / 'profile.json', []) or []
        stage_seconds = {str(x.get('stage')): _finite(x.get('seconds')) for x in profile if isinstance(x, dict)}
        gamma = result.get('gamma') or []
        seed = int(config.get('seed', result.get('settings', {}).get('random_state', 0)) or 0)
        row = {
            'run_id': str(result.get('run_id') or d.name),
            'task_name': str(result.get('task_name') or config.get('task_name') or d.name),
            'paper_dataset': str(config.get('paper_dataset') or result.get('paper_dataset') or ''),
            'paper_dataset_label': str(config.get('paper_dataset_label') or result.get('paper_dataset_label') or config.get('task_name') or result.get('task_name') or d.name),
            'paper_protocol': str(config.get('paper_protocol') or result.get('paper_protocol') or ''),
            'paper_phase': str(config.get('paper_phase') or result.get('paper_phase') or ''),
            'paper_role': str(config.get('paper_role') or result.get('paper_role') or ''),
            'paper_threads': int(config.get('paper_threads') or ((config.get('resources') or {}).get('max_threads') or 0) or 0),
            'engine': str(result.get('engine') or config.get('engine') or ''),
            'seed': seed,
            'software_version': result.get('software_version'),
            'n': n,
            'views': int(summary.get('views') or len(shapes) or len(config.get('views') or [])),
            'kernels': int(summary.get('kernels') or len(gamma)),
            'clusters': int(result.get('clusters') or config.get('clusters') or 0),
            'support_k': support_k,
            'K_scale': scale_k,
            'search': config.get('search'),
            'ann_backend': config.get('ann_backend'),
            'memory_mode': config.get('memory_mode'),
            'fit_seconds': _finite(result.get('fit_seconds', summary.get('fit_seconds'))),
            'total_seconds': _finite(result.get('total_seconds', summary.get('total_seconds'))),
            'peak_rss_bytes': result.get('peak_rss_bytes', summary.get('peak_rss_bytes')),
            'accepted_updates': (result.get('optimization_summary') or {}).get('accepted_updates', summary.get('accepted_updates')),
            'objective_initial': _finite((result.get('optimization_summary') or {}).get('objective_initial', summary.get('objective_initial'))),
            'objective_final': _finite((result.get('optimization_summary') or {}).get('objective_final', summary.get('objective_final'))),
            'simplemkkm_status': result.get('simplemkkm_status', summary.get('simplemkkm_status')),
            'simplemkkm_termination': result.get('simplemkkm_termination', summary.get('simplemkkm_termination')),
            **ev,
            **valid,
            'validation_warnings': ' | '.join(warnings),
            'run_dir': str(d),
            'profile_stages': len(stage_seconds),
        }
        rows.append(row)
        if warnings:
            global_warnings.append(f"{row['run_id']}: " + '; '.join(warnings))
    return rows, global_warnings


def _group_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for r in rows:
        ds = r.get('paper_dataset') or r.get('task_name')
        proto = r.get('paper_protocol') or r.get('engine')
        th = int(r.get('paper_threads') or 0)
        groups.setdefault((str(ds), str(proto), str(r['engine']), th), []).append(r)
    out=[]
    metrics = ['ACC','NMI','ARI','Purity','MacroF1_mapped','fit_seconds','total_seconds','peak_rss_bytes','accepted_updates']
    for (dataset, protocol, engine, threads), rs in sorted(groups.items()):
        task = rs[0].get('task_name') or dataset
        z={'task_name':task,'paper_dataset':dataset,'paper_dataset_label':rs[0].get('paper_dataset_label') or task,
           'paper_protocol':protocol,'paper_phase':rs[0].get('paper_phase') or '', 'paper_role':rs[0].get('paper_role') or '',
           'paper_threads':threads,'engine':engine,'runs':len(rs),'seeds':','.join(map(str,sorted({int(r['seed']) for r in rs}))),
           'n':max(int(r.get('n') or 0) for r in rs), 'views':max(int(r.get('views') or 0) for r in rs),
           'kernels':max(int(r.get('kernels') or 0) for r in rs), 'clusters':max(int(r.get('clusters') or 0) for r in rs),
           'valid_runs':sum(bool(r['valid_for_aggregation']) for r in rs)}
        for m in metrics:
            s=_stats([_finite(r.get(m)) for r in rs])
            z[f'{m}_mean']=s['mean']; z[f'{m}_std']=s['std']; z[f'{m}_ci95_halfwidth']=s['ci95_halfwidth']
        z['replication_warning'] = '' if len(rs) >= 3 else f'only {len(rs)} run(s); add repeated seeds for statistical inference'
        out.append(z)
    return out


def _paired_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    # Pair only when a dataset/seed has exactly one dense and one sparse run.
    # This avoids silently pairing an arbitrary ablation protocol with the dense
    # reference when a journal plan contains multiple sparse variants.
    bykey: dict[tuple[str,int], dict[str,list[dict]]] = {}
    for r in rows:
        ds = str(r.get('paper_dataset') or r.get('task_name'))
        bykey.setdefault((ds, int(r['seed'])), {}).setdefault(r['engine'], []).append(r)
    pairs=[]
    for (dataset,seed), engines in sorted(bykey.items()):
        dl=engines.get('reference_dense') or []; sl=engines.get('scalable_sparse') or []
        if len(dl) != 1 or len(sl) != 1: continue
        d=dl[0]; s=sl[0]; task=d.get('task_name') or dataset
        z={'task_name':task,'paper_dataset':dataset,'paper_dataset_label':d.get('paper_dataset_label') or task,'seed':seed,'dense_run_id':d['run_id'],'sparse_run_id':s['run_id']}
        for m in ['ACC','NMI','ARI','Purity','MacroF1_mapped']:
            a=_finite(d.get(m)); b=_finite(s.get(m)); z[f'delta_{m}_sparse_minus_dense']=None if a is None or b is None else b-a
        dt=_finite(d.get('fit_seconds')); st=_finite(s.get('fit_seconds'))
        dm=_finite(d.get('peak_rss_bytes')); sm=_finite(s.get('peak_rss_bytes'))
        z['fit_speedup_dense_over_sparse']=None if not dt or not st else dt/st
        z['peak_memory_ratio_dense_over_sparse']=None if not dm or not sm else dm/sm
        pairs.append(z)
    summaries=[]
    bytask: dict[str,list[dict]]={}
    for p in pairs: bytask.setdefault(p['task_name'],[]).append(p)
    for task,ps in sorted(bytask.items()):
        q={'task_name':task,'paired_runs':len(ps)}
        for m in ['ACC','NMI','ARI','fit_speedup_dense_over_sparse','peak_memory_ratio_dense_over_sparse']:
            key=f'delta_{m}_sparse_minus_dense' if m in {'ACC','NMI','ARI'} else m
            s=_stats([_finite(p.get(key)) for p in ps]); q[f'{key}_mean']=s['mean']; q[f'{key}_std']=s['std']
        # Inferential tests are emitted only when there are enough independent
        # paired seeds to avoid giving a false sense of statistical evidence.
        q['paired_test_warning']=''
        if len(ps) >= 5:
            try:
                from scipy.stats import ttest_1samp, wilcoxon
                for m in ['ACC','NMI','ARI']:
                    x=np.asarray([p[f'delta_{m}_sparse_minus_dense'] for p in ps],dtype=float)
                    q[f'{m}_paired_t_p']=float(ttest_1samp(x,0.0).pvalue)
                    try: q[f'{m}_wilcoxon_p']=float(wilcoxon(x).pvalue)
                    except ValueError: q[f'{m}_wilcoxon_p']=None
            except Exception as e:
                q['paired_test_warning']=f'statistical test unavailable: {e}'
        else:
            q['paired_test_warning']=f'only {len(ps)} paired seed(s); paired significance tests require >=5 here'
        summaries.append(q)
    return pairs,summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys or ['empty']); w.writeheader()
        if rows: w.writerows(rows)


def aggregate_runs(run_root: str | Path='runs', out_dir: str | Path | None=None) -> dict[str, Any]:
    root=Path(run_root)
    out=Path(out_dir) if out_dir is not None else root/'_paper_stats'
    out.mkdir(parents=True,exist_ok=True)
    runs,warnings=scan_runs(root)
    groups=_group_rows(runs)
    pairs,pair_groups=_paired_rows(runs)
    # Add a publication-level warning independent of numerical validity.
    for g in groups:
        if g['runs'] < 3:
            warnings.append(f"{g['task_name']} / {g['engine']}: only {g['runs']} run(s); current artifacts support descriptive comparison, not stable uncertainty estimates")
    payload={
        'run_root':str(root),'out_dir':str(out),'runs':runs,'groups':groups,'paired_runs':pairs,'paired_groups':pair_groups,
        'warnings':warnings,
        'notes':[
            'Final paper metrics are read from evaluation.json/result.evaluation, not the last accepted-iterate diagnostic metric.',
            'accepted-iterate ACC/NMI/ARI are convergence diagnostics with a fixed diagnostic discretization and may differ from final evaluation.',
            'Inferential paired tests are withheld until at least five paired seeds exist for a task.'
        ],
    }
    _write_csv(out/'paper_runs.csv',runs)
    _write_csv(out/'paper_group_stats.csv',groups)
    _write_csv(out/'paper_paired_runs.csv',pairs)
    _write_csv(out/'paper_paired_stats.csv',pair_groups)
    (out/'paper_stats.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload
