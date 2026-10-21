from __future__ import annotations

"""Unattended publication experiment orchestration.

The PaperLab runner is deliberately separate from the Web UI.  Each experiment
job launches a fresh Python subprocess so a completed run returns all memory to
the OS before the next dataset/seed starts.  Jobs are deterministic, resumable,
and failure-isolated: one failed dataset does not terminate an overnight suite.
"""

import argparse
import csv
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .cli import plan_multiview_config
from .paperdata import prepare_recipe
from .paperstats import aggregate_runs
from .runtime import system_info
from .prepared_catalog import inspect_manifest, status_for_recipe


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _slug(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(x)).strip("-._")
    return x[:80] or "job"


def _resolve_path(base: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _find_prepared_manifest(name: str, roots: list[Path]) -> Path | None:
    want = name.strip().lower()
    for root in roots:
        if not root.exists():
            continue
        for mp in list(root.glob("*/manifest.json")) + list(root.glob("*/*/manifest.json")):
            try:
                m = _read_json(mp)
                candidates = {str(m.get("id") or "").lower(), str(m.get("dataset") or "").lower(), mp.parent.name.lower()}
                if want in candidates:
                    return mp.resolve()
            except Exception:
                continue
    return None


def resolve_dataset_manifest(ds: dict, plan_path: Path, plan: dict, *, prepare: bool = False, overwrite: bool = False) -> Path | None:
    base = plan_path.parent
    if ds.get("manifest"):
        p = _resolve_path(base, ds["manifest"])
        if p.exists():
            check = inspect_manifest(p)
            if check['ready']: return p
            raise ValueError(f"dataset manifest {p}: {check['status']}: {check['message']}")
        if ds.get("optional"): return None
        raise FileNotFoundError(str(p))

    recipe = ds.get("recipe") or (ds.get("id") if ds.get("source") == "recipe" else None)
    if recipe:
        data_root = _resolve_path(base, plan.get("data_root", "paper_workspace/data"))
        mp = data_root / str(recipe) / "manifest.json"
        if not mp.exists() and prepare:
            prepare_recipe(str(recipe), data_root, overwrite=overwrite)
        check = status_for_recipe(data_root, str(recipe))
        if check['ready']: return mp.resolve()
        if ds.get("optional"): return None
        raise FileNotFoundError(f"dataset {recipe} {check['status']}: {check['message']}; expected {mp}")

    prepared_name = ds.get("prepared_name") or ds.get("id")
    roots = [_resolve_path(base, x) for x in plan.get("prepared_roots", ["dataset_cache", "paper_workspace/data"])]
    p = _find_prepared_manifest(str(prepared_name), roots)
    if p is not None: return p
    if ds.get("optional", True): return None
    raise FileNotFoundError(f"prepared MVC manifest not found for {prepared_name!r} under {roots}")


def _seeds(ds: dict, plan: dict) -> list[int]:
    if ds.get("seeds") is not None:
        return [int(x) for x in ds["seeds"]]
    if ds.get("repeats") is not None:
        n = int(ds["repeats"]); start = int(ds.get("seed_start", plan.get("seed_start", 0)))
        return list(range(start, start + n))
    if plan.get("seeds") is not None:
        return [int(x) for x in plan["seeds"]]
    n = int(plan.get("repeats", 5)); start = int(plan.get("seed_start", 0))
    return list(range(start, start + n))


def _thread_values(ds: dict, protocol: dict) -> list[int | None]:
    vals = ds.get("threads", protocol.get("threads"))
    if vals is None: return [None]
    if isinstance(vals, str) and vals.lower() in {"auto_ladder", "physical_ladder"}:
        p = max(1, int(system_info().get("physical_cores") or 1))
        out=[]; x=1
        while x < p:
            out.append(x); x*=2
        if p not in out: out.append(p)
        return out
    if isinstance(vals, (int, float)): return [int(vals)]
    return [None if x in (None, 0, "auto") else int(x) for x in vals]


def build_jobs(plan_path: str | Path, *, prepare: bool = False, overwrite_data: bool = False) -> tuple[dict, list[dict], list[str]]:
    pp = Path(plan_path).expanduser().resolve(); plan = _read_json(pp)
    base = pp.parent
    protocols = plan.get("protocols") or {}
    defaults = plan.get("config_defaults") or {}
    runs_root = _resolve_path(base, plan.get("runs_root", "paper_workspace/runs"))
    jobs=[]; warnings=[]
    for ds in plan.get("datasets") or []:
        if ds.get("enabled", True) is False: continue
        mp = resolve_dataset_manifest(ds, pp, plan, prepare=prepare, overwrite=overwrite_data)
        if mp is None:
            warnings.append(f"optional dataset skipped: {ds.get('id') or ds.get('prepared_name')}")
            continue
        manifest = _read_json(mp)
        dataset_id = str(ds.get("id") or manifest.get("id") or manifest.get("dataset") or mp.parent.name)
        dataset_label = str(ds.get("label") or manifest.get("dataset") or dataset_id)
        plist = ds.get("protocols") or plan.get("default_protocols") or list(protocols)
        for pname in plist:
            if pname not in protocols: raise KeyError(f"unknown protocol {pname!r}")
            proto = protocols[pname]
            for seed in _seeds(ds, plan):
                for threads in _thread_values(ds, proto):
                    cfg = _deep_merge(defaults, proto.get("config") or proto)
                    cfg = _deep_merge(cfg, ds.get("config") or {})
                    cfg["dataset_manifest"] = str(mp)
                    cfg["seed"] = int(seed)
                    cfg["task_name"] = dataset_label
                    cfg["paper_dataset"] = dataset_id
                    cfg["paper_dataset_label"] = dataset_label
                    cfg["paper_protocol"] = str(pname)
                    cfg["paper_phase"] = str(ds.get("phase") or plan.get("phase") or "main")
                    cfg["paper_role"] = str(manifest.get("benchmark_role") or ds.get("role") or "unknown")
                    if threads is not None:
                        r = cfg.setdefault("resources", {})
                        r["max_threads"] = int(threads); r["max_cpu_fraction"] = 1.0; r["leave_cores"] = 0
                        cfg["paper_threads"] = int(threads)
                    ttag = "auto" if threads is None else str(threads)
                    jid = _slug(f"{dataset_id}__{pname}__seed{seed:03d}__thr{ttag}")
                    run_dir = runs_root / jid
                    cfg["run_id"] = jid; cfg["runs_root"] = str(runs_root)
                    cfg["run_dir"] = str(run_dir); cfg["out"] = str(run_dir / "result.npz")
                    cfg["workspace"] = str(run_dir / "workspace")
                    jobs.append({"job_id":jid,"dataset_id":dataset_id,"dataset_label":dataset_label,"protocol":pname,
                                 "seed":int(seed),"threads":threads,"manifest":str(mp),"run_dir":str(run_dir),"config":cfg})
    return plan, jobs, warnings


def _completed(run_dir: Path) -> bool:
    if not (run_dir / "result.json").exists(): return False
    try:
        r=_read_json(run_dir/"result.json")
        return bool(r.get("labels_count") or r.get("evaluation") or r.get("simplemkkm_status"))
    except Exception:
        return False


def prepare_plan(plan_path: str | Path, *, overwrite: bool=False) -> dict:
    plan,jobs,warnings=build_jobs(plan_path,prepare=True,overwrite_data=overwrite)
    manifests=sorted({j["manifest"] for j in jobs})
    return {"plan":plan.get("name"),"prepared_manifests":manifests,"jobs":len(jobs),"warnings":warnings}


def run_plan(plan_path: str | Path, *, prepare: bool=False, retry_failed: bool=False,
             max_jobs: int | None=None, dry_run: bool=False, cooldown_seconds: float=0.0) -> dict:
    pp=Path(plan_path).expanduser().resolve(); plan,jobs,warnings=build_jobs(pp,prepare=prepare)
    base=pp.parent; runs_root=_resolve_path(base,plan.get("runs_root","paper_workspace/runs")); runs_root.mkdir(parents=True,exist_ok=True)
    state_path=runs_root/"_paperlab_state.json"; state=_read_json(state_path) if state_path.exists() else {"jobs":{}}
    state.setdefault("jobs",{})
    todo=[]
    for j in jobs:
        rd=Path(j["run_dir"]); rec=state["jobs"].get(j["job_id"],{})
        if _completed(rd):
            rec.update({"status":"done","run_dir":str(rd)}); state["jobs"][j["job_id"]]=rec; continue
        if rec.get("status")=="failed" and not retry_failed: continue
        todo.append(j)
    if max_jobs is not None: todo=todo[:max(0,int(max_jobs))]
    _write_json_atomic(state_path,state)
    if dry_run:
        return {"plan":plan.get("name"),"total_jobs":len(jobs),"pending_jobs":len(todo),"warnings":warnings,
                "jobs":[{k:j[k] for k in ["job_id","dataset_id","protocol","seed","threads","manifest"]} for j in todo]}

    for pos,j in enumerate(todo,1):
        rd=Path(j["run_dir"]); rd.mkdir(parents=True,exist_ok=True)
        cfg_path=rd/"paperlab_job_config.json"; _write_json_atomic(cfg_path,j["config"])
        rec={"status":"preflight","started_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"run_dir":str(rd),
             "dataset":j["dataset_id"],"protocol":j["protocol"],"seed":j["seed"],"threads":j["threads"]}
        state["jobs"][j["job_id"]]=rec; _write_json_atomic(state_path,state)
        print(f"[{pos}/{len(todo)}] PRECHECK {j['job_id']}",flush=True)
        try:
            preflight=plan_multiview_config(j["config"]); _write_json_atomic(rd/"paperlab_preflight.json",preflight)
            if not preflight.get("allowed",False):
                rec.update({"status":"skipped_resource","finished_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "reason":" | ".join(preflight.get("warnings") or (preflight.get("memory") or {}).get("reasons") or ["resource preflight rejected"])})
                _write_json_atomic(rd/"PAPERLAB_SKIPPED.json",rec); _write_json_atomic(state_path,state); continue
        except Exception as e:
            rec.update({"status":"failed_preflight","finished_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"error":f"{type(e).__name__}: {e}"})
            _write_json_atomic(rd/"PAPERLAB_FAILED.json",rec); _write_json_atomic(state_path,state); continue

        rec["status"]="running"; _write_json_atomic(state_path,state)
        cmd=[sys.executable,"-m","scsimplemkkm.cli","run-mv","--config",str(cfg_path)]
        env=os.environ.copy(); env["PYTHONUNBUFFERED"]="1"
        t0=time.time()
        with (rd/"paperlab_stdout.log").open("w",encoding="utf-8") as fo, (rd/"paperlab_stderr.log").open("w",encoding="utf-8") as fe:
            p=subprocess.run(cmd,stdout=fo,stderr=fe,env=env,cwd=str(base))
        elapsed=time.time()-t0
        if p.returncode==0 and _completed(rd):
            rec.update({"status":"done","finished_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_seconds":elapsed})
            (rd/"PAPERLAB_DONE.txt").write_text("done\n",encoding="ascii")
        else:
            rec.update({"status":"failed","finished_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"elapsed_seconds":elapsed,"returncode":p.returncode})
            _write_json_atomic(rd/"PAPERLAB_FAILED.json",rec)
        _write_json_atomic(state_path,state)
        try: aggregate_runs(runs_root,runs_root/"_paper_stats")
        except Exception as e: warnings.append(f"aggregation after {j['job_id']} failed: {e}")
        if cooldown_seconds>0 and pos<len(todo): time.sleep(float(cooldown_seconds))

    stats=aggregate_runs(runs_root,runs_root/"_paper_stats")
    return {"plan":plan.get("name"),"total_jobs":len(jobs),"processed_jobs":len(todo),"warnings":warnings,
            "states":state["jobs"],"paper_stats":str(runs_root/"_paper_stats"/"paper_stats.json"),
            "valid_runs":len(stats.get("runs") or [])}


def status_plan(plan_path: str | Path) -> dict:
    pp=Path(plan_path).expanduser().resolve(); plan,jobs,warnings=build_jobs(pp,prepare=False)
    runs_root=_resolve_path(pp.parent,plan.get("runs_root","paper_workspace/runs")); state_path=runs_root/"_paperlab_state.json"
    state=_read_json(state_path) if state_path.exists() else {"jobs":{}}
    counts={}
    rows=[]
    for j in jobs:
        rec=state.get("jobs",{}).get(j["job_id"],{})
        status="done" if _completed(Path(j["run_dir"])) else rec.get("status","pending")
        counts[status]=counts.get(status,0)+1
        rows.append({"job_id":j["job_id"],"dataset":j["dataset_id"],"protocol":j["protocol"],"seed":j["seed"],"threads":j["threads"],"status":status})
    return {"plan":plan.get("name"),"runs_root":str(runs_root),"counts":counts,"warnings":warnings,"jobs":rows}


def write_job_csv(status: dict, path: Path) -> None:
    rows=status.get("jobs") or []; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        keys=["job_id","dataset","protocol","seed","threads","status"]
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> int:
    ap=argparse.ArgumentParser(description="SC-SimpleMKKM unattended publication experiment runner")
    sub=ap.add_subparsers(dest="cmd",required=True)
    for name in ["prepare","status"]:
        p=sub.add_parser(name); p.add_argument("--plan",required=True)
        if name=="prepare": p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("run"); p.add_argument("--plan",required=True); p.add_argument("--prepare",action="store_true"); p.add_argument("--retry-failed",action="store_true"); p.add_argument("--max-jobs",type=int,default=None); p.add_argument("--dry-run",action="store_true"); p.add_argument("--cooldown-seconds",type=float,default=0.0)
    p=sub.add_parser("aggregate"); p.add_argument("--runs-root",required=True); p.add_argument("--out-dir",default=None)
    a=ap.parse_args()
    if a.cmd=="prepare": out=prepare_plan(a.plan,overwrite=a.overwrite)
    elif a.cmd=="run": out=run_plan(a.plan,prepare=a.prepare,retry_failed=a.retry_failed,max_jobs=a.max_jobs,dry_run=a.dry_run,cooldown_seconds=a.cooldown_seconds)
    elif a.cmd=="aggregate": out=aggregate_runs(a.runs_root,a.out_dir)
    else:
        out=status_plan(a.plan); write_job_csv(out,Path(a.plan).expanduser().resolve().parent/"paperlab_status.csv")
    print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
