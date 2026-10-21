from __future__ import annotations

"""Publication figures for SC-SimpleMKKM runs.

Every exported panel intentionally has *no embedded title*.  The manuscript
should provide titles/captions through LaTeX ``subfigure``/``subcaption``.  Each
panel is emitted as PDF, SVG and 300-dpi PNG plus the CSV used to draw it.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r",encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def _f(x, default=np.nan):
    try: return float(x)
    except Exception: return default


def _i(x, default=0):
    try: return int(float(x))
    except Exception: return default


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":"serif", "font.size":9.0, "axes.labelsize":9.5,
        "xtick.labelsize":8.5, "ytick.labelsize":8.5, "legend.fontsize":8.0,
        "axes.linewidth":0.8, "lines.linewidth":1.6, "lines.markersize":5.0,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.bbox":"tight",
    })


def _save(fig, root: Path, stem: str):
    root.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","svg","png"]:
        kw={"dpi":300} if ext=="png" else {}
        fig.savefig(root/f"{stem}.{ext}",bbox_inches="tight",pad_inches=0.03,**kw)


def _dataset_key(r: dict) -> str:
    return r.get("paper_dataset_label") or r.get("paper_dataset") or r.get("task_name") or "dataset"


def _protocol_key(r: dict) -> str:
    return r.get("paper_protocol") or r.get("engine") or "method"


def _write_data(path: Path, rows: Iterable[dict]):
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys or ["empty"]); w.writeheader(); w.writerows(rows)


def quality_panels(groups: list[dict], out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    made=[]
    for metric in ["ACC","NMI","ARI"]:
        rows=[]
        for g in groups:
            m=_f(g.get(f"{metric}_mean")); sd=_f(g.get(f"{metric}_std"),0.0)
            if math.isfinite(m): rows.append({"dataset":_dataset_key(g),"protocol":_protocol_key(g),"mean":m,"std":0.0 if not math.isfinite(sd) else sd,"runs":_i(g.get("runs"))})
        if not rows: continue
        datasets=list(dict.fromkeys(r["dataset"] for r in rows)); protocols=list(dict.fromkeys(r["protocol"] for r in rows))
        h=max(2.4,0.34*len(datasets)+1.1); fig,ax=plt.subplots(figsize=(6.7,h))
        y=np.arange(len(datasets)); width=0.8/max(1,len(protocols))
        for j,p in enumerate(protocols):
            vals=[]; errs=[]
            lookup={(r["dataset"],r["protocol"]):r for r in rows}
            for d in datasets:
                q=lookup.get((d,p)); vals.append(np.nan if q is None else q["mean"]); errs.append(0 if q is None else q["std"])
            off=(j-(len(protocols)-1)/2)*width
            ax.barh(y+off,vals,height=width*0.88,xerr=errs,label=p,capsize=2,alpha=0.92)
        ax.set_yticks(y); ax.set_yticklabels(datasets); ax.invert_yaxis(); ax.set_xlabel(metric)
        ax.set_xlim(0,1.02); ax.grid(axis="x",alpha=0.22,linewidth=0.6); ax.legend(frameon=False,ncol=min(3,len(protocols)),loc="lower right")
        ax.set_title("")
        fig.tight_layout(); stem=f"quality_{metric.lower()}"; _save(fig,out,stem); plt.close(fig); _write_data(out/f"{stem}.csv",rows); made.append(stem)
    return made


def scaling_panels(groups: list[dict], out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    made=[]
    specs=[("fit_seconds_mean","Runtime (s)","runtime_scaling",True),
           ("peak_rss_bytes_mean","Peak RSS (GiB)","memory_scaling",True)]
    for key,ylabel,stem,logy in specs:
        rows=[]
        for g in groups:
            n=_i(g.get("n") or g.get("n_mean")); val=_f(g.get(key))
            if n>0 and math.isfinite(val):
                if "bytes" in key: val/=2**30
                rows.append({"dataset":_dataset_key(g),"protocol":_protocol_key(g),"n":n,"value":val})
        if not rows: continue
        fig,ax=plt.subplots(figsize=(5.7,3.7))
        protocols=list(dict.fromkeys(r["protocol"] for r in rows))
        for p in protocols:
            q=sorted([r for r in rows if r["protocol"]==p],key=lambda x:x["n"])
            ax.plot([r["n"] for r in q],[r["value"] for r in q],marker="o",label=p)
        ax.set_xscale("log");
        if logy: ax.set_yscale("log")
        ax.set_xlabel("Number of samples, n"); ax.set_ylabel(ylabel); ax.grid(True,which="both",alpha=0.22,linewidth=0.6); ax.legend(frameon=False)
        ax.set_title(""); fig.tight_layout(); _save(fig,out,stem); plt.close(fig); _write_data(out/f"{stem}.csv",rows); made.append(stem)
    return made



def storage_gap_panel(groups: list[dict], out: Path) -> list[str]:
    """Measured peak RSS versus the float64 dense-kernel-bank storage equivalent."""
    import matplotlib.pyplot as plt
    rows=[]
    seen=set()
    # Prefer scalable runs because the point of this panel is the representation gap.
    cand=[g for g in groups if str(g.get("engine"))=="scalable_sparse"] or groups
    for g in cand:
        n=_i(g.get("n")); m=max(1,_i(g.get("kernels"),1)); rss=_f(g.get("peak_rss_bytes_mean"))
        key=(_dataset_key(g),n,m)
        if n<=0 or not math.isfinite(rss) or key in seen: continue
        seen.add(key)
        rows.append({"dataset":key[0],"n":n,"kernels":m,"measured_rss_gib":rss/2**30,"dense_bank_gib":8.0*m*n*n/2**30})
    if not rows: return []
    rows=sorted(rows,key=lambda x:x["n"])
    fig,ax=plt.subplots(figsize=(5.7,3.7))
    ax.plot([r["n"] for r in rows],[r["measured_rss_gib"] for r in rows],marker="o",label="Measured peak RSS")
    ax.plot([r["n"] for r in rows],[r["dense_bank_gib"] for r in rows],marker="s",linestyle="--",label="Dense kernel-bank equivalent")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of samples, n"); ax.set_ylabel("Memory (GiB)")
    ax.grid(True,which="both",alpha=0.22,linewidth=0.6); ax.legend(frameon=False)
    ax.set_title(""); fig.tight_layout(); _save(fig,out,"storage_gap"); plt.close(fig); _write_data(out/"storage_gap.csv",rows)
    return ["storage_gap"]

def thread_panel(runs: list[dict], out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    rows=[]
    groups=defaultdict(list)
    for r in runs:
        th=_i(r.get("paper_threads")); t=_f(r.get("fit_seconds"))
        if th>0 and math.isfinite(t): groups[(_dataset_key(r),_protocol_key(r),_i(r.get("seed")))].append((th,t))
    # Pool per dataset/protocol after normalizing each seed by its own smallest-thread runtime.
    pooled=defaultdict(list)
    for (d,p,s),vals in groups.items():
        vals=sorted(vals); base_th,base_t=vals[0]
        for th,t in vals: pooled[(d,p,th)].append(base_t/t)
    for (d,p,th),x in pooled.items(): rows.append({"dataset":d,"protocol":p,"threads":th,"speedup_mean":float(np.mean(x)),"speedup_std":float(np.std(x,ddof=1)) if len(x)>1 else 0.0,"runs":len(x)})
    if not rows: return []
    fig,ax=plt.subplots(figsize=(5.7,3.7))
    pairs=list(dict.fromkeys((r["dataset"],r["protocol"]) for r in rows))
    maxth=max(r["threads"] for r in rows)
    for d,p in pairs:
        q=sorted([r for r in rows if r["dataset"]==d and r["protocol"]==p],key=lambda x:x["threads"])
        ax.errorbar([r["threads"] for r in q],[r["speedup_mean"] for r in q],yerr=[r["speedup_std"] for r in q],marker="o",capsize=2,label=f"{d} / {p}")
    ax.plot([1,maxth],[1,maxth],linestyle="--",linewidth=1.0,label="Ideal")
    ax.set_xlabel("CPU threads"); ax.set_ylabel("Speedup"); ax.grid(True,alpha=0.22,linewidth=0.6); ax.legend(frameon=False,fontsize=7)
    ax.set_title(""); fig.tight_layout(); _save(fig,out,"thread_speedup"); plt.close(fig); _write_data(out/"thread_speedup.csv",rows)
    return ["thread_speedup"]


def stage_breakdown(runs_root: Path, runs: list[dict], out: Path, max_datasets: int=8) -> list[str]:
    import matplotlib.pyplot as plt
    raw=[]
    for r in runs:
        rd=Path(r.get("run_dir") or ""); p=rd/"profile.json"
        if not p.exists(): continue
        try: prof=json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        for s in prof:
            sec=_f(s.get("seconds")); st=str(s.get("stage") or "")
            if st and math.isfinite(sec) and sec>0: raw.append({"dataset":_dataset_key(r),"protocol":_protocol_key(r),"seed":_i(r.get("seed")),"stage":st,"seconds":sec})
    if not raw: return []
    # Mean stage time per dataset/protocol, then retain the top stages globally.
    cell=defaultdict(list)
    for r in raw: cell[(r["dataset"],r["protocol"],r["stage"])].append(r["seconds"])
    means=[{"dataset":d,"protocol":p,"stage":st,"seconds":float(np.mean(x))} for (d,p,st),x in cell.items()]
    stage_tot=defaultdict(float)
    for r in means: stage_tot[r["stage"]]+=r["seconds"]
    top=[x[0] for x in sorted(stage_tot.items(),key=lambda kv:kv[1],reverse=True)[:7]]
    labels=list(dict.fromkeys((r["dataset"],r["protocol"]) for r in means))[:max_datasets]
    fig,ax=plt.subplots(figsize=(6.8,max(2.5,0.38*len(labels)+1.0))); y=np.arange(len(labels)); left=np.zeros(len(labels))
    plotrows=[]
    for st in top:
        vals=[]
        for d,p in labels:
            v=sum(r["seconds"] for r in means if r["dataset"]==d and r["protocol"]==p and r["stage"]==st); vals.append(v); plotrows.append({"dataset":d,"protocol":p,"stage":st,"seconds":v})
        ax.barh(y,vals,left=left,label=st); left+=np.asarray(vals)
    ax.set_yticks(y); ax.set_yticklabels([f"{d} / {p}" for d,p in labels]); ax.invert_yaxis(); ax.set_xlabel("Mean stage time (s)"); ax.legend(frameon=False,fontsize=7,ncol=2)
    ax.set_title(""); fig.tight_layout(); _save(fig,out,"stage_breakdown"); plt.close(fig); _write_data(out/"stage_breakdown.csv",plotrows)
    return ["stage_breakdown"]


def generate(runs_root: str | Path, out_dir: str | Path | None=None) -> dict:
    _style(); root=Path(runs_root).expanduser().resolve(); stats=root/"_paper_stats"
    groups=_read_csv(stats/"paper_group_stats.csv"); runs=_read_csv(stats/"paper_runs.csv")
    out=Path(out_dir).expanduser().resolve() if out_dir else root/"_paper_figures"; out.mkdir(parents=True,exist_ok=True)
    made=[]; made+=quality_panels(groups,out); made+=scaling_panels(groups,out); made+=storage_gap_panel(groups,out); made+=thread_panel(runs,out); made+=stage_breakdown(root,runs,out)
    captions={
        "quality_acc":"Clustering accuracy (mean ± standard deviation over repeated seeds).",
        "quality_nmi":"Normalized mutual information (mean ± standard deviation over repeated seeds).",
        "quality_ari":"Adjusted Rand index (mean ± standard deviation over repeated seeds).",
        "runtime_scaling":"End-to-end fitting time versus sample size.",
        "memory_scaling":"Peak resident memory versus sample size.",
        "storage_gap":"Measured peak RSS versus the float64 dense-kernel-bank storage equivalent.",
        "thread_speedup":"Shared-memory multicore speedup; dashed line denotes ideal linear scaling.",
        "stage_breakdown":"Mean stage-wise runtime composition.",
    }
    (out/"SUBFIGURE_CAPTIONS.md").write_text("\n".join(f"- `{k}`: {captions.get(k,'')}" for k in made)+"\n",encoding="utf-8")
    payload={"runs_root":str(root),"out_dir":str(out),"figures":made,"title_policy":"No embedded titles; manuscript captions are stored separately."}
    (out/"figure_manifest.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    return payload


def main() -> int:
    ap=argparse.ArgumentParser(description="Generate title-free publication figures from PaperLab results")
    ap.add_argument("--runs-root",required=True); ap.add_argument("--out-dir",default=None)
    a=ap.parse_args(); print(json.dumps(generate(a.runs_root,a.out_dir),indent=2,ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
