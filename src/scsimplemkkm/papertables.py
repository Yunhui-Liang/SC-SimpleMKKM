from __future__ import annotations

"""Booktabs-ready paper tables for SC-SimpleMKKM PaperLab.

The module intentionally emits table bodies without ``\\caption``/``\\label`` so
manuscript source remains the single owner of captions.  Every table is also
written as CSV for traceability.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from .paperdata import list_recipes


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _i(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _esc(x: object) -> str:
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _mean_std(row: dict, metric: str, digits: int = 4) -> str:
    m = _f(row.get(f"{metric}_mean")); s = _f(row.get(f"{metric}_std"))
    if not math.isfinite(m):
        return "--"
    if not math.isfinite(s) or _i(row.get("runs")) <= 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def _seconds(x) -> str:
    v = _f(x)
    if not math.isfinite(v): return "--"
    if v < 1: return f"{v:.3f}"
    if v < 100: return f"{v:.2f}"
    return f"{v:.1f}"


def _gib(x) -> str:
    v = _f(x)
    if not math.isfinite(v): return "--"
    return f"{v / 2**30:.2f}"


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=keys or ["empty"]); w.writeheader(); w.writerows(rows)


def _write_tex(path: Path, columns: str, headers: list[str], rows: list[list[str]], *, notes: list[str] | None = None) -> None:
    lines=[r"\begin{tabular}{" + columns + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines += [""] + ["% " + x for x in notes]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def dataset_table(out: Path) -> str:
    src=[]; tex=[]
    for r in list_recipes():
        rid=str(r.get("id")); role=str(r.get("role") or "")
        row={"dataset":r.get("title") or rid,"id":rid,"role":role,"n":r.get("n"),"views":r.get("views"),"classes":r.get("classes"),"doi":r.get("doi") or ""}
        src.append(row)
        tex.append([_esc(row["dataset"]), _esc(role), f"{_i(row['n']):,}", str(_i(row["views"])), str(_i(row["classes"])), _esc(row["doi"] or "--")])
    _write_csv(out/"dataset_inventory.csv",src)
    _write_tex(out/"dataset_inventory.tex","llrrrr",["Dataset","Role","$n$","Views","Classes","DOI"],tex,
               notes=["standard_mvc, derived_multiview and synthetic_scale must be distinguished in manuscript text."])
    return "dataset_inventory"


def quality_table(groups: list[dict], out: Path) -> str | None:
    rows=[g for g in groups if math.isfinite(_f(g.get("ACC_mean")))]
    if not rows: return None
    rows=sorted(rows,key=lambda g:(str(g.get("paper_dataset_label") or g.get("paper_dataset")),str(g.get("paper_protocol"))))
    tex=[]
    for g in rows:
        tex.append([_esc(g.get("paper_dataset_label") or g.get("paper_dataset")), _esc(g.get("paper_protocol") or g.get("engine")), str(_i(g.get("runs"))),
                    _mean_std(g,"ACC"),_mean_std(g,"NMI"),_mean_std(g,"ARI")])
    _write_csv(out/"quality_table.csv",rows)
    _write_tex(out/"quality_table.tex","llrrrr",["Dataset","Protocol","Runs","ACC","NMI","ARI"],tex,
               notes=["Values are mean $\\pm$ standard deviation when repeated seeds are available."])
    return "quality_table"


def scale_table(groups: list[dict], out: Path) -> str | None:
    rows=[g for g in groups if _i(g.get("n"))>0 and (math.isfinite(_f(g.get("fit_seconds_mean"))) or math.isfinite(_f(g.get("peak_rss_bytes_mean"))))]
    if not rows: return None
    rows=sorted(rows,key=lambda g:(_i(g.get("n")),str(g.get("paper_protocol"))))
    tex=[]
    for g in rows:
        n=_i(g.get("n")); m=max(1,_i(g.get("kernels"),1)); dense_one=8*n*n/2**40; dense_bank=dense_one*m
        tex.append([_esc(g.get("paper_dataset_label") or g.get("paper_dataset")),f"{n:,}",str(_i(g.get("views"))),str(_i(g.get("kernels"))),
                    _seconds(g.get("fit_seconds_mean")),_gib(g.get("peak_rss_bytes_mean")),f"{dense_one:.2f}",f"{dense_bank:.2f}"])
    _write_csv(out/"scale_table.csv",rows)
    _write_tex(out/"scale_table.tex","lrrrrrrr",["Dataset","$n$","Views","Kernels","Time (s)","RSS (GiB)","1 dense $K$ (TiB)","Dense bank (TiB)"],tex,
               notes=["Dense equivalents use float64 and are mathematical storage equivalents, not measured allocations."])
    return "scale_table"


def thread_table(groups: list[dict], out: Path) -> str | None:
    cand=[g for g in groups if _i(g.get("paper_threads"))>0 and math.isfinite(_f(g.get("fit_seconds_mean")))]
    if not cand: return None
    by={}
    for g in cand:
        key=(str(g.get("paper_dataset_label") or g.get("paper_dataset")),str(g.get("paper_protocol")))
        by.setdefault(key,[]).append(g)
    src=[]; tex=[]
    for (d,p),gs in sorted(by.items()):
        gs=sorted(gs,key=lambda x:_i(x.get("paper_threads"))); base=gs[0]; t0=_f(base.get("fit_seconds_mean")); p0=_i(base.get("paper_threads"))
        for g in gs:
            th=_i(g.get("paper_threads")); t=_f(g.get("fit_seconds_mean")); speed=t0/t if t>0 else float("nan"); eff=(speed/(th/p0)) if th>0 else float("nan")
            r={"dataset":d,"protocol":p,"threads":th,"time_seconds":t,"speedup":speed,"parallel_efficiency":eff}; src.append(r)
            tex.append([_esc(d),_esc(p),str(th),_seconds(t),f"{speed:.2f}",f"{100*eff:.1f}\\%"])
    _write_csv(out/"thread_scaling_table.csv",src)
    _write_tex(out/"thread_scaling_table.tex","llrrrr",["Dataset","Protocol","Threads","Time (s)","Speedup","Efficiency"],tex)
    return "thread_scaling_table"


def generate(runs_root: str | Path, out_dir: str | Path | None = None) -> dict:
    root=Path(runs_root).expanduser().resolve(); stats=root/"_paper_stats"
    groups=_read_csv(stats/"paper_group_stats.csv")
    out=Path(out_dir).expanduser().resolve() if out_dir else root/"_paper_tables"; out.mkdir(parents=True,exist_ok=True)
    made=[dataset_table(out)]
    for fn in [quality_table,scale_table,thread_table]:
        x=fn(groups,out)
        if x: made.append(x)
    notes={"caption_policy":"No LaTeX captions or labels are embedded; manuscript owns captions/labels.","tables":made}
    (out/"table_manifest.json").write_text(json.dumps(notes,indent=2,ensure_ascii=False),encoding="utf-8")
    return {"runs_root":str(root),"out_dir":str(out),**notes}


def main() -> int:
    ap=argparse.ArgumentParser(description="Generate booktabs-ready publication tables from PaperLab results")
    ap.add_argument("--runs-root",required=True); ap.add_argument("--out-dir",default=None)
    a=ap.parse_args(); print(json.dumps(generate(a.runs_root,a.out_dir),indent=2,ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
