from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_paper_summary(run_dir: str | Path, meta: dict) -> tuple[Path, Path]:
    root = Path(run_dir)
    ev = meta.get('evaluation') or {}
    ext = ev.get('external') or {}
    inte = ev.get('internal') or {}
    opt = meta.get('optimization_summary') or {}
    row = {
        'run_dir': str(root),
        'engine': meta.get('engine'),
        'software_version': meta.get('software_version'),
        'n': meta.get('labels_count'),
        'views': len(meta.get('shapes') or {}),
        'kernels': len(meta.get('gamma') or []),
        'clusters': meta.get('clusters'),
        'fit_seconds': meta.get('fit_seconds'),
        'total_seconds': meta.get('total_seconds'),
        'peak_rss_bytes': meta.get('peak_rss_bytes'),
        'ACC': ext.get('ACC'),
        'NMI': ext.get('NMI'),
        'ARI': ext.get('ARI'),
        'Purity': ext.get('Purity'),
        'MacroF1_mapped': ext.get('MacroF1_mapped'),
        'inertia_sse': inte.get('inertia_sse'),
        'CalinskiHarabasz': inte.get('CalinskiHarabasz'),
        'DaviesBouldin': inte.get('DaviesBouldin'),
        'cluster_size_entropy_norm': inte.get('cluster_size_entropy_norm'),
        'postprocess_backend': meta.get('postprocess_backend'),
        'accepted_updates': opt.get('accepted_updates'),
        'objective_initial': opt.get('objective_initial'),
        'objective_final': opt.get('objective_final'),
        'objective_change': opt.get('objective_change'),
        'max_abs_gamma_change': opt.get('max_abs_gamma_change'),
        'simplemkkm_status': meta.get('simplemkkm_status'),
        'simplemkkm_termination': meta.get('simplemkkm_termination'),
    }
    csv_path = root / 'paper_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader(); w.writerow(row)
    json_path = root / 'paper_summary.json'
    json_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf-8')
    return csv_path, json_path


def _js(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace('</', '<\\/')


def write_run_report(run_dir: str | Path, meta: dict) -> Path:
    """Write a self-contained post-run HTML report for paper/teaching/handoff.

    No network/CDN dependency is used. The report intentionally embeds sampled
    resource data and block events, but not the full labels or embedding.
    """
    root = Path(run_dir)
    resources = _read_jsonl(root / 'resources.jsonl')
    events = _read_jsonl(root / 'events.jsonl')
    # Keep the HTML bounded for very long runs: preserve ends and uniformly thin middle.
    def thin(rows: list[dict], nmax: int = 2500) -> list[dict]:
        if len(rows) <= nmax:
            return rows
        step = max(1, len(rows) // nmax)
        out = rows[::step]
        if out[-1] is not rows[-1]: out.append(rows[-1])
        return out[: nmax + 1]
    resources = thin(resources, 2400)
    blocks = [e for e in events if (e.get('detail') or {}).get('block_kind')]
    blocks = thin(blocks, 1600)

    payload = {
        'meta': meta,
        'resources': resources,
        'blocks': blocks,
    }
    title = 'SC-SimpleMKKM Run Report'
    doc = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#687386;--line:#dfe5ee;--accent:#2457d6;}}
*{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,"Microsoft YaHei","Segoe UI",sans-serif;background:var(--bg);color:var(--text)}}
header{{padding:22px 28px;background:#111827;color:white}} header h1{{margin:0 0 4px;font-size:24px}} header p{{margin:0;color:#cbd5e1}}
main{{max-width:1450px;margin:auto;padding:18px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}} .panel{{margin:12px 0}} .card span{{display:block;color:var(--muted);font-size:12px}} .card strong{{font-size:20px;word-break:break-all}}
h2{{font-size:18px;margin:0 0 12px}} h3{{font-size:15px;margin:16px 0 8px}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left}} th{{background:#f8fafc;position:sticky;top:0}} .scroll{{overflow:auto;max-height:430px}}
canvas{{width:100%;height:240px;border:1px solid var(--line);border-radius:8px;background:white}} code{{font-family:ui-monospace,Consolas,monospace;font-size:12px}} .muted{{color:var(--muted)}}
.flow{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}} .flow b{{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:#f8fafc}}
@media(max-width:700px){{main{{padding:8px}} header{{padding:16px}}}}
</style></head><body>
<header><h1>{title}</h1><p>静态、可归档、可用于论文过程展示与后续诊断；不依赖网络。</p></header>
<main>
<div id="cards" class="grid"></div>
<section class="panel"><h2>诊断 / 警告</h2><div id="warnings"></div><div id="diag"></div></section>
<section class="panel"><h2>协议</h2><div id="protocol"></div></section><section class="panel"><h2>执行链</h2><div id="flowline" class="flow"></div></section>
<section class="panel"><h2>优化轨迹</h2><p class="muted">SimpleMKKM objective：越低越好；这里只画统一 strict acceptance 标准下真正接受的 outer states。单点也显示 marker。</p><p id="optNote" class="muted"></p><canvas id="obj" width="1200" height="260"></canvas><h3>Line-search candidate objective</h3><p class="muted">这是 cheap search 的候选值，本身不要求单调，也不能直接进入正式 objective 曲线；真正接受还要经过 strict recheck。</p><canvas id="ls" width="1200" height="240"></canvas><div id="lsTable" class="scroll"></div><h3>ACC / NMI / ARI trajectory</h3><p class="muted" id="trajNote"></p><canvas id="qual" width="1200" height="260"></canvas><h3>Kernel-weight trajectory</h3><canvas id="gam" width="1200" height="300"></canvas></section>
<section class="panel"><h2>CPU / RAM 时间线</h2><canvas id="res" width="1200" height="300"></canvas><p class="muted">左轴归一化为百分比：进程 CPU capacity、系统 CPU、进程 RSS/总RAM。</p></section>
<section class="panel"><h2>阶段 Profile</h2><div id="profile" class="scroll"></div></section>
<section class="panel"><h2>评估</h2><div id="eval"></div></section>
<section class="panel"><h2>Kernel / View 权重</h2><div id="weights" class="scroll"></div></section>
<section class="panel"><h2>最近计算 Block</h2><div id="blocks" class="scroll"></div></section>
<section class="panel"><h2>机器与可复现信息</h2><div id="env" class="scroll"></div></section>
</main>
<script>
const P={_js(payload)}; const M=P.meta||{{}};
const f=(x,d=3)=>x==null?'—':(Number(x).toFixed(d)); const bytes=n=>{{if(n==null)return'—';let x=Number(n);for(const u of ['B','KiB','MiB','GiB','TiB']){{if(x<1024||u==='TiB')return x.toFixed(x<10?2:1)+' '+u;x/=1024}}}};
function table(rows, cols){{if(!rows?.length)return'<p class="muted">无</p>';return '<table><thead><tr>'+cols.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+String(c[2]?c[2](r[c[0]]):r[c[0]]??'')+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}}
const ext=M.evaluation?.external||{{}}, inn=M.evaluation?.internal||{{}};
const cards=[['样本',Number(M.labels_count||0).toLocaleString()],['Views',Object.keys(M.shapes||{{}}).length],['Kernels',(M.gamma||[]).length],['总时间',f(M.total_seconds,2)+' s'],['Fit',f(M.fit_seconds,2)+' s'],['Peak RAM',bytes(M.peak_rss_bytes)],['ACC',ext.ACC==null?'—':f(ext.ACC,4)],['NMI',ext.NMI==null?'—':f(ext.NMI,4)],['ARI',ext.ARI==null?'—':f(ext.ARI,4)]];
document.querySelector('#cards').innerHTML=cards.map(x=>'<div class="card"><span>'+x[0]+'</span><strong>'+x[1]+'</strong></div>').join('');
const D=M.diagnostics||{{}}, W=M.warnings||D.warnings||[];document.querySelector('#warnings').innerHTML=W.length?'<p style="color:#991b1b"><b>Warning:</b><br>'+W.map(x=>String(x)).join('<br>')+'</p>':'<p class="muted">No configured structural/numerical warning.</p>';const sd=D.spectral_boundary||{{}},gd=D.ground_truth||{{}},so=D.solver||{{}};document.querySelector('#diag').innerHTML=table([{{k:'GT classes / requested c',v:gd.available?(gd.true_clusters+' / '+gd.requested_c):'—'}},{{k:'Final inner residual',v:so.final_inner_residual??'—'}},{{k:'lambda_c',v:sd.lambda_c??'—'}},{{k:'lambda_c+1',v:sd.lambda_c_plus_1??'—'}},{{k:'gap',v:sd.gap??'—'}},{{k:'relative gap',v:sd.relative_gap??'—'}},{{k:'spectral status',v:sd.status??'—'}}],[['k','Item'],['v','Value']]);
function line(cv, series, ymin=null, ymax=null){{const c=document.querySelector(cv),g=c.getContext('2d'),W=c.width,H=c.height,p=48;g.clearRect(0,0,W,H);let vals=series.flatMap(s=>(s.v||[]).filter(Number.isFinite));if(!vals.length){{g.fillStyle='#687386';g.fillText('no trajectory data',p,40);return;}}if(ymin==null)ymin=Math.min(...vals);if(ymax==null)ymax=Math.max(...vals);if(ymax<=ymin){{const pad=Math.max(Math.abs(ymax)*1e-6,1e-9);ymin-=pad;ymax+=pad;}}else{{const pad=(ymax-ymin)*0.08;ymin-=pad;ymax+=pad;}}g.strokeStyle='#dfe5ee';g.strokeRect(p,20,W-p-15,H-55);g.fillStyle='#687386';g.font='12px system-ui';g.fillText(Number(ymax).toPrecision(6),2,30);g.fillText(Number(ymin).toPrecision(6),2,H-38);series.forEach((s,j)=>{{g.strokeStyle=['#2457d6','#dc2626','#16a34a','#9333ea','#0891b2','#c2410c','#7c3aed','#4d7c0f'][j%8];g.fillStyle=g.strokeStyle;g.lineWidth=2;g.beginPath();let pts=[];(s.v||[]).forEach((v,i)=>{{if(!Number.isFinite(v))return;let x=p+(i/Math.max(1,s.v.length-1))*(W-p-15),y=20+(ymax-v)/(ymax-ymin)*(H-75);pts.push([x,y]);if(pts.length>1)g.lineTo(x,y);else g.moveTo(x,y)}});g.stroke();pts.forEach(([x,y])=>{{g.beginPath();g.arc(x,y,3.5,0,Math.PI*2);g.fill()}});if(s.name)g.fillText(s.name,p+8+j*105,H-12);}});}}
const T=M.iteration_metrics||[]; const objs=T.length?T.map(x=>Number(x.objective)):(M.objective||[]).map(Number); line('#obj',[{{name:'J (lower)',v:objs}}],null,null);
const acceptedUpdates=Math.max(0,T.length-1);const OS=M.optimization_summary||{{}};let gmove=0;if(T.length>1&&Array.isArray(T[0].gamma)&&Array.isArray(T.at(-1).gamma)){{for(let i=0;i<Math.min(T[0].gamma.length,T.at(-1).gamma.length);i++)gmove=Math.max(gmove,Math.abs(Number(T.at(-1).gamma[i])-Number(T[0].gamma[i])));}}const j0=objs.length?objs[0]:NaN,jf=objs.length?objs.at(-1):NaN;document.querySelector('#optNote').innerHTML=acceptedUpdates===0?'<b style="color:#991b1b">没有 accepted outer update；权重没有真正迭代。</b> 请结合 strict acceptance / termination / residual 判断原因。':('accepted updates=<b>'+acceptedUpdates+'</b> · strict J: '+Number(j0).toPrecision(8)+' → '+Number(jf).toPrecision(8)+' · monotone='+(OS.accepted_objective_monotone===false?'NO':'YES')+' · max positive step='+(Number(OS.max_positive_objective_step||0).toExponential(2))+' · max |Δγ|='+gmove.toExponential(3)+' · termination='+(M.simplemkkm_termination||'—'));
const LS=M.line_search||[];line('#ls',[{{name:'trial J',v:LS.map(x=>Number(x.objective))}}],null,null);document.querySelector('#lsTable').innerHTML=table(LS.slice(-40),[['outer','Outer'],['trial','Trial'],['phase','Phase'],['step','Step',x=>Number(x).toExponential(2)],['objective','J',x=>Number(x).toPrecision(9)],['inner_residual','Residual',x=>Number(x).toExponential(2)],['certified','Certified']]);
const qseries=[{{name:'ACC',v:T.map(x=>Number(x.ACC))}},{{name:'NMI',v:T.map(x=>Number(x.NMI))}},{{name:'ARI',v:T.map(x=>Number(x.ARI))}}];line('#qual',qseries,0,1);const scopes=[...new Set(T.map(x=>x.metric_scope).filter(Boolean))];document.querySelector('#trajNote').textContent=T.length?('accepted states='+T.length+'（真正更新 '+acceptedUpdates+' 次） · metric scope: '+(scopes.join(', ')||'OFF')+' · n='+(T.find(x=>x.metric_n)?.metric_n??'—')+' · 诊断离散化使用固定随机种子。'):'ground truth / trajectory metrics unavailable';
let gm=[];const ng=Math.max(0,...T.map(x=>Array.isArray(x.gamma)?x.gamma.length:0));for(let j=0;j<ng;j++)gm.push({{name:'γ'+(j+1),v:T.map(x=>Array.isArray(x.gamma)?Number(x.gamma[j]):NaN)}});line('#gam',gm,0,null);
const proto=M.protocol||{{}};document.querySelector('#protocol').innerHTML='<table><tbody>'+Object.entries(proto).map(([k,v])=>'<tr><th>'+k+'</th><td>'+String(v??'—')+'</td></tr>').join('')+'</tbody></table>';document.querySelector('#flowline').innerHTML=M.engine==='reference_dense'?'<b>Row L2</b>→<b>Dense all-pairs φ</b>→<b>kcenter + knorm</b>→<b>SimpleMKKM</b>→<b>dense eigsh</b>→<b>row-normalized H</b>→<b>replicated Lloyd</b>→<b>Evaluation</b>':'<b>Data blocks</b>→<b>ANN candidates</b>→<b>Exact rerank</b>→<b>Sparse supports</b>→<b>φ + PSD</b>→<b>implicit center/norm</b>→<b>SimpleMKKM / LOBPCG</b>→<b>bounded-memory Lloyd</b>→<b>Evaluation</b>';
const R=P.resources||[], total=R.at(-1)?.ram_total_bytes||1; line('#res',[{{name:'proc CPU',v:R.map(x=>Number(x.process_cpu_percent||0))}},{{name:'system CPU',v:R.map(x=>Number(x.system_cpu_percent||0))}},{{name:'RSS %',v:R.map(x=>100*Number(x.process_rss_bytes||0)/total)}}],0,100);
document.querySelector('#profile').innerHTML=table(M.profile||[],[['stage','Stage'],['seconds','Seconds',x=>f(x,3)],['cpu_core_equiv','CPU cores(avg)',x=>f(x,2)],['rss_peak_bytes','Peak RSS',bytes],['read_bytes','Read',bytes],['write_bytes','Write',bytes]]);
let er=[];Object.entries(M.evaluation||{{}}).forEach(([grp,o])=>Object.entries(o||{{}}).forEach(([k,v])=>er.push({{grp,k,v:typeof v==='number'?f(v,6):v}})));document.querySelector('#eval').innerHTML=table(er,[['grp','Type'],['k','Metric'],['v','Value']]);
document.querySelector('#weights').innerHTML='<h3>View</h3>'+table(M.view_weight_table||[],[['view','View'],['sum_gamma','Σγ',x=>f(x,6)],['sum_gamma2','Σγ²',x=>f(x,6)]])+'<h3>Kernel</h3>'+table(M.gamma_table||[],[['view','View'],['kernel','Kernel'],['gamma','γ',x=>f(x,7)],['gamma2','γ²',x=>f(x,7)]]);
const br=(P.blocks||[]).slice(-300).reverse().map(e=>Object.assign({{stage:e.stage||'',progress:e.progress}},e.detail||{{}}));document.querySelector('#blocks').innerHTML=table(br,[['block_kind','Block'],['stage','Stage'],['view','View'],['family','Family'],['rows_start','Start'],['rows_end','End'],['lloyd_iter','Lloyd iter']]);
const env=[['SC-SimpleMKKM version',M.software_version||'—'],['Platform',M.platform],['Python',M.python],['CPU',M.cpu_model],['Physical cores',M.physical_cores],['Logical cores',M.logical_cores],['RAM total',bytes(M.ram_total_bytes)],['FAISS',M.faiss],['Run dir',M.run_dir],['Labels',M.labels_path],['Result JSON',M.result_json||'result.json']];document.querySelector('#env').innerHTML='<table><tbody>'+env.map(x=>'<tr><th>'+x[0]+'</th><td><code>'+String(x[1]??'—')+'</code></td></tr>').join('')+'</tbody></table>';
</script></body></html>'''
    path = root / 'RUN_REPORT.html'
    path.write_text(doc, encoding='utf-8')
    return path


def write_tabular_artifacts(run_dir: str | Path, meta: dict) -> dict[str, str]:
    root=Path(run_dir); out={}
    def write_csv(name, rows):
        path=root/name
        if rows:
            keys=[]
            for r in rows:
                for k in r:
                    if k not in keys: keys.append(k)
            with path.open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
        else: path.write_text('',encoding='utf-8')
        out[name]=str(path); return path
    traj=meta.get('iteration_metrics') or []
    if traj:
        objrows=[]; metricrows=[]; grow=[]
        for r in traj:
            base={k:v for k,v in r.items() if k!='gamma'}
            objrows.append({k:base.get(k) for k in ['accepted_iter','objective','inner_iterations','inner_residual','certified','reason']})
            metricrows.append(base)
            for j,g in enumerate(r.get('gamma') or []): grow.append({'accepted_iter':r.get('accepted_iter'),'kernel_index':j,'gamma':g})
        write_csv('objective.csv',objrows)
        write_csv('iteration_metrics.csv',metricrows)
        write_csv('gamma_trajectory.csv',grow)
    else:
        write_csv('objective.csv',[{'outer_iter':i+1,'objective':v,'inner_iterations':(meta.get('inner_iterations') or [None]*999)[i] if i < len(meta.get('inner_iterations') or []) else None,'inner_residual':(meta.get('inner_residual') or [None]*999)[i] if i < len(meta.get('inner_residual') or []) else None,'inner_seconds':(meta.get('inner_seconds') or [None]*999)[i] if i < len(meta.get('inner_seconds') or []) else None} for i,v in enumerate(meta.get('objective') or [])])
    write_csv('line_search.csv',meta.get('line_search') or [])
    write_csv('acceptance_checks.csv',meta.get('acceptance_checks') or [])
    write_csv('kernel_weights.csv',meta.get('gamma_table') or [])
    write_csv('view_weights.csv',meta.get('view_weight_table') or [])
    (root/'graph_diagnostics.json').write_text(json.dumps(meta.get('graphs') or {},indent=2,ensure_ascii=False),encoding='utf-8'); out['graph_diagnostics.json']=str(root/'graph_diagnostics.json')
    (root/'diagnostics.json').write_text(json.dumps(meta.get('diagnostics') or {},indent=2,ensure_ascii=False),encoding='utf-8'); out['diagnostics.json']=str(root/'diagnostics.json')
    env={k:meta.get(k) for k in ['python','platform','os_name','numpy','faiss','hnswlib','pynndescent','threadpoolctl','psutil','cpu_model','physical_cores','logical_cores','ram_total_bytes']}
    (root/'environment.json').write_text(json.dumps(env,indent=2,ensure_ascii=False),encoding='utf-8'); out['environment.json']=str(root/'environment.json')
    resources=_read_jsonl(root/'resources.jsonl')
    if resources: write_csv('resources.csv',resources)
    events=_read_jsonl(root/'events.jsonl'); blocks=[]
    for e in events:
        d=e.get('detail') or {}
        if d.get('block_kind'):
            blocks.append({'time':e.get('time'),'stage':e.get('stage'),'progress':e.get('progress'),**d})
    if blocks: write_csv('blocks.csv',blocks)
    return out


def write_artifact_manifest(run_dir: str | Path, meta: dict, extra: dict[str,str] | None=None) -> Path:
    root=Path(run_dir); extra=extra or {}
    files=[
        ('labels.npy','result','最终离散标签；int32；默认永久保存'),
        ('result.npz','result','γ、objective、centers 等小型数值结果；实际文件可能位于 run_dir 外'),
        ('result.json','result','权威元数据与路径索引'),
        ('evaluation.json','paper','内部/外部聚类指标'),
        ('paper_summary.csv','paper','论文表格单行摘要'),('paper_summary.json','paper','论文摘要 JSON'),
        ('objective.csv','paper','strict accepted outer-state objective 轨迹'),('iteration_metrics.csv','paper','strict accepted iteration 的 J/ACC/NMI/ARI/residual 轨迹'),('gamma_trajectory.csv','paper','accepted iteration 的逐核权重轨迹'),('line_search.csv','diagnostic','cheap projected/golden line-search 全过程'),('acceptance_checks.csv','diagnostic','候选点 strict recheck 与最终接受/拒绝依据'),('kernel_weights.csv','paper','逐 kernel 权重'),('view_weights.csv','paper','按 view 汇总权重'),
        ('profile.csv','profile','阶段 wall/CPU/RAM/I/O'),('profile.json','profile','阶段 profile JSON'),
        ('events.jsonl','process','阶段与 block 事件'),('resources.jsonl','process','低开销 CPU/RAM/I/O 时间序列'),('resources.csv','paper','资源曲线 CSV'),('blocks.csv','paper','计算 block 时间线 CSV'),
        ('graph_diagnostics.json','diagnostic','support/连通性等图诊断'),('diagnostics.json','diagnostic','GT/c、谱边界、solver 认证与警告汇总'),('environment.json','reproducibility','机器与软件环境'),
        ('RUN_REPORT.html','teaching','自包含运行报告：过程、资源、结果、profile'),
        ('HANDOFF.md','handoff','人工可读交接'),('handoff.json','handoff','机器可读交接'),
    ]
    items=[]
    for name,role,desc in files:
        p=root/name
        # result.npz is normally outside run_dir; resolve from meta if available.
        if name=='result.npz': p=Path(meta.get('result_npz') or meta.get('out') or '') if (meta.get('result_npz') or meta.get('out')) else p
        items.append({'name':name,'role':role,'description':desc,'path':str(p),'exists':p.exists() if str(p) else False,'bytes':p.stat().st_size if str(p) and p.exists() and p.is_file() else None})
    for name,path in extra.items():
        if name not in {x['name'] for x in items}:
            p=Path(path);items.append({'name':name,'role':'extra','description':'additional artifact','path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None})
    manifest={'run_dir':str(root),'storage_policy':{'always_keep':['labels.npy','result.json','evaluation.json','paper_summary.csv','profile.csv','RUN_REPORT.html','HANDOFF.md'],'keep_for_diagnosis':['events.jsonl','resources.jsonl','handoff.json'],'not_saved_by_default':['full embedding H','ANN index','full sparse operators/support caches']},'artifacts':items}
    path=root/'ARTIFACTS.json';path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8');return path
