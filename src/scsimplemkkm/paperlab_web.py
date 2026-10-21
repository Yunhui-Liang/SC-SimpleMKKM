from __future__ import annotations

import argparse, csv, json, os, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from ._version import __version__
from .paperdata import list_recipes, prepare_recipe
from .paperlab import run_plan, status_plan
from .paperstats import aggregate_runs
from .paperfigs import generate as generate_figures
from .papertables import generate as generate_tables
from .runtime import system_info
from .prepared_catalog import status_for_recipe
from .shared_workspace import workspace, update_workspace, legacy_data_roots
from .runstore import list_runs, load_run

ROOT = Path(__file__).resolve().parent / "paperlabui"

class State:
    def __init__(self):
        self.lock=threading.Lock(); self.mode='idle'; self.message=''; self.log=[]; self.result=None; self.error=None; self.stop_after_current=False; self.started=None; self.finished=None; self.plan_path=None
    def reset(self,mode,msg=''):
        with self.lock:
            self.mode=mode; self.message=msg; self.log=[]; self.result=None; self.error=None; self.stop_after_current=False; self.started=time.time(); self.finished=None
    def add(self,msg):
        with self.lock:
            self.message=str(msg); self.log.append(str(msg)); self.log=self.log[-300:]
    def snap(self):
        with self.lock:
            return {'mode':self.mode,'message':self.message,'log':self.log[-120:],'result':self.result,'error':self.error,'stop_after_current':self.stop_after_current,'started':self.started,'finished':self.finished,'plan_path':self.plan_path}
STATE=State()

def _recipe_rows(root: Path):
    rows=[]
    for x in list_recipes():
        r=dict(x)
        info=status_for_recipe(root,str(x['id']))
        r.update(ready=info['ready'], status=info['status'], status_reason=info['message'],
                 manifest=info['manifest'] if info['ready'] else None,
                 last_failure=info.get('last_failure'), relocated=info.get('relocated',False))
        if info['ready']:
            r['n']=info['n']; r['views']=len(info['views']); r['classes']=info['classes']
        if r.get('role')=='standard_mvc': r['group']='Canonical / real MVC'
        elif r.get('role')=='derived_multiview': r['group']='Large derived multi-view'
        else: r['group']='Synthetic scale'
        if r['id'] in {'pamap2_mvc','realdisp_mvc'}: r['group']='Large real multi-sensor MVC'
        rows.append(r)
    return rows


def _auto_repeats(n):
    if n is None: return 3
    n=int(n)
    if n<=30000: return 10
    if n<=150000: return 5
    if n<=2_000_000: return 3
    return 1

def _base_defaults():
    return {
      'auto_clusters_from_ground_truth':True,'k':10,'mmap':True,'memory_mode':'auto','search':'ann','ann_backend':'auto','candidate_factor':4,'query_batch_size':8192,'support_mode':'directed','support_policy':'ranking_grouped','diffusion_power':1,
      'solver':{'inner_solver':'auto','inner_tol':1e-4,'inner_max_iter':200,'outer_tol':1e-5,'outer_max_iter':100,'spmm_backend':'auto'},
      'discretization':{'backend':'batched_lloyd','block_rows':262144,'max_iter':30,'tol':1e-4,'init_sample':200000,'init_restarts':5,'init_local_iter':8},
      'trajectory':{'enabled':False},'spectral_diagnostics':{'enabled':False},'evaluation':{'enabled':True,'block_rows':262144},
      'resources':{'max_ram_fraction':0.72,'reserve_gb':5.0,'max_cpu_fraction':0.9,'leave_cores':1,'monitor_interval':1.0,'allow_oom_risk':False}
    }

def make_plan(req: dict) -> Path:
    ids=[str(x) for x in req.get('datasets') or []]
    if not ids: raise ValueError('Select at least one dataset.')
    catalog={x['id']:x for x in list_recipes()}
    data_root=Path(req.get('data_root') or workspace()['data_root']).expanduser().resolve()
    runs_root=Path(req.get('runs_root') or workspace()['runs_root']).expanduser().resolve()
    plan_dir=Path(req.get('plan_dir') or workspace()['plans_root']).expanduser().resolve(); plan_dir.mkdir(parents=True,exist_ok=True)
    mode=str(req.get('experiment_mode') or 'main')
    fixed=int(req.get('repeats') or 5); policy=str(req.get('repeat_policy') or 'auto')
    seed_start=int(req.get('seed_start') or 0); dense=bool(req.get('dense_compare',True)); dense_max=int(req.get('dense_max_n') or 15000)
    sparse={'engine':'scalable_sparse','kernel_bank':{'mode':'shared','preset':'rbf_self_tuned','local_scale_k':7}}
    protocols={'sparse_self':sparse}
    if dense: protocols['dense_self']={'engine':'reference_dense','kernel_bank':{'mode':'shared','preset':'rbf_self_tuned','local_scale_k':7},'trajectory':{'enabled':False},'search':'exact'}
    datasets=[]
    for rid in ids:
        if rid not in catalog: raise KeyError(f'unknown recipe: {rid}')
        meta=catalog[rid]; n=meta.get('n'); rep=_auto_repeats(n) if policy=='auto' else fixed
        ds={'id':rid,'source':'recipe','seeds':list(range(seed_start,seed_start+rep)),'phase':'main'}
        if mode=='thread_scaling':
            ds['protocols']=['sparse_self']; ds['seeds']=[seed_start]; ds['threads']='physical_ladder'
        else:
            ds['protocols']=['sparse_self'] + (['dense_self'] if dense and n is not None and int(n)<=dense_max else [])
        datasets.append(ds)
    plan={'schema':'scsimplemkkm.paperplan.v1','name':f'interactive_{mode}_{time.strftime("%Y%m%d_%H%M%S")}', 'phase':mode,
          'data_root':str(data_root),'runs_root':str(runs_root),'prepared_roots':[str(data_root)],'repeats':fixed,'seed_start':seed_start,
          'default_protocols':['sparse_self'],'config_defaults':_base_defaults(),'protocols':protocols,'datasets':datasets}
    if mode=='ablation':
        protocols.update({
          'sparse_rbf7':{'engine':'scalable_sparse','kernel_bank':{'mode':'shared','preset':'rbf7'}},
          'sparse_linear':{'engine':'scalable_sparse','kernel_bank':{'mode':'shared','preset':'linear'}},
        })
        for ds in datasets: ds['protocols']=['sparse_self','sparse_rbf7','sparse_linear']
    path=plan_dir/f"{plan['name']}.json"; path.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
    with STATE.lock: STATE.plan_path=str(path)
    return path

def _prepare_worker(ids,root,overwrite):
    STATE.reset('preparing',f'Preparing {len(ids)} datasets')
    ok=[]; fail=[]
    for i,rid in enumerate(ids,1):
        if STATE.stop_after_current: break
        STATE.add(f'[{i}/{len(ids)}] preparing {rid}')
        try:
            if status_for_recipe(root,rid)['ready']:
                ok.append({'id':rid,'status':'ready (reused)','manifest':str(Path(root)/rid/'manifest.json')})
                continue
            prepare_recipe(rid,root,overwrite=overwrite)
            check=status_for_recipe(root,rid)
            if not check['ready']:
                raise ValueError(f"preparation incomplete: {check['status']}: {check['message']}")
            (Path(root)/rid/'_prepare_error.json').unlink(missing_ok=True)
            ok.append({'id':rid,'status':'ready','manifest':check['manifest']})
        except Exception as e:
            err=f'{type(e).__name__}: {e}'
            fail.append({'id':rid,'error':err}); STATE.add(f'FAILED {rid}: {err}')
            p=Path(root)/rid/'_prepare_error.json'; p.parent.mkdir(parents=True,exist_ok=True)
            try: p.write_text(json.dumps({'id':rid,'error':err,'timestamp':time.time()},ensure_ascii=False),encoding='utf-8')
            except OSError: pass
    with STATE.lock:
        STATE.mode='done' if not fail else 'done_with_errors'; STATE.result={'prepared':len(ok),'failed':fail,'items':ok}; STATE.finished=time.time()

def _run_worker(plan_path,prepare,retry):
    STATE.reset('running','Unattended experiment queue started')
    with STATE.lock: STATE.plan_path=str(plan_path)
    try:
        # Process one job per call so UI can stop cleanly after the current job.
        while True:
            st=status_plan(plan_path); counts=st.get('counts') or {}; pending=int(counts.get('pending',0))+int(counts.get('failed_preflight',0) if retry else 0)
            if pending<=0: break
            if STATE.stop_after_current: STATE.add('Stop requested: queue paused before next job.'); break
            before=counts.copy(); STATE.add('Running next pending job…')
            run_plan(plan_path,prepare=prepare,retry_failed=retry,max_jobs=1,cooldown_seconds=0)
            after=status_plan(plan_path).get('counts') or {}; STATE.add(f'queue: {after}')
            if after==before and int(after.get('pending',0))>0:
                STATE.add('No queue progress detected; stopping to avoid an infinite loop.'); break
        st=status_plan(plan_path)
        with STATE.lock: STATE.mode='paused' if STATE.stop_after_current else 'done'; STATE.result=st; STATE.finished=time.time()
    except Exception:
        with STATE.lock: STATE.mode='failed'; STATE.error=traceback.format_exc(); STATE.finished=time.time()

def _outputs(req):
    runs=Path(req.get('runs_root') or workspace()['runs_root']).expanduser().resolve(); out=Path(req.get('out_dir') or runs/'_paper_outputs').expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    stats=aggregate_runs(runs,out/'stats'); figs=generate_figures(runs,out/'figures'); tabs=generate_tables(runs,out/'tables')
    return {'runs_root':str(runs),'out_dir':str(out),'stats':stats,'figures':figs,'tables':tabs}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): return
    def _json(self,obj,status=200):
        b=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n=int(self.headers.get('Content-Length','0')); return json.loads((self.rfile.read(n) if n else b'{}').decode())
    def _static(self,p,ctype):
        p=ROOT/p
        if not p.exists(): self.send_error(404); return
        b=p.read_bytes(); self.send_response(200); self.send_header('Content-Type',ctype); self.send_header('Cache-Control','no-store, max-age=0'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u=urlparse(self.path); path=u.path; q=parse_qs(u.query)
        if path in {'/','/index.html'}: return self._static(Path('index.html'),'text/html; charset=utf-8')
        if path=='/style.css': return self._static(Path('style.css'),'text/css; charset=utf-8')
        if path=='/app.js': return self._static(Path('app.js'),'application/javascript; charset=utf-8')
        if path=='/api/health': return self._json({'ok':True,'service':'paperlab-web','version':__version__,'index_exists':(ROOT/'index.html').exists()})
        if path=='/api/workspace': return self._json(workspace())
        if path=='/api/runs':
            root=(q.get('root') or [workspace()['runs_root']])[0]
            return self._json({'root':str(Path(root).resolve()),'runs':list_runs(root)})
        if path=='/api/run-detail':
            rid=(q.get('run_id') or [''])[0]; root=(q.get('root') or [workspace()['runs_root']])[0]
            if not rid: return self._json({'error':'run_id required'},400)
            try: return self._json(load_run(rid,root))
            except FileNotFoundError as e: return self._json({'error':str(e)},404)
        if path=='/api/env': return self._json(system_info())
        if path=='/api/recipes':
            root=Path((q.get('root') or [workspace()['data_root']])[0]).expanduser().resolve(); return self._json({'recipes':_recipe_rows(root),'root':str(root)})
        if path=='/api/state':
            s=STATE.snap(); pp=s.get('plan_path')
            if pp and Path(pp).exists():
                try: s['plan_status']=status_plan(pp)
                except Exception as e: s['plan_status_error']=str(e)
            return self._json(s)
        self.send_error(404)
    def do_POST(self):
        path=urlparse(self.path).path
        try:
            req=self._body()
            if path=='/api/workspace': return self._json(update_workspace(req))
            if path=='/api/audit-data':
                root=Path(req.get('data_root') or workspace()['data_root']).expanduser().resolve()
                rows=_recipe_rows(root)
                dest=root/'paperlab_dataset_status.csv'; dest.parent.mkdir(parents=True,exist_ok=True)
                cols=['id','title','role','status','ready','status_reason','manifest','n','views']
                with dest.open('w',newline='',encoding='utf-8-sig') as f:
                    writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
                return self._json({'ok':True,'path':str(dest),'rows':len(rows),
                                   'ready':sum(1 for x in rows if x['ready']),
                                   'problems':[{'id':x['id'],'status':x['status'],'reason':x['status_reason']} for x in rows if not x['ready']]})
            if path=='/api/prepare':
                ids=req.get('datasets') or []
                if STATE.snap()['mode'] in {'preparing','running'}: return self._json({'error':'PaperLab is busy'},409)
                root=str(Path(req.get('data_root') or workspace()['data_root']).expanduser().resolve())
                threading.Thread(target=_prepare_worker,args=(ids,root,bool(req.get('overwrite',False))),daemon=True).start(); return self._json({'ok':True},202)
            if path=='/api/plan':
                pp=make_plan(req); st=status_plan(pp)
                return self._json({'ok':True,'plan_path':str(pp),'status':st})
            if path=='/api/run':
                # Explicit plan paths support resuming an existing queue. Without one,
                # always build from the submitted dataset/parameter selection.
                # STATE.plan_path is the last plan *globally*, not this request.
                pp=req.get('plan_path')
                if not pp: pp=str(make_plan(req))
                if STATE.snap()['mode'] in {'preparing','running'}: return self._json({'error':'PaperLab is busy'},409)
                threading.Thread(target=_run_worker,args=(Path(pp),bool(req.get('prepare',True)),bool(req.get('retry_failed',False))),daemon=True).start(); return self._json({'ok':True,'plan_path':str(pp)},202)
            if path=='/api/stop':
                with STATE.lock: STATE.stop_after_current=True
                return self._json({'ok':True,'message':'will stop after current job'})
            if path=='/api/outputs': return self._json(_outputs(req))
        except Exception as e: return self._json({'error':f'{type(e).__name__}: {e}'},400)
        self.send_error(404)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--port',type=int,default=0); ap.add_argument('--ready-file',default=None); a=ap.parse_args()
    srv=ThreadingHTTPServer((a.host,a.port),Handler); port=srv.server_address[1]; ready={'ok':True,'service':'paperlab-web','version':__version__,'pid':os.getpid(),'url':f'http://{a.host}:{port}/','index_exists':(ROOT/'index.html').exists()}
    if a.ready_file:
        p=Path(a.ready_file); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(ready),encoding='utf-8')
    print(f"SC-SimpleMKKM PaperLab UI: {ready['url']}",flush=True)
    try: srv.serve_forever()
    finally: srv.server_close()
    return 0
if __name__=='__main__': raise SystemExit(main())
