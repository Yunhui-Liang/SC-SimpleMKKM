from __future__ import annotations

import argparse
import os
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .cli import doctor_report, plan_multiview_config, run_multiview_config, autofill_config
from .runtime import RunControl, CancelledError
from .datasets import catalog as dataset_catalog, prepare_dataset, generate_scale_tier, inspect_multiview_mat, prepare_multiview_mat, prepare_multiview_directory
from .runstore import managed_config, list_runs, load_run, report_path
from .paperstats import aggregate_runs
from ._version import __version__
from .shared_workspace import workspace, update_workspace
from .paperdata import list_recipes
from .prepared_catalog import status_for_recipe

ROOT = Path(__file__).resolve().parent / 'webui'


class JobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = 'idle'
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.config: dict | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.log: list[str] = []
        self.current_event: dict | None = None
        self.metrics: list[dict] = []
        self.blocks: list[dict] = []
        self.plan: dict | None = None
        self.control: RunControl | None = None
        self._last_log_stage = ''
        self._last_log_time = 0.0

    def reset(self, config: dict, plan: dict) -> None:
        with self.lock:
            self.status = 'running'
            self.started_at = time.time()
            self.finished_at = None
            self.config = config
            self.result = None
            self.error = None
            self.log = []
            self.current_event = None
            self.metrics = []
            self.blocks = []
            self.plan = plan
            self.control = RunControl()
            self._last_log_stage = ''
            self._last_log_time = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            elapsed = None if self.started_at is None else (self.finished_at or now) - self.started_at
            return {
                'status': self.status,
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'elapsed_seconds': elapsed,
                'result': self.result,
                'error': self.error,
                'log': self.log[-120:],
                'current_event': self.current_event,
                'metrics': self.metrics[-180:],
                'blocks': self.blocks[-120:],
                'plan': self.plan,
            }

    def append(self, msg: str) -> None:
        with self.lock:
            self.log.append(str(msg))
            if len(self.log) > 1000:
                self.log = self.log[-800:]

    def progress(self, ev: dict) -> None:
        if ev.get('type', '').startswith('profile_'):
            return
        now = time.time()
        with self.lock:
            self.current_event = ev
            stage = str(ev.get('stage') or '')
            msg = ev.get('message')
            # Log only stage transitions / outer summaries / every ~2s.
            detail = ev.get('detail') or {}
            if detail.get('block_kind'):
                self.blocks.append({'time': ev.get('time'), 'stage': stage, 'progress': ev.get('progress'), **detail})
                if len(self.blocks) > 2000:
                    self.blocks = self.blocks[-1200:]
            important = stage in {'simplemkkm:outer', 'postprocess:kmeans'}
            if msg and (stage != self._last_log_stage or important or now - self._last_log_time >= 2.0):
                self.log.append(str(msg))
                self._last_log_stage = stage
                self._last_log_time = now
                if len(self.log) > 1000:
                    self.log = self.log[-800:]

    def metric(self, sample: dict) -> None:
        with self.lock:
            self.metrics.append(sample)
            if len(self.metrics) > 1800:
                self.metrics = self.metrics[-1200:]



class DataJobState:
    def __init__(self) -> None:
        self.lock=threading.Lock(); self.status='idle'; self.message=''; self.result=None; self.error=None
    def snapshot(self):
        with self.lock: return {'status':self.status,'message':self.message,'result':self.result,'error':self.error}

DATAJOB=DataJobState()

def _run_dataset_job(req: dict) -> None:
    with DATAJOB.lock:
        DATAJOB.status='running'; DATAJOB.message='Preparing dataset...'; DATAJOB.result=None; DATAJOB.error=None
    try:
        mode=req.get('mode','real'); out=req.get('out_root') or 'datasets'
        if mode=='scale':
            n=int(req.get('n',1000000)); d=int(req.get('d',16)); c=int(req.get('clusters',10))
            dims_raw=req.get('dims')
            dims=None
            if isinstance(dims_raw, str):
                dims=[int(x.strip()) for x in dims_raw.split(',') if x.strip()]
            elif isinstance(dims_raw, (list, tuple)):
                dims=[int(x) for x in dims_raw]
            res=generate_scale_tier(out,n,d,c,dims=dims,noise=float(req.get('noise',0.30)))
        elif mode=='mv_mat':
            res=prepare_multiview_mat(
                str(req['path']), out, name=req.get('name') or None,
                views_key=str(req.get('views_key') or 'auto'),
                labels_key=str(req.get('labels_key') or 'auto'),
                aux_key=str(req.get('aux_key') or 'M'),
                k=int(req.get('k',10)), search=str(req.get('search') or 'exact'),
                overwrite=bool(req.get('overwrite',False)),
            )
        elif mode=='mv_mat_dir':
            res=prepare_multiview_directory(
                str(req['path']), out, pattern=str(req.get('pattern') or '*.mat'),
                views_key=str(req.get('views_key') or 'auto'), labels_key=str(req.get('labels_key') or 'auto'),
                aux_key=str(req.get('aux_key') or 'M'), k=int(req.get('k',10)),
                search=str(req.get('search') or 'exact'), overwrite=bool(req.get('overwrite',False)),
            )
        else:
            res=prepare_dataset(str(req['dataset']),out)
        with DATAJOB.lock: DATAJOB.status='done'; DATAJOB.message='Dataset ready'; DATAJOB.result=res
    except Exception:
        err=traceback.format_exc()
        with DATAJOB.lock: DATAJOB.status='failed'; DATAJOB.error=err; DATAJOB.message='Dataset preparation failed'

JOB = JobState()


def _run_job(config: dict, plan: dict) -> None:
    JOB.reset(config, plan)
    JOB.append('Resource preflight accepted. Starting SC-SimpleMKKM...')
    try:
        control = JOB.control
        out, meta = run_multiview_config(
            config,
            progress_callback=JOB.progress,
            control=control,
            monitor_callback=JOB.metric,
        )
        JOB.append(f'Finished. Result: {out}')
        with JOB.lock:
            JOB.status = 'done'
            JOB.finished_at = time.time()
            JOB.result = {'result': str(out), 'meta': str(out) + '.json', **meta}
    except CancelledError as e:
        JOB.append(str(e))
        with JOB.lock:
            JOB.status = 'cancelled'
            JOB.finished_at = time.time()
            JOB.error = str(e)
    except Exception:
        err = traceback.format_exc()
        JOB.append(err)
        with JOB.lock:
            JOB.status = 'failed'
            JOB.finished_at = time.time()
            JOB.error = err


class Handler(BaseHTTPRequestHandler):
    server_version = f'SCSimpleMKKMWeb/{__version__}'

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def _body_json(self) -> dict:
        n = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(n) if n else b'{}'
        return json.loads(raw.decode('utf-8'))

    def _static(self, p: Path, ctype: str) -> None:
        if not p.exists():
            self.send_error(404); return
        data = p.read_bytes()
        self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path in {'/', '/index.html'}:
            index = ROOT / 'index.html'
            if index.exists():
                return self._static(index, 'text/html; charset=utf-8')
            # Never return an opaque 404 for the launcher page. If packaging is
            # broken, show an actionable diagnostic page with the resolved path.
            body = (
                '<!doctype html><meta charset="utf-8"><title>SC-SimpleMKKM UI diagnostic</title>'
                '<h2>SC-SimpleMKKM Web UI assets are missing</h2>'
                '<p>The backend is running, but index.html was not packaged.</p>'
                f'<pre>expected: {index}</pre>'
                '<p>Please send windows\run_windows.log and web_console.err.log.</p>'
            ).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if path == '/api/health':
            return self._json({
                'ok': True,
                'service': 'scsimplemkkm-web',
                'version': __version__,
                'pid': os.getpid(),
                'webui_root': str(ROOT),
                'index_exists': (ROOT / 'index.html').exists(),
            })
        if path == '/style.css':
            return self._static(ROOT / 'style.css', 'text/css; charset=utf-8')
        if path == '/app.js':
            return self._static(ROOT / 'app.js', 'application/javascript; charset=utf-8')
        if path == '/api/workspace':
            return self._json(workspace())
        if path == '/api/doctor':
            return self._json(doctor_report())
        if path == '/api/status':
            return self._json(JOB.snapshot())
        if path == '/api/runs':
            root = (qs.get('root') or [workspace()['runs_root']])[0]
            return self._json({'runs': list_runs(root), 'root': root})
        if path == '/api/run-detail':
            run_id = (qs.get('run_id') or [''])[0]
            root = (qs.get('root') or [workspace()['runs_root']])[0]
            if not run_id:
                return self._json({'error': 'run_id is required'}, status=400)
            try:
                return self._json(load_run(run_id, root))
            except FileNotFoundError as e:
                return self._json({'error': str(e)}, status=404)
        if path == '/api/datasets':
            cfg=workspace()
            root = (qs.get('root') or [cfg['dataset_root']])[0]
            mv_root = (qs.get('mv_root') or [cfg['mv_root']])[0]
            paper_root = (qs.get('paper_root') or [cfg['data_root']])[0]
            rows=dataset_catalog(root, mv_root)
            for x in list_recipes():
                status=status_for_recipe(paper_root,str(x['id']))
                item={'id':'paperlab:'+str(x['id']), 'title':x['title'],
                      'kind':'paperlab_recipe', 'category':'PaperLab / '+str(x['role']),
                      'n':status.get('n') or x.get('n') or 0,
                      'd':'/'.join(str(v['d']) for v in status['views']) if status['ready'] else '—',
                      'c':status.get('classes') or x.get('classes') or 0,
                      'tier':str(x.get('views') or '')+'-view', 'ready':status['ready'],
                      'status':status['status'], 'status_reason':status['message'],
                      'bundled':False, 'recommended_engine':'scalable_sparse',
                      'recommended_k':status.get('recommended_k',10),
                      'recommended_kernel_preset':'rbf_self_tuned',
                      'description':str(x.get('description','')),
                      'manifest_path':status['manifest'] if status['ready'] else None,
                      'views':status['views'], 'labels':status['labels']}
                rows.append(item)
            return self._json({'catalog':rows, 'job':DATAJOB.snapshot(),
                               'root':root,'mv_root':mv_root,'paper_root':paper_root})
        if path == '/api/dataset-status':
            return self._json(DATAJOB.snapshot())
        if path == '/run-report':
            run_id = (qs.get('run_id') or [''])[0]
            root = (qs.get('root') or [workspace()['runs_root']])[0]
            if run_id:
                try:
                    return self._static(report_path(run_id, root), 'text/html; charset=utf-8')
                except FileNotFoundError:
                    self.send_error(404); return
            with JOB.lock:
                rp = None if not JOB.result else JOB.result.get('run_report_html')
            if rp and Path(rp).exists():
                return self._static(Path(rp), 'text/html; charset=utf-8')
            self.send_error(404); return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == '/api/workspace': return self._json(update_workspace(self._body_json()))
            if path == '/api/check-paths':
                cfg = self._body_json(); rows=[]
                for v in cfg.get('views', []):
                    p=Path(str(v.get('path','')))
                    rows.append({'name': v.get('name'), 'path': str(p), 'exists': p.exists(), 'size': p.stat().st_size if p.exists() and p.is_file() else None})
                return self._json({'views': rows})
            if path == '/api/plan':
                cfg = autofill_config(self._body_json()); plan = plan_multiview_config(cfg)
                plan['resolved_clusters'] = int(cfg['clusters'])
                return self._json(plan)
            if path == '/api/run':
                cfg0 = autofill_config(self._body_json())
                with JOB.lock:
                    if JOB.status == 'running':
                        return self._json({'error': 'A job is already running.'}, status=409)
                cfg, run_id, run_dir = managed_config(cfg0)
                plan = plan_multiview_config(cfg)
                if not plan.get('allowed', False):
                    return self._json({'error': 'Resource preflight rejected the run.', 'plan': plan}, status=400)
                th = threading.Thread(target=_run_job, args=(cfg, plan), daemon=True, name=f'scsimplemkkm-{run_id}')
                th.start()
                return self._json({'ok': True, 'plan': plan, 'run_id': run_id, 'run_dir': str(run_dir)}, status=202)
            if path == '/api/mv-mat-inspect':
                req=self._body_json()
                return self._json(inspect_multiview_mat(
                    str(req['path']),
                    views_key=str(req.get('views_key') or 'auto'),
                    labels_key=str(req.get('labels_key') or 'auto'),
                    aux_key=str(req.get('aux_key') or 'M'),
                ))
            if path == '/api/dataset-prepare':
                req=self._body_json()
                with DATAJOB.lock:
                    if DATAJOB.status=='running': return self._json({'error':'A dataset job is already running.'},status=409)
                threading.Thread(target=_run_dataset_job,args=(req,),daemon=True,name='scsimplemkkm-data').start()
                return self._json({'ok':True},status=202)
            if path == '/api/paper-stats':
                req=self._body_json()
                root=str(req.get('root') or 'runs')
                out=req.get('out_dir') or None
                result=aggregate_runs(root,out)
                return self._json({
                    'ok': True, 'run_root': result['run_root'], 'out_dir': result['out_dir'],
                    'run_count': len(result['runs']), 'group_count': len(result['groups']),
                    'pair_count': len(result['paired_runs']), 'warnings': result['warnings'],
                    'groups': result['groups'], 'paired_groups': result['paired_groups'],
                })
            if path == '/api/cancel':
                with JOB.lock:
                    c = JOB.control
                    running = JOB.status == 'running'
                if c is not None and running:
                    c.cancel(); JOB.append('Cancellation requested; stopping at the next cooperative checkpoint.')
                    return self._json({'ok': True})
                return self._json({'ok': False, 'error': 'No running job.'}, status=409)
        except Exception as e:
            return self._json({'error': str(e)}, status=400)
        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description='Local browser UI for SC-SimpleMKKM')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765, help='Use 0 to let the OS choose a free port.')
    ap.add_argument('--ready-file', default=None, help='Optional JSON file written after the server is bound and ready.')
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    actual_port = int(httpd.server_address[1])
    ready = {
        'ok': True,
        'service': 'scsimplemkkm-web',
        'version': __version__,
        'pid': os.getpid(),
        'host': args.host,
        'port': actual_port,
        'url': f'http://{args.host}:{actual_port}/',
        'webui_root': str(ROOT),
        'index_exists': (ROOT / 'index.html').exists(),
    }
    if args.ready_file:
        rp = Path(args.ready_file)
        rp.parent.mkdir(parents=True, exist_ok=True)
        tmp = rp.with_suffix(rp.suffix + '.tmp')
        tmp.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(rp)
    print(f'SC-SimpleMKKM Web UI: {ready["url"]}', flush=True)
    print('Low-overhead monitoring is enabled.', flush=True)
    print('No authentication is provided; bind to 127.0.0.1 unless using a trusted tunnel/network.', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
