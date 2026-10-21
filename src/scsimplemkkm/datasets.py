from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import urllib.request
import zipfile
from importlib.resources import files as resource_files
from pathlib import Path
import numpy as np

from .data import prepare_float32_l2
from .benchdata import generate_multiview_gaussian
from .mat_import import read_multiview_mat

# ---------------------------------------------------------------------------
# Dataset catalog
# ---------------------------------------------------------------------------
# ``bundled`` entries are shipped inside the wheel and are meant for laptop
# correctness/reference tests.  ``download`` entries are larger public datasets
# prepared into the same .npy/mmap-ready format on first use.
CATALOG = {
    'iris': {
        'title': 'Iris', 'n': 150, 'd': 4, 'c': 3, 'tier': 'Tiny',
        'kind': 'bundled', 'category': 'Laptop / Dense reference',
        'recommended_engine': 'reference_dense', 'recommended_k': 10,
        'description': 'Tiny 3-class sanity test; ideal for checking the full dense protocol and trajectories.',
    },
    'wine': {
        'title': 'Wine', 'n': 178, 'd': 13, 'c': 3, 'tier': 'Tiny',
        'kind': 'bundled', 'category': 'Laptop / Dense reference',
        'recommended_engine': 'reference_dense', 'recommended_k': 10,
        'description': 'Small 3-class benchmark for full dense SimpleMKKM.',
    },
    'breast_cancer': {
        'title': 'Breast Cancer Wisconsin', 'n': 569, 'd': 30, 'c': 2, 'tier': 'Small',
        'kind': 'bundled', 'category': 'Laptop / Dense reference',
        'recommended_engine': 'reference_dense', 'recommended_k': 10,
        'description': 'Small binary benchmark; still comfortable for dense all-pairs kernels on a laptop.',
    },
    'digits': {
        'title': 'Digits', 'n': 1797, 'd': 64, 'c': 10, 'tier': 'Medium',
        'kind': 'bundled', 'category': 'Laptop / Stress',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'description': '1,797-sample stress test. Sparse mode is the safe default; dense 12-kernel reference can take minutes on a laptop.',
    },
    'synth3k': {
        'title': 'ScaleSynth 3K', 'n': 3000, 'd': 16, 'c': 10, 'tier': '3K',
        'kind': 'bundled', 'category': 'Laptop / Stress',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'description': 'Processed synthetic 3K stress test. Sparse mode is the default; dense 12-kernel mode is deliberately a heavier stress option.',
    },
    'letter': {
        'title': 'Letter Recognition', 'n': 20000, 'd': 16, 'c': 26, 'tier': '20K',
        'kind': 'uci_letter', 'category': 'Public / Scale',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'url': 'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip',
        'doi': '10.24432/C5ZP40',
        'description': '20K public benchmark. Use c=26; dense 12-kernel mode is not a laptop default.',
    },
    'covertype': {
        'title': 'Covertype', 'n': 581012, 'd': 54, 'c': 7, 'tier': '500K',
        'kind': 'uci_covtype', 'category': 'Public / Scale',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'url': 'https://archive.ics.uci.edu/static/public/31/covertype.zip',
        'doi': '10.24432/C50K5N',
        'description': '581K scale benchmark; intended for SC-SimpleMKKM.',
    },
    'poker': {
        'title': 'Poker Hand', 'n': 1025010, 'd': 10, 'c': 10, 'tier': '1M',
        'kind': 'uci_poker', 'category': 'Public / Scale',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'url': 'https://archive.ics.uci.edu/static/public/158/poker+hand.zip',
        'doi': '10.24432/C5KW38',
        'description': 'About 1M samples; intended for ANN/sparse single-node profiling.',
    },
    'higgs': {
        'title': 'HIGGS', 'n': 11000000, 'd': 28, 'c': 2, 'tier': '11M',
        'kind': 'uci_higgs', 'category': 'Public / Scale',
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'url': 'https://archive.ics.uci.edu/static/public/280/higgs.zip',
        'doi': '10.24432/C5V312',
        'description': '11M-scale systems stress test; not a dense-reference target.',
    },
}


def _bundled_dir(dataset_id: str) -> Path:
    # Wheels are installed unpacked by pip/uv in our Windows runner, so the
    # resource is a normal filesystem path.  ``Path(str(...))`` also keeps this
    # simple for source-tree execution.
    return Path(str(resource_files('scsimplemkkm').joinpath('sampledata', dataset_id)))


def _manifest_if_ready(dataset_id: str, out_root: str | Path = 'datasets') -> dict | None:
    info = CATALOG[dataset_id]
    if info['kind'] == 'bundled':
        d = _bundled_dir(dataset_id)
        mp = d / 'manifest.json'; xp = d / 'view1.npy'; yp = d / 'labels.npy'
        if mp.exists() and xp.exists() and yp.exists():
            m = json.loads(mp.read_text(encoding='utf-8'))
            m.update({
                'id': dataset_id,
                'view': str(xp.resolve()),
                'labels': str(yp.resolve()),
                'ready': True,
                'bundled': True,
            })
            return m
        return None
    d = Path(out_root) / dataset_id
    mp = d / 'manifest.json'
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text(encoding='utf-8'))
        xp = Path(str(m.get('view', d / 'view1.npy')))
        yp = Path(str(m.get('labels', d / 'labels.npy')))
        if not xp.is_absolute(): xp = d / xp
        if not yp.is_absolute(): yp = d / yp
        if xp.exists() and yp.exists():
            m.update({'id': dataset_id, 'view': str(xp.resolve()), 'labels': str(yp.resolve()), 'ready': True, 'bundled': False})
            return m
    except Exception:
        return None
    return None


def _resolve_manifest_path(base: Path, value) -> str | None:
    if value in (None, ''):
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())


def discover_prepared_multiview(root: str | Path = 'dataset_cache') -> list[dict]:
    """PaperLab and imported MVC manifests, with path relocation + NPY validation."""
    from .prepared_catalog import prepared_rows
    found=[]
    for item in prepared_rows(root):
        views=item['views']; m=json.loads(Path(item['manifest']).read_text(encoding='utf-8'))
        dims=[v['d'] for v in views]
        found.append({
            'id':'mvc:'+Path(item['manifest']).parent.name,
            'title':item['dataset'], 'n':item['n'], 'd':'/'.join(map(str,dims)),
            'c':item['classes'], 'tier':f'{len(views)}-view',
            'kind':'prepared_multiview', 'category':'Prepared MVC / Synthetic MVC',
            'recommended_engine':item['recommended_engine'],
            'recommended_k':item['recommended_k'],
            'recommended_kernel_preset':item['recommended_kernel_preset'],
            'description':str(m.get('purpose') or 'Validated prepared multi-view dataset'),
            'ready':True, 'status':'ready', 'bundled':False,
            'views':views, 'labels':item['labels'],
            'manifest_path':item['manifest'], 'manifest':m,
        })
    return found


def catalog(out_root: str | Path = 'datasets', mv_root: str | Path = 'dataset_cache') -> list[dict]:
    rows = []
    for k, v in CATALOG.items():
        r = {'id': k, **v}
        m = _manifest_if_ready(k, out_root)
        r['ready'] = bool(m)
        r['bundled'] = v['kind'] == 'bundled'
        if m:
            r['view'] = m.get('view')
            r['labels'] = m.get('labels')
            r['manifest'] = m
        rows.append(r)
    rows.extend(discover_prepared_multiview(mv_root))
    return rows


def dataset_manifest(dataset_id: str, out_root: str | Path = 'datasets') -> dict | None:
    did = dataset_id.lower()
    if did not in CATALOG:
        raise KeyError(did)
    return _manifest_if_ready(did, out_root)



# ---------------------------------------------------------------------------
# Multi-view MATLAB cell-array convenience (v0.7.3)
# ---------------------------------------------------------------------------
def inspect_multiview_mat(path: str | Path, *, views_key: str = 'auto', labels_key: str = 'auto', aux_key: str = 'M') -> dict:
    """Inspect aligned MATLAB v5/v7.3 views without writing prepared data."""
    info = read_multiview_mat(path, views_key=views_key, labels_key=labels_key, aux_key=aux_key)
    info.pop('views')
    info.pop('labels')
    return info


def _write_mv_run_config(out: Path, manifest: dict, *, preset: str, k: int, search: str) -> Path:
    tag = 'rbf1' if preset.lower().startswith('rbf1') else ('rbf_self_tuned' if 'self' in preset.lower() else preset.lower())
    run_dir = out / f'run_{tag}'
    cfg = {
        'dataset_manifest': str((out / 'manifest.json').resolve()),
        'engine': 'scalable_sparse',
        'clusters': int(manifest['classes']),
        'auto_clusters_from_ground_truth': True,
        'k': int(k),
        'kernel_bank': {'mode': 'shared', 'preset': preset},
        'solver': {'inner_solver': 'auto'},
        'search': search,
        'ann_backend': 'auto',
        'support_mode': 'directed',
        'support_policy': 'ranking_grouped',
        'diffusion_power': 1,
        'seed': 0,
        'workspace': str((run_dir / 'workspace').resolve()),
        'run_dir': str(run_dir.resolve()),
        'out': str((run_dir / 'result.npz').resolve()),
        'evaluation': {'enabled': True},
    }
    cp = out / f'run_config_{tag}.json'
    cp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    return cp


def prepare_multiview_mat(
    path: str | Path,
    out_root: str | Path = 'dataset_cache',
    *,
    name: str | None = None,
    views_key: str = 'auto',
    labels_key: str = 'auto',
    aux_key: str = 'M',
    k: int = 10,
    search: str = 'exact',
    overwrite: bool = False,
) -> dict:
    """Convert a MATLAB cell-array multi-view dataset to mmap-ready NPY files.

    The generated directory contains one row-L2-normalized float32 NPY per
    view, int32 labels, ``manifest.json``, and ready-to-run RBF1/RBF-self-tuned/RBF7 configs.
    """
    src = Path(path).expanduser().resolve()
    dataset_name = str(name or src.stem).strip()
    if not dataset_name:
        raise ValueError('dataset name cannot be empty')
    out = Path(out_root).expanduser().resolve() / dataset_name
    manifest_path = out / 'manifest.json'
    if manifest_path.exists() and not overwrite:
        try:
            m = json.loads(manifest_path.read_text(encoding='utf-8'))
            if all(Path(v['path']).exists() for v in m.get('views', [])) and Path(m.get('labels', '')).exists():
                return m
        except Exception:
            pass
    source = read_multiview_mat(src, views_key=views_key, labels_key=labels_key,
                                aux_key=aux_key, load_views=True, require_labels=True)
    views = source['views']
    y = source['labels']
    n = source['n']
    out.mkdir(parents=True, exist_ok=True)
    classes = int(np.unique(y).size)
    ypath = out / 'labels.npy'
    np.save(ypath, y.astype(np.int32, copy=False))

    vrows = []
    for i, X in enumerate(views, start=1):
        vname = f'view{i}'
        target = out / f'{vname}.npy'
        from scipy import sparse
        if sparse.issparse(X):
            mm = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=X.shape)
            X = X.tocsr()
            for start in range(0, n, 4096):
                stop = min(n, start + 4096)
                block = X[start:stop].toarray().astype(np.float64, copy=False)
                norms = np.maximum(np.linalg.norm(block, axis=1), np.finfo(np.float64).eps)
                mm[start:stop] = (block / norms[:, None]).astype(np.float32)
            mm.flush(); del mm
        else:
            prep = prepare_float32_l2(np.asarray(X), name=vname, workspace=out, block_rows=65536, reuse_cache=False)
            if prep.path and Path(prep.path).exists():
                shutil.move(str(prep.path), str(target))
            else:
                mm = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=prep.array.shape)
                mm[:] = np.asarray(prep.array, dtype=np.float32)
                mm.flush(); del mm
        vrows.append({
            'name': vname, 'path': str(target.resolve()), 'n': n,
            'd': int(X.shape[1]), 'dtype': 'float32', 'l2_normalized': True, 'k': int(k),
        })

    manifest = {
        'schema': 'scsimplemkkm.multiview.v1',
        'dataset': dataset_name,
        'source_mat': str(src),
        'source_keys': source['keys'],
        'source_format': source['source_format'],
        'views_key': source['views_key'], 'labels_key': source['labels_key'],
        'aux_key': source['aux_key'],
        'aux_shape': source['aux_shape'],
        'n': n, 'classes': classes, 'view_count': len(vrows),
        'view_dims': [int(v['d']) for v in vrows],
        'views': vrows, 'labels': str(ypath.resolve()),
        'dtype': 'float32', 'row_l2_normalized': True,
        'recommended_engine': 'scalable_sparse', 'recommended_k': int(k),
        'recommended_kernel_preset': 'rbf_self_tuned',
        'single_kernel_rule': {
            # Backward-compatible global-center fields retained for v0.7.3 tools.
            'preset': 'rbf1', 'rbf_scale_alpha': 1.0,
            'global_center_formula': 'exp(-||xi-xj||^2 / mean_pairwise_sqdist_view)',
            # v0.7.4 recommended no-global-bandwidth single affinity.
            'recommended_preset': 'rbf_self_tuned',
            'local_scale_k': 7,
            'formula': 'exp(-||xi-xj||^2 / (sigma_i sigma_j)), sigma_i = distance to 7th neighbor',
            'provenance': 'Zelnik-Manor and Perona, Self-Tuning Spectral Clustering, NIPS 2004',
            'note': 'removes the global Gaussian bandwidth search; K=7 is fixed paper-wide and labels are not used',
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    c1 = _write_mv_run_config(out, manifest, preset='rbf1', k=k, search=search)
    cs = _write_mv_run_config(out, manifest, preset='rbf_self_tuned', k=k, search=search)
    c7 = _write_mv_run_config(out, manifest, preset='rbf7', k=k, search=search)
    manifest['config_rbf1'] = str(c1.resolve())
    manifest['config_rbf_self_tuned'] = str(cs.resolve())
    manifest['config_rbf7'] = str(c7.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def prepare_multiview_directory(
    directory: str | Path,
    out_root: str | Path = 'dataset_cache',
    *,
    pattern: str = '*.mat',
    views_key: str = 'auto',
    labels_key: str = 'auto',
    aux_key: str = 'M',
    k: int = 10,
    search: str = 'exact',
    overwrite: bool = False,
) -> dict:
    """Batch-convert a directory of common MVC MATLAB files.

    Each file is isolated: one malformed dataset is reported in ``errors`` and
    does not prevent the remaining datasets from being prepared.  Successfully
    prepared manifests are automatically discovered by ``catalog`` afterwards.
    """
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f'MVC MAT directory not found: {root}')
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f'No files matching {pattern!r} under {root}')
    prepared=[]; errors=[]
    for src in files:
        try:
            prepared.append(prepare_multiview_mat(
                src, out_root, name=src.stem, views_key=views_key,
                labels_key=labels_key, aux_key=aux_key, k=k, search=search,
                overwrite=overwrite,
            ))
        except Exception as e:
            errors.append({'file': str(src), 'error': f'{type(e).__name__}: {e}'})
    return {
        'batch': True, 'source_dir': str(root), 'pattern': pattern,
        'prepared_count': len(prepared), 'error_count': len(errors),
        'datasets': prepared, 'errors': errors,
    }


def _download(url: str, dst: Path, progress=None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'SC-SimpleMKKM/0.7.4'})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open('wb') as f:
        total = int(r.headers.get('Content-Length') or 0); done = 0
        while True:
            b = r.read(4 * 1024 * 1024)
            if not b: break
            f.write(b); done += len(b)
            if progress: progress(done, total)
    return dst


def _save_processed(out: Path, X: np.ndarray, y: np.ndarray, source: dict, *, l2: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / 'view1_raw.npy'
    mm = np.lib.format.open_memmap(raw_path, mode='w+', dtype=np.float32, shape=X.shape)
    mm[:] = np.asarray(X, dtype=np.float32); mm.flush(); del mm
    if l2:
        prep = prepare_float32_l2(np.load(raw_path, mmap_mode='r'), name='view1', workspace=out, block_rows=100000)
        final = Path(prep.path) if prep.path else raw_path
        target = out / 'view1.npy'
        if final.resolve() != target.resolve():
            shutil.move(str(final), str(target))
        raw_path.unlink(missing_ok=True)
    else:
        target = out / 'view1.npy'; raw_path.rename(target)
    np.save(out / 'labels.npy', np.asarray(y, dtype=np.int32))
    manifest = {
        'id': next((k for k,v in CATALOG.items() if v is source), None),
        'dataset': source['title'], 'source_url': source.get('url'), 'doi': source.get('doi'),
        'n': int(X.shape[0]), 'd': int(X.shape[1]), 'classes': int(np.unique(y).size),
        'view': str(target.resolve()), 'labels': str((out/'labels.npy').resolve()),
        'dtype': 'float32', 'l2_normalized': bool(l2), 'purpose': 'clustering benchmark / scale test',
        'ready': True, 'bundled': False,
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    cfg = {
        'clusters': int(np.unique(y).size), 'k': int(source.get('recommended_k', 10)),
        'engine': str(source.get('recommended_engine', 'scalable_sparse')),
        'views': [{'name':'view1','path':str(target.resolve()),'k':int(source.get('recommended_k', 10))}],
        'ground_truth': str((out/'labels.npy').resolve()), 'search':'ann','ann_backend':'auto',
        'support_mode':'directed','support_policy':'ranking_grouped','diffusion_power':1,
        'out':str((out/'run/result.npz').resolve()), 'run_dir':str((out/'run').resolve()), 'workspace':str((out/'run/workspace').resolve()),
    }
    (out/'run_config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def _extract_member(z: zipfile.ZipFile, pred) -> str:
    for n in z.namelist():
        if pred(n): return n
    raise FileNotFoundError('Expected dataset member not found in zip')


def prepare_dataset(dataset_id: str, out_root: str | Path, *, keep_download: bool = True) -> dict:
    did = dataset_id.lower()
    if did not in CATALOG:
        raise KeyError(f'unknown dataset: {dataset_id}')
    info = CATALOG[did]
    # Bundled laptop datasets require zero download/preprocessing.
    ready = _manifest_if_ready(did, out_root)
    if ready is not None:
        return ready
    out = Path(out_root)/did; out.mkdir(parents=True, exist_ok=True)
    zip_path = out / 'download.zip'
    if not zip_path.exists(): _download(info['url'], zip_path)
    if info['kind'] == 'uci_letter':
        with zipfile.ZipFile(zip_path) as z:
            name=_extract_member(z, lambda n:n.lower().endswith('letter-recognition.data'))
            rows=[]
            with io.TextIOWrapper(z.open(name),encoding='ascii') as t:
                for line in t:
                    parts=line.strip().split(',')
                    if parts: rows.append(parts)
        X=np.asarray([[float(v) for v in r[1:]] for r in rows],dtype=np.float32)
        y=np.asarray([ord(r[0].upper())-ord('A') for r in rows],dtype=np.int32)
    elif info['kind'] == 'uci_covtype':
        with zipfile.ZipFile(zip_path) as z:
            name = _extract_member(z, lambda n: n.lower().endswith('covtype.data.gz'))
            with z.open(name) as zg, gzip.GzipFile(fileobj=zg) as gf:
                a = np.loadtxt(gf, delimiter=',', dtype=np.float32)
        X, y = a[:,:-1], a[:,-1].astype(np.int32)-1
    elif info['kind'] == 'uci_poker':
        chunks=[]
        with zipfile.ZipFile(zip_path) as z:
            for suffix in ('poker-hand-training-true.data','poker-hand-testing.data'):
                name=_extract_member(z, lambda n,s=suffix:n.lower().endswith(s))
                with z.open(name) as f: chunks.append(np.loadtxt(f, delimiter=',', dtype=np.float32))
        a=np.vstack(chunks); X,y=a[:,:-1],a[:,-1].astype(np.int32)
    elif info['kind'] == 'uci_higgs':
        with zipfile.ZipFile(zip_path) as z:
            name=_extract_member(z, lambda n:n.lower().endswith('higgs.csv.gz'))
            Xp=out/'view1_raw.npy'; yp=out/'labels.npy'
            Xmm=np.lib.format.open_memmap(Xp,mode='w+',dtype=np.float32,shape=(info['n'],info['d']))
            ymm=np.lib.format.open_memmap(yp,mode='w+',dtype=np.int32,shape=(info['n'],))
            with z.open(name) as zg, gzip.GzipFile(fileobj=zg) as gf, io.TextIOWrapper(gf,encoding='ascii') as t:
                for i,line in enumerate(t):
                    vals=np.fromstring(line,sep=',',dtype=np.float32); ymm[i]=int(vals[0]); Xmm[i]=vals[1:]
            Xmm.flush(); ymm.flush(); del Xmm,ymm
        prep=prepare_float32_l2(np.load(Xp,mmap_mode='r'),name='view1',workspace=out,block_rows=100000)
        target=out/'view1.npy'; shutil.move(prep.path,str(target)); Xp.unlink(missing_ok=True)
        manifest={
            'id':did,'dataset':info['title'],'source_url':info['url'],'doi':info.get('doi'),
            'n':info['n'],'d':info['d'],'classes':2,'view':str(target.resolve()),'labels':str(yp.resolve()),
            'dtype':'float32','l2_normalized':True,'ready':True,'bundled':False,
        }
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        return manifest
    else:
        raise ValueError(info['kind'])
    manifest=_save_processed(out,X,y,info)
    manifest['id']=did
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if not keep_download: zip_path.unlink(missing_ok=True)
    return manifest


def generate_scale_tier(
    out_root: str | Path,
    n: int,
    d: int = 16,
    clusters: int = 10,
    seed: int = 20261003,
    *,
    dims: list[int] | None = None,
    noise: float = 0.30,
) -> dict:
    dims2 = [int(x) for x in (dims if dims is not None else [d])]
    if not dims2 or any(x < 2 for x in dims2):
        raise ValueError('synthetic view dimensions must all be >= 2')
    m = generate_multiview_gaussian(
        Path(out_root)/str(n), n, dims=dims2, n_clusters=clusters,
        noise=float(noise), seed=seed+n, block_rows=100000,
    )
    m.update({
        'dataset': f'ScaleSynth-MVC-{n}', 'classes': int(clusters),
        'view_count': len(dims2), 'view_dims': dims2,
        'recommended_engine': 'scalable_sparse', 'recommended_k': 10,
        'recommended_kernel_preset': 'rbf_self_tuned',
    })
    mp = Path(out_root)/str(n)/'manifest.json'
    mp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding='utf-8')
    return m


def main() -> int:
    ap=argparse.ArgumentParser(description='SC-SimpleMKKM one-click dataset manager')
    sub=ap.add_subparsers(dest='cmd',required=True)
    l=sub.add_parser('list'); l.add_argument('--out-root',default='datasets')
    p=sub.add_parser('prepare');p.add_argument('dataset',choices=sorted(CATALOG));p.add_argument('--out-root',default='datasets')
    g=sub.add_parser('scale');g.add_argument('--out-root',default='scale_data');g.add_argument('--sizes',default='3000,10000,50000,200000,1000000');g.add_argument('--dim',type=int,default=16);g.add_argument('--clusters',type=int,default=10)
    mi=sub.add_parser('inspect-mv-mat');mi.add_argument('path');mi.add_argument('--views-key',default='auto');mi.add_argument('--labels-key',default='auto');mi.add_argument('--aux-key',default='M')
    mp=sub.add_parser('prepare-mv-mat');mp.add_argument('path');mp.add_argument('--out-root',default='dataset_cache');mp.add_argument('--name',default=None);mp.add_argument('--views-key',default='auto');mp.add_argument('--labels-key',default='auto');mp.add_argument('--aux-key',default='M');mp.add_argument('--k',type=int,default=10);mp.add_argument('--search',choices=['exact','ann'],default='exact');mp.add_argument('--overwrite',action='store_true')
    a=ap.parse_args()
    if a.cmd=='list': print(json.dumps(catalog(a.out_root),indent=2,ensure_ascii=False)); return 0
    if a.cmd=='prepare': print(json.dumps(prepare_dataset(a.dataset,a.out_root),indent=2,ensure_ascii=False)); return 0
    if a.cmd=='inspect-mv-mat': print(json.dumps(inspect_multiview_mat(a.path,views_key=a.views_key,labels_key=a.labels_key,aux_key=a.aux_key),indent=2,ensure_ascii=False)); return 0
    if a.cmd=='prepare-mv-mat': print(json.dumps(prepare_multiview_mat(a.path,a.out_root,name=a.name,views_key=a.views_key,labels_key=a.labels_key,aux_key=a.aux_key,k=a.k,search=a.search,overwrite=a.overwrite),indent=2,ensure_ascii=False)); return 0
    for tok in a.sizes.split(','): generate_scale_tier(a.out_root,int(tok),a.dim,a.clusters)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
