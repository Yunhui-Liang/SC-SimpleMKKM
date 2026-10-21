from __future__ import annotations

"""Reproducible dataset preparation for the SC-SimpleMKKM paper workflow.

This module is intentionally independent of the Web UI.  It downloads public
sources, converts them into mmap-friendly multi-view NPY files, records source
provenance, and writes a common manifest consumed by ``scsimplemkkm run-mv``.

Three dataset roles are distinguished in every manifest:

* ``standard_mvc``: the source already provides multiple aligned feature views.
* ``derived_multiview``: semantically documented feature groups or deterministic
  transforms of one public source are used as views.  These are scale/robustness
  benchmarks and must not be described as canonical MVC datasets.
* ``synthetic_scale``: controlled generated views for extreme-scale systems tests.
"""

import argparse
import contextlib
import gzip
import hashlib
import io
import itertools
import json
import os
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .benchdata import generate_multiview_gaussian


PAPER_DATASETS: dict[str, dict] = {
    "mfeat_mvc": {
        "title": "UCI Multiple Features (MFeat)",
        "role": "standard_mvc", "n": 2000, "classes": 10, "views": 6,
        "source_url": "https://archive.ics.uci.edu/static/public/72/multiple%2Bfeatures.zip",
        "doi": "10.24432/C5HC70", "license": "CC BY 4.0",
        "description": "Six aligned handwritten-digit feature sets: Fourier, profile, Karhunen-Love, pixels, Zernike and morphology.",
    },
    "daily_sports_mvc": {
        "title": "UCI Daily and Sports Activities",
        "role": "standard_mvc", "n": 9120, "classes": 19, "views": 5,
        "source_url": "https://archive.ics.uci.edu/static/public/256/daily%2Band%2Bsports%2Bactivities.zip",
        "doi": "10.24432/C5C59F", "license": "CC BY 4.0",
        "description": "Five body-worn Xsens units; each 5-s segment becomes five aligned sensor views.",
    },
    "mnist_mvc": {
        "title": "MNIST derived multi-view",
        "role": "derived_multiview", "n": 70000, "classes": 10, "views": 3,
        "source_urls": {
            "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        },
        "description": "Deterministic image views: pixels, 2x2 pooled pixels, and row/column intensity profiles.",
    },
    "fashion_mnist_mvc": {
        "title": "Fashion-MNIST derived multi-view",
        "role": "derived_multiview", "n": 70000, "classes": 10, "views": 3,
        "source_urls": {
            "train_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
            "train_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
            "test_images": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        },
        "license": "MIT (repository); dataset citation per Fashion-MNIST project",
        "description": "Deterministic image views: pixels, 2x2 pooled pixels, and row/column intensity profiles.",
    },
    "covertype_mvc": {
        "title": "UCI Covertype feature-group multi-view",
        "role": "derived_multiview", "n": 581012, "classes": 7, "views": 3,
        "source_url": "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        "doi": "10.24432/C50K5N",
        "description": "Three documented feature groups: 10 continuous cartographic variables, 4 wilderness indicators, and 40 soil indicators.",
    },
    "poker_mvc": {
        "title": "UCI Poker Hand feature-group multi-view",
        "role": "derived_multiview", "n": 1025010, "classes": 10, "views": 2,
        "source_url": "https://archive.ics.uci.edu/static/public/158/poker%2Bhand.zip",
        "doi": "10.24432/C5KW38",
        "description": "Two aligned views formed from card suits and card ranks; labels are poker-hand classes.",
    },
    "susy_mvc": {
        "title": "UCI SUSY low/high-level multi-view",
        "role": "derived_multiview", "n": 5000000, "classes": 2, "views": 2,
        "source_url": "https://archive.ics.uci.edu/static/public/279/susy.zip",
        "doi": "10.24432/C54606", "license": "CC BY 4.0",
        "description": "Two documented feature groups: 8 low-level detector variables and 10 high-level derived variables.",
    },
    "higgs_mvc": {
        "title": "UCI HIGGS low/high-level multi-view",
        "role": "derived_multiview", "n": 11000000, "classes": 2, "views": 2,
        "source_url": "https://archive.ics.uci.edu/static/public/280/higgs.zip",
        "doi": "10.24432/C5V312", "license": "CC BY 4.0",
        "description": "Two documented feature groups: 21 low-level kinematic variables and 7 high-level derived variables.",
    },
}


# Additional canonical multi-view clustering datasets.  These sources use a
# common MATLAB convention (X: cell/list of N-by-D views, y/Y: labels).
# They are kept separate from derived feature-group scale benchmarks so paper
# tables can report benchmark provenance without ambiguity.
CANONICAL_MAT_DATASETS: dict[str, dict] = {
    "mvc_100leaves": {"title":"100Leaves","n":1600,"classes":100,"views":3,"filename":"100Leaves.mat","repo":"chuanbin"},
    "mvc_3sources": {"title":"3Sources","n":169,"classes":6,"views":3,"filename":"3Sources.mat","repo":"chuanbin"},
    "mvc_bbc4view": {"title":"BBC4view","n":685,"classes":5,"views":4,"filename":"BBC4view.mat","repo":"chuanbin"},
    "mvc_bbcsport": {"title":"BBCSport","n":544,"classes":5,"views":2,"filename":"BBCSport.mat","repo":"chuanbin"},
    "mvc_caltech101_7": {"title":"Caltech101-7","n":1474,"classes":7,"views":6,"filename":"Caltech101-7.mat","repo":"chuanbin"},
    "mvc_caltech101_20": {"title":"Caltech101-20","n":2386,"classes":20,"views":6,"filename":"Caltech101-20.mat","repo":"zskong"},
    "mvc_citeseer": {"title":"CiteSeer","n":3312,"classes":6,"views":2,"filename":"CiteSeer.mat","repo":"chuanbin"},
    "mvc_cora": {"title":"Cora","n":2708,"classes":7,"views":4,"filename":"Cora.mat","repo":"chuanbin"},
    "mvc_coil20": {"title":"COIL20","n":1440,"classes":20,"views":3,"filename":"COIL20.mat","repo":"zskong"},
    "mvc_flower17": {"title":"Flower17","n":1360,"classes":17,"views":7,"filename":"flower17.mat","repo":"zskong"},
    "mvc_handwritten": {"title":"Handwritten","n":2000,"classes":10,"views":6,"filename":"Handwritten.mat","repo":"chuanbin"},
    "mvc_mnist10k": {"title":"MNIST-10k MVC","n":10000,"classes":10,"views":3,"filename":"MNIST-10k.mat","repo":"chuanbin"},
    "mvc_mnist4": {"title":"MNIST-4 MVC","n":4000,"classes":4,"views":3,"filename":"MNIST-4.mat","repo":"chuanbin"},
    "mvc_msrc_v5": {"title":"MSRC-v5","n":210,"classes":7,"views":5,"filename":"MSRC-v5.mat","repo":"chuanbin"},
    "mvc_msrcv1": {"title":"MSRCV1","n":210,"classes":7,"views":6,"filename":"MSRCV1.mat","repo":"zskong"},
    "mvc_movies": {"title":"Movies","n":617,"classes":17,"views":2,"filename":"Movies.mat","repo":"chuanbin"},
    "mvc_nuswide": {"title":"NUS-WIDE subset","n":2400,"classes":12,"views":5,"filename":"NUS-WIDE.mat","repo":"chuanbin"},
    "mvc_outdoorscene": {"title":"OutdoorScene","n":2688,"classes":8,"views":4,"filename":"OutdoorScene.mat","repo":"chuanbin"},
    "mvc_prokaryotic": {"title":"Prokaryotic","n":551,"classes":4,"views":3,"filename":"Prokaryotic.mat","repo":"chuanbin"},
    "mvc_proteinfold": {"title":"ProteinFold","n":694,"classes":27,"views":12,"filename":"ProteinFold.mat","repo":"chuanbin"},
    "mvc_reuters1200": {"title":"Reuters-1200","n":1200,"classes":6,"views":5,"filename":"Reuters-1200.mat","repo":"chuanbin"},
    "mvc_reuters1500": {"title":"Reuters-1500","n":1500,"classes":6,"views":5,"filename":"Reuters-1500.mat","repo":"chuanbin"},
    "mvc_scene": {"title":"Scene (4-view)","n":2688,"classes":8,"views":4,"filename":"scene.mat","repo":"zskong"},
    "mvc_uci_digits": {"title":"UCI Digits MVC","n":2000,"classes":10,"views":3,"filename":"UCI.mat","repo":"chuanbin"},
    "mvc_webkb": {"title":"WebKB","n":1051,"classes":2,"views":2,"filename":"WebKB.mat","repo":"chuanbin"},
    "mvc_wikipedia": {"title":"Wikipedia","n":2866,"classes":10,"views":2,"filename":"Wikipedia.mat","repo":"chuanbin"},
    "mvc_wikipedia_test": {"title":"Wikipedia-test","n":693,"classes":10,"views":2,"filename":"Wikipedia-test.mat","repo":"chuanbin"},
    "mvc_yale": {"title":"Yale","n":165,"classes":15,"views":3,"filename":"Yale.mat","repo":"chuanbin"},
    "mvc_aloi100": {"title":"ALOI100","n":10800,"classes":100,"views":4,"filename":"ALOI100.mat","repo":"zskong"},
}

_REPO_BASES = {
    "chuanbin": "https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master/",
    "zskong": "https://raw.githubusercontent.com/zskong/Multi-view-data/main/data/",
}

for _rid, _meta in CANONICAL_MAT_DATASETS.items():
    _meta.update({
        "role":"standard_mvc",
        "source_url": _REPO_BASES[_meta["repo"]] + _meta["filename"],
        "source_repository": (
            "https://github.com/ChuanbinZhang/Multi-view-datasets" if _meta["repo"] == "chuanbin"
            else "https://github.com/zskong/Multi-view-data"
        ),
        "description":"Canonical aligned multi-view clustering benchmark distributed as MATLAB feature views.",
    })

PAPER_DATASETS.update(CANONICAL_MAT_DATASETS)
PAPER_DATASETS.update({
    "pamap2_mvc": {
        "title":"PAMAP2 three-IMU MVC", "role":"standard_mvc", "n":1942872, "classes":12, "views":3,
        "source_url":"https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip",
        "doi":"10.24432/C5NW2H", "license":"CC BY 4.0",
        "description":"Timestamp-level aligned hand/chest/ankle IMU views from the 12 UCI PAMAP2 protocol activities; optional recordings and transient label 0 are excluded, as are invalid orientation channels.",
    },
    "realdisp_mvc": {
        "title":"REALDISP nine-sensor MVC", "role":"standard_mvc", "n":None, "classes":33, "views":9,
        "source_url":"https://archive.ics.uci.edu/static/public/305/realdisp%2Bactivity%2Brecognition%2Bdataset.zip",
        "doi":"10.24432/C5GP6D", "license":"CC BY 4.0",
        "description":"Timestamp-level aligned nine-body-sensor views from REALDISP; label 0 is removed. This is a large raw multi-sensor benchmark and requires substantial SSD space.",
    },
})


def _sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Download with a best-effort HTTP Range resume path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite and dest.stat().st_size > 0:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() and not overwrite else 0
    if overwrite and part.exists():
        part.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "SC-SimpleMKKM-PaperLab/0.8.1"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status = getattr(r, "status", None)
            # If server ignored Range, restart rather than concatenate a full file.
            if start and status != 206:
                start = 0
                part.unlink(missing_ok=True)
            mode = "ab" if start else "wb"
            with part.open(mode) as f:
                copied = start
                while True:
                    b = r.read(4 << 20)
                    if not b:
                        break
                    f.write(b); copied += len(b)
                    if copied and copied % (256 << 20) < (4 << 20):
                        print(f"downloaded {copied / 2**30:.2f} GiB: {dest.name}", flush=True)
    except Exception:
        # Keep .part so a later unattended retry can resume.
        raise
    part.replace(dest)
    return dest


def _normalize_block(X: np.ndarray) -> np.ndarray:
    Y = np.asarray(X, dtype=np.float32)
    den = np.linalg.norm(Y, axis=1, keepdims=True)
    np.maximum(den, np.float32(1e-12), out=den)
    return Y / den


def _write_manifest(root: Path, recipe_id: str, title: str, role: str,
                    views: list[dict], labels: Path, classes: int,
                    provenance: dict, purpose: str) -> dict:
    n = int(np.load(labels, mmap_mode="r").shape[0])
    m = {
        "schema": "scsimplemkkm.paperdata.v1",
        "id": recipe_id, "dataset": title, "benchmark_role": role,
        "n": n, "classes": int(classes), "view_count": len(views),
        "view_dims": [int(v["d"]) for v in views], "views": views,
        "labels": str(labels.resolve()), "dtype": "float32",
        "row_l2_normalized": True, "recommended_engine": "scalable_sparse",
        "recommended_k": 10, "recommended_kernel_preset": "rbf_self_tuned",
        "recommended_K_scale": 7, "purpose": purpose,
        "provenance": provenance,
    }
    (root / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m


def _write_views_from_dense(root: Path, arrays: list[tuple[str, np.ndarray]], labels: np.ndarray,
                            recipe_id: str, meta: dict, purpose: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    yp = root / "labels.npy"; np.save(yp, np.asarray(labels, dtype=np.int32))
    views = []
    for name, X in arrays:
        xp = root / f"{name}.npy"
        mm = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32, shape=X.shape)
        step = max(1024, min(65536, int(256 * 1024**2 / max(4 * X.shape[1], 1))))
        for s in range(0, X.shape[0], step):
            e = min(X.shape[0], s + step)
            mm[s:e] = _normalize_block(X[s:e])
        mm.flush(); del mm
        views.append({"name": name, "path": str(xp.resolve()), "n": int(X.shape[0]),
                      "d": int(X.shape[1]), "dtype": "float32", "l2_normalized": True, "k": 10})
    prov = {k: v for k, v in meta.items() if k in {"source_url", "source_urls", "doi", "license", "description", "source_sha256"}}
    return _write_manifest(root, recipe_id, meta["title"], meta["role"], views, yp,
                           int(meta["classes"]), prov, purpose)


def prepare_mfeat(root: Path, *, overwrite: bool = False) -> dict:
    meta = PAPER_DATASETS["mfeat_mvc"]; root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() and not overwrite:
        return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    zpath = _download(meta["source_url"], root / "source.zip", overwrite=overwrite)
    names = ["mfeat-fou", "mfeat-fac", "mfeat-kar", "mfeat-pix", "mfeat-zer", "mfeat-mor"]
    arrays = []
    with zipfile.ZipFile(zpath) as z:
        low = {Path(n).name.lower(): n for n in z.namelist()}
        for name in names:
            key = name.lower()
            if key not in low:
                # Some UCI distributions include mfeat.tar instead of all files at zip root.
                tar_member = next((n for n in z.namelist() if Path(n).name.lower() == "mfeat.tar"), None)
                if tar_member is None:
                    raise FileNotFoundError(f"{name} not found in UCI archive")
                with z.open(tar_member) as tf, tarfile.open(fileobj=io.BytesIO(tf.read())) as tar:
                    member = next((m for m in tar.getmembers() if Path(m.name).name.lower() == key), None)
                    if member is None: raise FileNotFoundError(name)
                    X = np.loadtxt(tar.extractfile(member), dtype=np.float32)
            else:
                X = np.loadtxt(z.open(low[key]), dtype=np.float32)
            arrays.append((name.replace("mfeat-", "view_"), X))
    y = np.repeat(np.arange(10, dtype=np.int32), 200)
    meta = dict(meta); meta["source_sha256"] = _sha256(zpath)
    return _write_views_from_dense(root, arrays, y, "mfeat_mvc", meta,
                                   "standard real MVC quality benchmark")


def prepare_daily_sports(root: Path, *, overwrite: bool = False) -> dict:
    meta = PAPER_DATASETS["daily_sports_mvc"]; root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() and not overwrite:
        return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    zpath = _download(meta["source_url"], root / "source.zip", overwrite=overwrite)
    n = 9120; dims = [125 * 9] * 5
    y = np.lib.format.open_memmap(root / "labels.npy", mode="w+", dtype=np.int32, shape=(n,))
    xs = [np.lib.format.open_memmap(root / f"view{i+1}.npy", mode="w+", dtype=np.float32, shape=(n, d)) for i, d in enumerate(dims)]
    row = 0
    with zipfile.ZipFile(zpath) as z:
        # Canonical files are data/aXX/pY/sZZ.txt.  Sort numerically by path.
        members = [n0 for n0 in z.namelist() if n0.lower().endswith(".txt") and "/data/" in ("/" + n0.lower())]
        if not members:
            members = [n0 for n0 in z.namelist() if n0.lower().endswith(".txt") and "/a" in ("/" + n0.lower())]
        def key(p: str):
            parts = p.replace("\\", "/").split("/")
            a = next((int(x[1:]) for x in parts if len(x) == 3 and x.lower().startswith("a") and x[1:].isdigit()), 0)
            subj = next((int(x[1:]) for x in parts if x.lower().startswith("p") and x[1:].isdigit()), 0)
            seg = next((int(Path(x).stem[1:]) for x in parts if Path(x).stem.lower().startswith("s") and Path(x).stem[1:].isdigit()), 0)
            return (a, subj, seg)
        members.sort(key=key)
        if len(members) != n:
            raise ValueError(f"expected {n} segment files, found {len(members)}")
        for member in members:
            a, _, _ = key(member)
            raw = np.loadtxt(z.open(member), delimiter=",", dtype=np.float32)
            if raw.shape != (125, 45):
                raise ValueError(f"unexpected {member} shape {raw.shape}")
            for iv in range(5):
                block = raw[:, iv*9:(iv+1)*9].reshape(1, -1)
                xs[iv][row] = _normalize_block(block)[0]
            y[row] = a - 1; row += 1
            if row % 1000 == 0: print(f"DailySports prepared {row:,}/{n:,}", flush=True)
    y.flush(); del y
    views=[]
    for i, x in enumerate(xs):
        x.flush(); del x
        xp=root/f"view{i+1}.npy"; views.append({"name":f"sensor{i+1}","path":str(xp.resolve()),"n":n,"d":dims[i],"dtype":"float32","l2_normalized":True,"k":10})
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"license":meta.get("license"),"description":meta["description"],"source_sha256":_sha256(zpath)}
    return _write_manifest(root,"daily_sports_mvc",meta["title"],meta["role"],views,root/"labels.npy",19,prov,
                           "standard real multi-sensor MVC quality benchmark")


def _read_idx_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12)); data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(n, r, c)
        if magic == 2049:
            n = struct.unpack(">I", f.read(4))[0]; data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(n)
        raise ValueError(f"unknown IDX magic={magic}")


def prepare_idx_images(root: Path, recipe_id: str, *, overwrite: bool = False) -> dict:
    meta = PAPER_DATASETS[recipe_id]; root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() and not overwrite:
        return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files={}
    for key,url in meta["source_urls"].items(): files[key]=_download(url,root/f"{key}.gz",overwrite=overwrite)
    Xi=np.concatenate([_read_idx_gz(files["train_images"]),_read_idx_gz(files["test_images"])],axis=0)
    y=np.concatenate([_read_idx_gz(files["train_labels"]),_read_idx_gz(files["test_labels"])],axis=0).astype(np.int32)
    X=Xi.astype(np.float32)/np.float32(255.0)
    pooled=X.reshape(X.shape[0],14,2,14,2).mean(axis=(2,4))
    profiles=np.concatenate([X.sum(axis=1),X.sum(axis=2)],axis=1)
    arrays=[("pixels",X.reshape(X.shape[0],-1)),("pool2x2",pooled.reshape(X.shape[0],-1)),("rowcol_profiles",profiles)]
    prov={"source_urls":meta["source_urls"],"description":meta["description"]}
    if meta.get("license"): prov["license"]=meta["license"]
    meta2=dict(meta); meta2.update(prov)
    return _write_views_from_dense(root,arrays,y,recipe_id,meta2,"derived multi-view quality/medium-scale benchmark")


def _extract_zip_member(zip_path: Path, suffix: str) -> tuple[zipfile.ZipFile, str]:
    z=zipfile.ZipFile(zip_path)
    member=next((n for n in z.namelist() if n.lower().endswith(suffix.lower())),None)
    if member is None:
        z.close(); raise FileNotFoundError(f"{suffix} not found in {zip_path}")
    return z,member


def _stream_numeric_rows(text_stream, total_rows: int, feature_count: int,
                         view_cols: list[list[int]], out_root: Path,
                         *, label_index: int, label_shift: int = 0,
                         block_lines: int = 50000, delimiter: str = ",") -> tuple[list[dict], Path]:
    yp=out_root/"labels.npy"; ymm=np.lib.format.open_memmap(yp,mode="w+",dtype=np.int32,shape=(total_rows,))
    vmm=[]; views=[]
    for i,cols in enumerate(view_cols,1):
        xp=out_root/f"view{i}.npy"; mm=np.lib.format.open_memmap(xp,mode="w+",dtype=np.float32,shape=(total_rows,len(cols)))
        vmm.append(mm); views.append({"name":f"view{i}","path":str(xp.resolve()),"n":total_rows,"d":len(cols),"dtype":"float32","l2_normalized":True,"k":10})
    row=0; width=feature_count+1
    while row < total_rows:
        lines=list(itertools.islice(text_stream,block_lines))
        if not lines: break
        a=np.loadtxt(io.StringIO("".join(lines)),delimiter=delimiter,dtype=np.float32,ndmin=2)
        if a.shape[1] != width: raise ValueError(f"expected {width} columns, got {a.shape}")
        b=a.shape[0]; labels=a[:,label_index].astype(np.int32)+int(label_shift); ymm[row:row+b]=labels
        feat=np.delete(a,label_index,axis=1)
        for mm,cols in zip(vmm,view_cols): mm[row:row+b]=_normalize_block(feat[:,cols])
        row+=b
        if row % max(block_lines*10,1) < block_lines: print(f"prepared {row:,}/{total_rows:,}",flush=True)
    if row != total_rows: raise ValueError(f"row count mismatch: expected {total_rows}, got {row}")
    ymm.flush(); del ymm
    for mm in vmm: mm.flush()
    return views,yp


def prepare_covertype(root: Path, *, overwrite: bool=False) -> dict:
    meta=PAPER_DATASETS["covertype_mvc"]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite: return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    zp=_download(meta["source_url"],root/"source.zip",overwrite=overwrite); z,m=_extract_zip_member(zp,"covtype.data.gz")
    try:
        with z.open(m) as raw, gzip.GzipFile(fileobj=raw) as gf, io.TextIOWrapper(gf,encoding="ascii") as txt:
            # label is last column, hence index 54 in the 55-column row.
            views,yp=_stream_numeric_rows(txt,581012,54,[list(range(0,10)),list(range(10,14)),list(range(14,54))],root,label_index=54,label_shift=-1,block_lines=50000)
    finally: z.close()
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"description":meta["description"],"source_sha256":_sha256(zp)}
    return _write_manifest(root,"covertype_mvc",meta["title"],meta["role"],views,yp,7,prov,"derived feature-group multi-view scale benchmark")


def prepare_poker(root: Path, *, overwrite: bool=False) -> dict:
    meta=PAPER_DATASETS["poker_mvc"]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite: return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    zp=_download(meta["source_url"],root/"source.zip",overwrite=overwrite)
    n=1025010; yp=root/"labels.npy"; ymm=np.lib.format.open_memmap(yp,mode="w+",dtype=np.int32,shape=(n,))
    cols=[list(range(0,10,2)),list(range(1,10,2))]
    vmm=[np.lib.format.open_memmap(root/f"view{i+1}.npy",mode="w+",dtype=np.float32,shape=(n,5)) for i in range(2)]
    off=0
    with zipfile.ZipFile(zp) as z:
        members=[]
        for suffix in ["poker-hand-training-true.data","poker-hand-testing.data"]:
            m=next(n0 for n0 in z.namelist() if n0.lower().endswith(suffix)); members.append(m)
        for m in members:
            with io.TextIOWrapper(z.open(m),encoding="ascii") as txt:
                while True:
                    lines=list(itertools.islice(txt,50000))
                    if not lines: break
                    a=np.loadtxt(io.StringIO("".join(lines)),delimiter=",",dtype=np.float32,ndmin=2)
                    b=a.shape[0]; ymm[off:off+b]=a[:,-1].astype(np.int32)
                    feat=a[:,:10]
                    for mm,c in zip(vmm,cols): mm[off:off+b]=_normalize_block(feat[:,c])
                    off+=b
    if off!=n: raise ValueError(f"Poker expected {n} rows, got {off}")
    ymm.flush(); del ymm
    views=[]
    for i,mm in enumerate(vmm): mm.flush(); views.append({"name":["suits","ranks"][i],"path":str((root/f"view{i+1}.npy").resolve()),"n":n,"d":5,"dtype":"float32","l2_normalized":True,"k":10})
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"description":meta["description"],"source_sha256":_sha256(zp)}
    return _write_manifest(root,"poker_mvc",meta["title"],meta["role"],views,yp,10,prov,"derived feature-group multi-view million-scale benchmark")


def _prepare_physics(root: Path, recipe_id: str, feature_count: int, splits: list[list[int]], member_suffix: str, *, overwrite=False) -> dict:
    meta=PAPER_DATASETS[recipe_id]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite: return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    zp=_download(meta["source_url"],root/"source.zip",overwrite=overwrite); z,m=_extract_zip_member(zp,member_suffix)
    try:
        with z.open(m) as raw, gzip.GzipFile(fileobj=raw) as gf, io.TextIOWrapper(gf,encoding="ascii") as txt:
            views,yp=_stream_numeric_rows(txt,int(meta["n"]),feature_count,splits,root,label_index=0,label_shift=0,block_lines=50000)
    finally: z.close()
    names = ["low_level","high_level"]
    for v,nm in zip(views,names): v["name"]=nm
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"license":meta.get("license"),"description":meta["description"],"source_sha256":_sha256(zp)}
    return _write_manifest(root,recipe_id,meta["title"],meta["role"],views,yp,2,prov,"derived documented feature-group extreme-scale benchmark")


def prepare_susy(root: Path, *, overwrite=False) -> dict:
    return _prepare_physics(root,"susy_mvc",18,[list(range(0,8)),list(range(8,18))],"susy.csv.gz",overwrite=overwrite)


def prepare_higgs(root: Path, *, overwrite=False) -> dict:
    return _prepare_physics(root,"higgs_mvc",28,[list(range(0,21)),list(range(21,28))],"higgs.csv.gz",overwrite=overwrite)



def _labels_zero_based(y: np.ndarray) -> np.ndarray:
    a=np.asarray(y).reshape(-1)
    if a.dtype.kind in "OUS":
        _, inv=np.unique(a.astype(str), return_inverse=True); return inv.astype(np.int32)
    vals=np.asarray(a)
    uniq=np.unique(vals)
    mp={v:i for i,v in enumerate(uniq.tolist())}
    return np.asarray([mp[v] for v in vals.tolist()],dtype=np.int32)


def _mat_views_and_labels(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    import scipy.sparse as sp
    from .mat_import import read_multiview_mat

    source=read_multiview_mat(path,load_views=True,require_labels=True)
    labels=_labels_zero_based(np.asarray(source['labels']))
    views=[np.asarray(v.toarray() if sp.issparse(v) else v,dtype=np.float32) for v in source['views']]
    if len(views)<2: raise ValueError("standard MVC recipe resolved to fewer than two views")
    return views,labels


def prepare_canonical_mat(root: Path, recipe_id: str, *, overwrite: bool=False) -> dict:
    meta=PAPER_DATASETS[recipe_id]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite:
        return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    src=_download(meta["source_url"],root/meta["filename"],overwrite=overwrite)
    views,y=_mat_views_and_labels(src)
    arrays=[(f"view{i+1}",a) for i,a in enumerate(views)]
    meta2=dict(meta); meta2["source_sha256"]=_sha256(src)
    m=_write_views_from_dense(root,arrays,y,recipe_id,meta2,"canonical real multi-view clustering benchmark")
    # Validate published metadata rather than silently trusting it.
    if meta.get("n") and int(meta["n"])!=int(m["n"]): m.setdefault("warnings",[]).append(f"catalog n={meta['n']} but source n={m['n']}")
    if meta.get("views") and int(meta["views"])!=int(m["view_count"]): m.setdefault("warnings",[]).append(f"catalog views={meta['views']} but source views={m['view_count']}")
    (root/"manifest.json").write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding="utf-8")
    return m


def _iter_text_blocks(stream, block_lines=50000):
    while True:
        lines=list(itertools.islice(stream,block_lines))
        if not lines: break
        yield lines


def prepare_pamap2(root: Path, *, overwrite: bool=False) -> dict:
    meta=PAPER_DATASETS["pamap2_mvc"]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite: return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    zp=_download(meta["source_url"],root/"source.zip",overwrite=overwrite)
    with contextlib.ExitStack() as stack:
        z=stack.enter_context(zipfile.ZipFile(zp))
        members=sorted(n for n in z.namelist() if "/protocol/subject" in n.replace("\\", "/").lower() and n.lower().endswith(".dat"))
        if not members:
            nested=[n for n in z.namelist() if n.lower().endswith(".zip") and "pamap2" in n.lower()]
            if len(nested)!=1:
                raise ValueError("PAMAP2 Protocol/*.dat files not found in outer or nested archive")
            temp=stack.enter_context(tempfile.SpooledTemporaryFile(max_size=32 << 20))
            with z.open(nested[0]) as source:
                shutil.copyfileobj(source,temp,length=8 << 20)
            temp.seek(0)
            z=stack.enter_context(zipfile.ZipFile(temp))
            members=sorted(n for n in z.namelist() if "/protocol/subject" in n.replace("\\", "/").lower() and n.lower().endswith(".dat"))
        if not members: raise ValueError("PAMAP2 Protocol/*.dat files not found in archive")
        count=0
        for m in members:
            with io.TextIOWrapper(z.open(m),encoding="ascii",errors="replace") as f:
                for lines in _iter_text_blocks(f):
                    a=np.loadtxt(io.StringIO("".join(lines)),dtype=np.float32,ndmin=2)
                    count += int(np.count_nonzero(a[:,1] != 0))
        yp=root/"labels.npy"; ymm=np.lib.format.open_memmap(yp,mode="w+",dtype=np.int32,shape=(count,))
        vcols=[list(range(3,16)),list(range(20,33)),list(range(37,50))]
        names=["hand_imu","chest_imu","ankle_imu"]
        vmm=[np.lib.format.open_memmap(root/f"view{i+1}.npy",mode="w+",dtype=np.float32,shape=(count,13)) for i in range(3)]
        off=0
        labels_seen=[]
        for m in members:
            with io.TextIOWrapper(z.open(m),encoding="ascii",errors="replace") as f:
                for lines in _iter_text_blocks(f):
                    a=np.loadtxt(io.StringIO("".join(lines)),dtype=np.float32,ndmin=2)
                    mask=a[:,1]!=0; a=a[mask]
                    if not len(a): continue
                    b=len(a); rawy=a[:,1].astype(np.int32); labels_seen.extend(np.unique(rawy).tolist())
                    ymm[off:off+b]=rawy
                    for mm,cols in zip(vmm,vcols):
                        x=np.nan_to_num(a[:,cols],nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32,copy=False)
                        mm[off:off+b]=_normalize_block(x)
                    off+=b
        # remap non-contiguous activity IDs to 0..C-1 in-place
        vals=np.unique(np.asarray(ymm)); mp={int(v):i for i,v in enumerate(vals.tolist())}
        step=250000
        for st in range(0,count,step):
            ed=min(count,st+step); block=np.asarray(ymm[st:ed]); ymm[st:ed]=np.asarray([mp[int(v)] for v in block],dtype=np.int32)
        ymm.flush(); del ymm
        views=[]
        for i,mm in enumerate(vmm): mm.flush(); del mm; views.append({"name":names[i],"path":str((root/f"view{i+1}.npy").resolve()),"n":count,"d":13,"dtype":"float32","l2_normalized":True,"k":10})
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"license":meta["license"],"description":meta["description"],"source_sha256":_sha256(zp),"preprocessing":"Protocol only; activity 0 removed; per-IMU orientation channels excluded; NaN/Inf -> 0; row L2 normalization."}
    return _write_manifest(root,"pamap2_mvc",meta["title"],meta["role"],views,yp,len(vals),prov,"large real multi-sensor MVC benchmark")


def prepare_realdisp(root: Path, *, overwrite: bool=False) -> dict:
    meta=PAPER_DATASETS["realdisp_mvc"]; root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists() and not overwrite: return json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    zp=_download(meta["source_url"],root/"source.zip",overwrite=overwrite)
    with zipfile.ZipFile(zp) as z:
        members=sorted(n for n in z.namelist() if n.lower().endswith(".log"))
        if not members: raise ValueError("REALDISP .log files not found in archive")
        count=0
        for m in members:
            with io.TextIOWrapper(z.open(m),encoding="ascii",errors="replace") as f:
                for lines in _iter_text_blocks(f,20000):
                    a=np.loadtxt(io.StringIO("".join(lines)),dtype=np.float32,ndmin=2)
                    if a.shape[1] < 120: raise ValueError(f"REALDISP row width {a.shape[1]} < 120 in {m}")
                    count += int(np.count_nonzero(a[:,119] != 0))
        yp=root/"labels.npy"; ymm=np.lib.format.open_memmap(yp,mode="w+",dtype=np.int32,shape=(count,))
        vmm=[np.lib.format.open_memmap(root/f"view{i+1}.npy",mode="w+",dtype=np.float32,shape=(count,13)) for i in range(9)]
        off=0
        for m in members:
            with io.TextIOWrapper(z.open(m),encoding="ascii",errors="replace") as f:
                for lines in _iter_text_blocks(f,20000):
                    a=np.loadtxt(io.StringIO("".join(lines)),dtype=np.float32,ndmin=2); a=a[a[:,119]!=0]
                    if not len(a): continue
                    b=len(a); ymm[off:off+b]=a[:,119].astype(np.int32)
                    for i,mm in enumerate(vmm):
                        x=np.nan_to_num(a[:,2+i*13:2+(i+1)*13],nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32,copy=False)
                        mm[off:off+b]=_normalize_block(x)
                    off+=b
        vals=np.unique(np.asarray(ymm)); mp={int(v):i for i,v in enumerate(vals.tolist())}; step=250000
        for st in range(0,count,step):
            ed=min(count,st+step); block=np.asarray(ymm[st:ed]); ymm[st:ed]=np.asarray([mp[int(v)] for v in block],dtype=np.int32)
        ymm.flush(); del ymm
        views=[]
        for i,mm in enumerate(vmm): mm.flush(); del mm; views.append({"name":f"body_sensor_{i+1}","path":str((root/f"view{i+1}.npy").resolve()),"n":count,"d":13,"dtype":"float32","l2_normalized":True,"k":10})
    prov={"source_url":meta["source_url"],"doi":meta["doi"],"license":meta["license"],"description":meta["description"],"source_sha256":_sha256(zp),"preprocessing":"All log scenarios; label 0 removed; nine 13-channel body-sensor views; NaN/Inf -> 0; row L2 normalization."}
    return _write_manifest(root,"realdisp_mvc",meta["title"],meta["role"],views,yp,len(vals),prov,"large real nine-view multi-sensor MVC benchmark")

def prepare_recipe(recipe_id: str, out_root: str | Path="paper_data", *, overwrite: bool=False) -> dict:
    rid=recipe_id.lower(); root=Path(out_root).expanduser().resolve()/rid
    funcs={
        "mfeat_mvc":prepare_mfeat,
        "daily_sports_mvc":prepare_daily_sports,
        "pamap2_mvc":prepare_pamap2,
        "realdisp_mvc":prepare_realdisp,
        "mnist_mvc":lambda r,overwrite=False:prepare_idx_images(r,"mnist_mvc",overwrite=overwrite),
        "fashion_mnist_mvc":lambda r,overwrite=False:prepare_idx_images(r,"fashion_mnist_mvc",overwrite=overwrite),
        "covertype_mvc":prepare_covertype,
        "poker_mvc":prepare_poker,
        "susy_mvc":prepare_susy,
        "higgs_mvc":prepare_higgs,
    }
    for _rid in CANONICAL_MAT_DATASETS:
        funcs[_rid]=(lambda rr, overwrite=False, _rid=_rid: prepare_canonical_mat(rr,_rid,overwrite=overwrite))
    if rid.startswith("scalesynth_"):
        token=rid.split("_",1)[1].lower(); mult={"k":1000,"m":1_000_000}
        if token[-1] in mult: n=int(float(token[:-1])*mult[token[-1]])
        else: n=int(token)
        m=generate_multiview_gaussian(root,n,dims=[32,24,16],n_clusters=10,noise=0.30,seed=20261006+n,block_rows=100000)
        m["id"]=rid; m["benchmark_role"]="synthetic_scale"; m["recommended_k"]=10; m["recommended_K_scale"]=7
        m["provenance"]={"generator":"scsimplemkkm.paperdata","description":"controlled synthetic multi-view Gaussian benchmark"}
        (root/"manifest.json").write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding="utf-8")
        return m
    if rid not in funcs: raise KeyError(f"unknown paper dataset recipe: {recipe_id}")
    return funcs[rid](root,overwrite=overwrite)


def list_recipes() -> list[dict]:
    return [{"id":k,**v} for k,v in PAPER_DATASETS.items()] + [
        {"id":"scalesynth_1m","title":"ScaleSynth MVC 1M","role":"synthetic_scale","n":1_000_000,"views":3,"classes":10},
        {"id":"scalesynth_5m","title":"ScaleSynth MVC 5M","role":"synthetic_scale","n":5_000_000,"views":3,"classes":10},
        {"id":"scalesynth_10m","title":"ScaleSynth MVC 10M","role":"synthetic_scale","n":10_000_000,"views":3,"classes":10},
    ]


def main() -> int:
    ap=argparse.ArgumentParser(description="SC-SimpleMKKM publication dataset preparation")
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("list")
    p=sub.add_parser("prepare"); p.add_argument("dataset"); p.add_argument("--out-root",default="paper_data"); p.add_argument("--overwrite",action="store_true")
    pa=sub.add_parser("prepare-all"); pa.add_argument("datasets",nargs="*"); pa.add_argument("--out-root",default="paper_data"); pa.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    if a.cmd=="list": print(json.dumps(list_recipes(),indent=2,ensure_ascii=False)); return 0
    ids=[a.dataset] if a.cmd=="prepare" else (a.datasets or [x["id"] for x in list_recipes()])
    results=[]
    for rid in ids:
        try: results.append({"id":rid,"ok":True,"manifest":prepare_recipe(rid,a.out_root,overwrite=a.overwrite)})
        except Exception as e:
            results.append({"id":rid,"ok":False,"error":f"{type(e).__name__}: {e}"}); print(f"FAILED {rid}: {e}",flush=True)
    print(json.dumps(results,indent=2,ensure_ascii=False))
    return 0 if all(x["ok"] for x in results) else 2


if __name__=="__main__":
    raise SystemExit(main())
