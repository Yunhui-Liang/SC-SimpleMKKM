from __future__ import annotations

"""Read aligned MATLAB multi-view cells from v5 and v7.3 MAT files."""

from pathlib import Path
import hashlib

import numpy as np
from scipy import sparse


VIEW_KEYS = ("X", "Xs", "views", "data", "fea")
LABEL_KEYS = ("Y", "y", "labels", "label", "gt", "truth")


def _choose_key(keys: list[str], requested: str, candidates: tuple[str, ...], role: str, *, required: bool) -> str | None:
    value = str(requested or "auto").strip()
    if value.lower() != "auto":
        if value not in keys:
            raise KeyError(f"{role} variable {value!r} not found in MAT file; keys={keys}. Use 'auto' or enter the variable name.")
        return value
    matches = [key for key in candidates if key in keys]
    if len(matches) > 1:
        raise ValueError(f"Several {role} variables found: {matches}. Enter the intended variable name.")
    if matches:
        return matches[0]
    if required:
        raise KeyError(f"No recognized {role} variable in MAT file; keys={keys}. Enter the variable name explicitly.")
    return None


def _aligned_view(view, n: int | None, name: str):
    if sparse.issparse(view):
        shape = view.shape
        dtype = view.dtype
    else:
        view = np.asarray(view)
        shape = view.shape
        dtype = view.dtype
    if len(shape) != 2:
        raise ValueError(f"{name} must be a 2D feature matrix; got shape={shape}")
    if np.dtype(dtype).kind not in "biufc":
        raise ValueError(f"{name} must be numeric; got dtype={dtype}")
    if n is not None and shape[0] != n:
        if shape[1] == n:
            view = view.T
        else:
            raise ValueError(f"{name} shape {shape} does not align with {n} labels")
    return view


def _hdf_shape(node) -> tuple[int, int]:
    import h5py

    if isinstance(node, h5py.Group):
        if "MATLAB_sparse" not in node.attrs or not {"data", "ir", "jc"} <= set(node.keys()):
            raise ValueError(f"Unsupported MATLAB v7.3 view group: {node.name}")
        return int(node.attrs["MATLAB_sparse"]), int(node["jc"].shape[0] - 1)
    if len(node.shape) != 2:
        raise ValueError(f"MATLAB v7.3 view {node.name} must be 2D; got {node.shape}")
    # MATLAB v7.3 stores ordinary arrays with axes reversed in HDF5.
    return int(node.shape[1]), int(node.shape[0])


def _hdf_view(node):
    import h5py

    if isinstance(node, h5py.Group):
        rows, cols = _hdf_shape(node)
        return sparse.csc_matrix(
            (np.asarray(node["data"]), np.asarray(node["ir"], dtype=np.int64),
             np.asarray(node["jc"], dtype=np.int64)),
            shape=(rows, cols),
        )
    return np.asarray(node).T


def read_multiview_mat(
    path: str | Path,
    *,
    views_key: str = "auto",
    labels_key: str = "auto",
    aux_key: str = "M",
    load_views: bool = False,
    require_labels: bool = False,
) -> dict:
    """Return metadata and, when requested, aligned numeric views and labels.

    v7.3 inspection reads only HDF5 references and shapes. Preparing a dataset
    loads the views and lets the caller normalize them in blocks.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open("rb") as stream:
        is_v73 = stream.read(10) == b"MATLAB 7.3"

    if is_v73:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("MATLAB v7.3 import requires h5py; reinstall the Windows runtime.") from exc
        with h5py.File(p, "r") as obj:
            keys = sorted(key for key in obj if not key.startswith("#"))
            vk = _choose_key(keys, views_key, VIEW_KEYS, "view", required=True)
            lk = _choose_key(keys, labels_key, LABEL_KEYS, "label", required=require_labels)
            cell = obj[vk]
            if not isinstance(cell, h5py.Dataset) or cell.dtype.kind != "O":
                raise ValueError(f"{vk} must be a MATLAB cell array of feature matrices")
            refs = np.asarray(cell).ravel(order="F")
            if not len(refs):
                raise ValueError(f"{vk} contains no views")
            labels = np.asarray(obj[lk]).reshape(-1) if lk else None
            n = int(labels.size) if labels is not None else None
            views = []
            shapes = []
            dtypes = []
            for i, ref in enumerate(refs, 1):
                node = obj[ref]
                shape = _hdf_shape(node)
                if n is not None and shape[0] != n and shape[1] != n:
                    raise ValueError(f"view{i} shape {shape} does not align with {n} labels")
                if load_views:
                    view = _aligned_view(_hdf_view(node), n, f"view{i}")
                    views.append(view)
                    shape = view.shape
                elif n is not None and shape[0] != n:
                    shape = shape[1], shape[0]
                shapes.append(shape)
                dtypes.append(str(node["data"].dtype if isinstance(node, h5py.Group) else node.dtype))
            aux_shape = list(reversed(obj[aux_key].shape)) if aux_key in obj and hasattr(obj[aux_key], "shape") else None
            source_format = "MATLAB v7.3"
    else:
        from scipy.io import loadmat

        try:
            obj = loadmat(p, squeeze_me=False, struct_as_record=False)
        except TypeError as exc:
            # The public BBC4view.mat (also present in a second GitHub mirror)
            # contains valid X and y records followed by malformed trailing
            # bytes.  Loading only these two records stops before that tail.
            known_bbc4view = "b319f3887f068064c8608753fcc9e49b8d437b025266f52f47b1b39c44f01500"
            if ("Expecting miMATRIX type" not in str(exc)
                    or hashlib.sha256(p.read_bytes()).hexdigest() != known_bbc4view):
                raise
            obj = loadmat(p, variable_names=["X", "y"], squeeze_me=False, struct_as_record=False)
        keys = sorted(key for key in obj if not key.startswith("__"))
        vk = _choose_key(keys, views_key, VIEW_KEYS, "view", required=True)
        lk = _choose_key(keys, labels_key, LABEL_KEYS, "label", required=require_labels)
        cell = np.asarray(obj[vk])
        if cell.dtype != object:
            raise ValueError(f"{vk} must be a MATLAB cell array; got shape={cell.shape}, dtype={cell.dtype}")
        labels = np.asarray(obj[lk]).reshape(-1) if lk else None
        n = int(labels.size) if labels is not None else None
        raw_views = list(cell.ravel(order="C"))
        if not raw_views:
            raise ValueError(f"{vk} contains no views")
        aligned = [_aligned_view(view, n, f"view{i}") for i, view in enumerate(raw_views, 1)]
        views = aligned if load_views else []
        shapes = [view.shape for view in aligned]
        dtypes = [str(view.dtype) for view in aligned]
        aux_shape = list(np.asarray(obj[aux_key]).shape) if aux_key in obj else None
        source_format = "MATLAB v5"

    nset = {int(shape[0]) for shape in shapes}
    if len(nset) != 1:
        raise ValueError(f"all views must share sample count; got {sorted(nset)}")
    n = next(iter(nset))
    if labels is not None and len(labels) != n:
        raise ValueError(f"{lk} length {len(labels)} != n={n}")
    return {
        "path": str(p), "dataset": p.stem,
        "views_key": vk, "labels_key": lk,
        "aux_key": aux_key if aux_key in keys else None,
        "aux_shape": aux_shape,
        "n": n, "view_count": len(shapes),
        "view_dims": [int(shape[1]) for shape in shapes],
        "view_dtypes": dtypes,
        "classes": int(np.unique(labels).size) if labels is not None else None,
        "keys": keys, "source_format": source_format,
        "views": views, "labels": labels,
    }
