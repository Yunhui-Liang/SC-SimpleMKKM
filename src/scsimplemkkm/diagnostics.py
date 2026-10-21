from __future__ import annotations

"""Post-fit diagnostics shared by the dense reference and scalable engine.

These diagnostics are deliberately separate from the optimization loop.  They
must never silently change a SimpleMKKM iterate; they only explain whether a
finished run is easy or numerically delicate (graph fragmentation, boundary
spectral gap, cluster-count mismatch, etc.).
"""

from typing import Any
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from .operators import FusedCenteredOperator


def ground_truth_diagnostic(y_true: np.ndarray | None, requested_c: int) -> dict[str, Any]:
    if y_true is None:
        return {'available': False, 'requested_c': int(requested_c), 'warnings': []}
    y = np.asarray(y_true).reshape(-1)
    vals = np.unique(y)
    tc = int(vals.size)
    warnings: list[str] = []
    if tc != int(requested_c):
        warnings.append(
            f'Ground-truth contains {tc} classes but requested c={int(requested_c)}. '
            'External metrics remain computable, but this is not the standard known-c benchmark protocol.'
        )
    return {
        'available': True,
        'n': int(y.size),
        'true_clusters': tc,
        'requested_c': int(requested_c),
        'matches_requested_c': bool(tc == int(requested_c)),
        'warnings': warnings,
    }


def graph_diagnostic_summary(graphs: dict, requested_c: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for view, fams in (graphs or {}).items():
        for family, g in (fams or {}).items():
            comp = int((g or {}).get('connected_components') or 0)
            isolated = int((g or {}).get('isolated') or 0)
            row = {
                'view': str(view), 'family': str(family),
                'connected_components': comp, 'isolated': isolated,
                'mean_degree': (g or {}).get('mean_degree'),
            }
            rows.append(row)
            if comp > int(requested_c):
                warnings.append(
                    f'{view}/{family}: support graph has {comp} connected components > c={int(requested_c)}; '
                    'the top-c spectral boundary may be degenerate or weakly separated.'
                )
            if isolated > 0:
                warnings.append(f'{view}/{family}: support graph contains {isolated} isolated samples.')
    return {'rows': rows, 'warnings': warnings}


def spectral_boundary_diagnostic(
    model,
    *,
    max_n: int = 200_000,
    tol: float = 1e-5,
    maxiter: int = 400,
) -> dict[str, Any]:
    """Estimate lambda_c, lambda_{c+1}, and their gap for the *final* fused operator.

    This is diagnostic-only.  For the scalable engine it uses ARPACK on a
    LinearOperator and therefore never materializes an n-by-n matrix.  To keep a
    monitoring feature from dominating a million-sample run, ``max_n`` is a
    hard automatic gate by default.
    """
    n = int(model.n); c = int(model.c)
    if c + 1 >= n:
        return {'status': 'skipped', 'reason': 'c+1 >= n', 'n': n, 'c': c}
    if n > int(max_n):
        return {'status': 'skipped_large_n', 'reason': f'n={n} exceeds spectral diagnostic max_n={int(max_n)}', 'n': n, 'c': c, 'max_n': int(max_n)}
    gamma = np.asarray(model.gamma_, dtype=np.float64)
    try:
        # Dense reference operators expose .K.  Build the same fused matrix used
        # by the reference eigensolver, in float64.
        if all(hasattr(op, 'K') for op in model.kernels):
            Kf = np.zeros((n, n), dtype=np.float64)
            for g, op in zip(gamma, model.kernels):
                if g != 0.0:
                    Kf += float(g * g) * np.asarray(op.K, dtype=np.float64)
            Kf = 0.5 * (Kf + Kf.T)
            vals = eigsh(Kf, k=c + 1, which='LA', tol=float(tol), maxiter=int(maxiter), return_eigenvectors=False)
            backend = 'dense_eigsh'
        else:
            fused = FusedCenteredOperator(model.kernels, gamma, dtype=model.dtype)
            def mv(x):
                return fused.matmat(np.asarray(x, dtype=model.dtype).reshape(n, 1))[:, 0]
            def mm(X):
                return fused.matmat(np.asarray(X, dtype=model.dtype))
            A = LinearOperator((n, n), matvec=mv, matmat=mm, dtype=np.dtype(model.dtype))
            vals = eigsh(A, k=c + 1, which='LA', tol=float(tol), maxiter=int(maxiter), return_eigenvectors=False)
            backend = 'implicit_eigsh'
        vals = np.sort(np.asarray(vals, dtype=np.float64))[::-1]
        lam_c = float(vals[c - 1]); lam_cp1 = float(vals[c])
        gap = float(lam_c - lam_cp1)
        rel_gap = float(gap / max(abs(lam_c), np.finfo(float).eps))
        warnings: list[str] = []
        # This threshold is only an engineering diagnostic, not a theorem.
        if rel_gap < 1e-4:
            warnings.append(f'Small final boundary eigengap: relative gap={rel_gap:.3e}. Inner top-c solves may be numerically delicate.')
        return {
            'status': 'ok', 'backend': backend, 'n': n, 'c': c,
            'lambda_c': lam_c, 'lambda_c_plus_1': lam_cp1,
            'gap': gap, 'relative_gap': rel_gap,
            'top_eigenvalues': vals.tolist(), 'warnings': warnings,
        }
    except ArpackNoConvergence as e:
        return {'status': 'not_converged', 'n': n, 'c': c, 'error': str(e), 'warnings': ['Final spectral-boundary diagnostic did not converge.']}
    except Exception as e:
        return {'status': 'failed', 'n': n, 'c': c, 'error': f'{type(e).__name__}: {e}', 'warnings': ['Final spectral-boundary diagnostic failed; optimization result is still preserved.']}


def aggregate_run_warnings(*parts: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        for w in p.get('warnings') or []:
            s = str(w)
            if s not in out:
                out.append(s)
    return out
