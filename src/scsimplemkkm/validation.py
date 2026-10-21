from __future__ import annotations

import numpy as np

from .operators import ImplicitCenteredNormalizedKernel, FusedCenteredOperator


def dense_inner_reference(
    kernels: list[ImplicitCenteredNormalizedKernel], gamma: np.ndarray, c: int
) -> tuple[float, np.ndarray]:
    """Small-n validation only: explicit dense top-c eigenspace."""
    K = np.zeros((kernels[0].n, kernels[0].n), dtype=np.float64)
    for g, op in zip(np.asarray(gamma, dtype=np.float64), kernels):
        K += (g * g) * op.explicit()
    lam, U = np.linalg.eigh(0.5 * (K + K.T))
    idx = np.argsort(lam)[-c:]
    return float(np.sum(lam[idx])), U[:, idx]


def fused_operator_error(
    kernels: list[ImplicitCenteredNormalizedKernel], gamma: np.ndarray, Z: np.ndarray
) -> float:
    fused = FusedCenteredOperator(kernels, gamma)
    A = fused.matmat(Z).astype(np.float64)
    B = np.zeros_like(A)
    for g, op in zip(np.asarray(gamma, dtype=np.float64), kernels):
        B += (g * g) * op.matmat(Z).astype(np.float64)
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), np.finfo(float).eps))
