from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
from scipy.linalg import solve_triangular, eigh
from scipy.sparse.linalg import LinearOperator, lobpcg, eigsh, ArpackNoConvergence
import warnings

from .operators import ImplicitCenteredNormalizedKernel, FusedCenteredOperator
from .runtime import ProgressSink
from .discretize import discretize_embedding, DiscretizationResult


def _cross_gram64(A: np.ndarray, B: np.ndarray, block_rows: int = 131_072) -> np.ndarray:
    """Compute A^T B with float64 accumulation and bounded temporaries."""
    A = np.asarray(A); B = np.asarray(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError('row mismatch in block Gram')
    ca = 1 if A.ndim == 1 else A.shape[1]
    cb = 1 if B.ndim == 1 else B.shape[1]
    if A.ndim == 1: A = A[:, None]
    if B.ndim == 1: B = B[:, None]
    G = np.zeros((ca, cb), dtype=np.float64)
    bs = max(1, int(block_rows))
    for i in range(0, A.shape[0], bs):
        j = min(A.shape[0], i + bs)
        Ab = np.asarray(A[i:j], dtype=np.float64)
        Bb = np.asarray(B[i:j], dtype=np.float64)
        G += Ab.T @ Bb
    return G


def _fro2_64(A: np.ndarray, block_rows: int = 131_072) -> float:
    A = np.asarray(A)
    if A.ndim == 1: A = A[:, None]
    total = 0.0
    bs = max(1, int(block_rows))
    for i in range(0, A.shape[0], bs):
        j = min(A.shape[0], i + bs)
        total += float(np.einsum('ij,ij->', A[i:j], A[i:j], dtype=np.float64, optimize=True))
    return float(total)


def _subspace_stats(H: np.ndarray, AH: np.ndarray, block_rows: int = 131_072) -> tuple[np.ndarray, float, float]:
    """Return Rayleigh matrix, relative invariant-subspace residual, objective.

    A cancellation-free residual is evaluated blockwise only when the cheap
    orthonormal identity predicts that the residual is already small.  This
    preserves fast candidate screening while allowing strict 1e-6--1e-8
    acceptance checks without a full n x c residual allocation.
    """
    T = _cross_gram64(H, AH, block_rows)
    ah2 = _fro2_64(AH, block_rows)
    t2 = float(np.einsum('ij,ij->', T, T, dtype=np.float64))
    r2 = max(0.0, ah2 - t2)
    denom = max(np.sqrt(ah2), np.finfo(np.float64).eps)
    rel = float(np.sqrt(r2) / denom)
    if rel <= 1e-4:
        r2_stable = 0.0
        bs = max(1, int(block_rows))
        for i in range(0, H.shape[0], bs):
            j = min(H.shape[0], i + bs)
            Hb = np.asarray(H[i:j], dtype=np.float64)
            Ab = np.asarray(AH[i:j], dtype=np.float64)
            Rb = Ab - Hb @ T
            r2_stable += float(np.einsum('ij,ij->', Rb, Rb, dtype=np.float64, optimize=True))
        rel = float(np.sqrt(max(r2_stable, 0.0)) / denom)
    return T, rel, float(np.trace(T))


@dataclass
class SubspaceInfo:
    n_iter: int
    rel_residual: float
    objective: float
    fused_operator_bytes: int = 0
    solver: str = ''


@dataclass
class FitHistory:
    objective: list[float] = field(default_factory=list)
    gamma: list[np.ndarray] = field(default_factory=list)
    inner_iterations: list[int] = field(default_factory=list)
    inner_residual: list[float] = field(default_factory=list)
    line_search_trials: list[int] = field(default_factory=list)
    inner_seconds: list[float] = field(default_factory=list)
    line_search: list[dict] = field(default_factory=list)
    iteration_metrics: list[dict] = field(default_factory=list)
    acceptance_checks: list[dict] = field(default_factory=list)


class SparseSimpleMKKM:
    """Self-contained Python implementation of the SimpleMKKM core.

    Host objective (unchanged):
        min_{gamma in simplex} J(gamma)
        J(gamma) = max_{H^T H=I} Tr(H^T sum_v gamma_v^2 K_v H).

    The reduced-gradient construction follows the published SimpleMKKM
    equations. Dense eig/EVD is replaced only as a numerical solver: the inner
    top-c invariant subspace is obtained to a residual tolerance. Therefore the
    optimized objective is unchanged; finite tolerance gives the expected
    numerical epsilon-version of the exact inner solve.
    """

    def __init__(
        self,
        kernels: list[ImplicitCenteredNormalizedKernel],
        n_clusters: int,
        inner_tol: float = 1e-4,
        inner_max_iter: int = 200,
        outer_tol: float = 1e-5,
        outer_max_iter: int = 100,
        golden_delta: float = 1e-3,
        numerical_precision: float = 1e-16,
        random_state: int = 0,
        dtype=np.float32,
        orth_method: str = 'cholqr2',
        inner_solver: str = 'lobpcg',
        search_power_steps: int = 10,
        progress: ProgressSink | None = None,
        verbose: bool = False,
        iteration_observer=None,
        ambiguity_retry: bool = True,
        ambiguity_rtol: float = 1e-5,
        acceptance_tol: float = 1e-6,
        acceptance_max_iter: int = 800,
        acceptance_rtol: float = 1e-8,
        acceptance_atol: float = 1e-10,
        spmm_backend: str = 'auto',
        reduction_block_rows: int = 131_072,
        fused_buffer_dir: str | None = None,
    ) -> None:
        if not kernels:
            raise ValueError('at least one kernel is required')
        n = kernels[0].shape[0]
        if any(k.shape != (n, n) for k in kernels):
            raise ValueError('all kernels must have the same shape')
        if not (1 <= n_clusters < n):
            raise ValueError('n_clusters must be in [1, n-1]')
        self.kernels = kernels
        self.n = n
        self.m = len(kernels)
        self.c = int(n_clusters)
        self.inner_tol = float(inner_tol)
        self.inner_max_iter = int(inner_max_iter)
        self.outer_tol = float(outer_tol)
        self.outer_max_iter = int(outer_max_iter)
        self.golden_delta = float(golden_delta)
        self.numerical_precision = float(numerical_precision)
        self.random_state = int(random_state)
        self.dtype = dtype
        self.orth_method = orth_method
        self.inner_solver = str(inner_solver).lower()
        self.search_power_steps = max(1, int(search_power_steps))
        self.progress = progress or ProgressSink()
        self.verbose = verbose
        self.iteration_observer = iteration_observer
        self.ambiguity_retry = bool(ambiguity_retry)
        self.ambiguity_rtol = float(ambiguity_rtol)
        self.acceptance_tol = float(acceptance_tol)
        self.acceptance_max_iter = int(acceptance_max_iter)
        self.acceptance_rtol = float(acceptance_rtol)
        self.acceptance_atol = float(acceptance_atol)
        self.spmm_backend = str(spmm_backend or 'auto').lower()
        self.reduction_block_rows = max(4096, int(reduction_block_rows))
        # One reusable fused-data array per support family prevents repeated
        # hundreds-of-MB allocations during golden-section trials at 10M scale.
        self._fused_data_buffers: dict[tuple[int, int], np.ndarray] = {}
        self.fused_buffer_dir = fused_buffer_dir
        if self.acceptance_tol <= 0:
            raise ValueError('acceptance_tol must be positive')
        self.gamma_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.history_ = FitHistory()
        self.postprocess_backend_: str | None = None
        self.discretization_result_: DiscretizationResult | None = None
        self.fit_status_: str = 'not_run'
        self.termination_reason_: str = ''

    def _orth_qr(self, Z: np.ndarray) -> np.ndarray:
        Q, _ = np.linalg.qr(np.asarray(Z, dtype=self.dtype), mode='reduced')
        return np.asarray(Q, dtype=self.dtype)

    def _orth_cholqr2(self, Z: np.ndarray) -> np.ndarray:
        # Blocked Cholesky-QR2: only c x c state is global float64.  v0.7.5
        # promoted the complete n x c matrix to float64 twice per pass, which
        # becomes multi-GB at 10M scale.
        Q = np.array(Z, dtype=self.dtype, copy=True, order='C')
        bs = self.reduction_block_rows
        for _ in range(2):
            G = _cross_gram64(Q, Q, bs)
            G = 0.5 * (G + G.T)
            ridge = max(np.finfo(np.float64).eps * max(float(np.trace(G)), 1.0), 1e-15)
            try:
                L = np.linalg.cholesky(G + ridge * np.eye(G.shape[0]))
                for s0 in range(0, Q.shape[0], bs):
                    e0 = min(Q.shape[0], s0 + bs)
                    Bt = np.asarray(Q[s0:e0], dtype=np.float64).T
                    Qt = solve_triangular(L, Bt, lower=True, overwrite_b=True, check_finite=False)
                    Q[s0:e0] = Qt.T.astype(self.dtype, copy=False)
            except np.linalg.LinAlgError:
                return self._orth_qr(Q)
        return Q

    def _orth64_cholqr2(self, Z: np.ndarray) -> np.ndarray:
        Q = np.array(Z, dtype=np.float64, copy=True, order='C')
        bs = self.reduction_block_rows
        for _ in range(2):
            G = _cross_gram64(Q, Q, bs)
            G = 0.5 * (G + G.T)
            ridge = max(np.finfo(np.float64).eps * max(float(np.trace(G)), 1.0), 1e-15)
            try:
                L = np.linalg.cholesky(G + ridge * np.eye(G.shape[0]))
                for s0 in range(0, Q.shape[0], bs):
                    e0 = min(Q.shape[0], s0 + bs)
                    Bt = Q[s0:e0].T.copy()
                    Qt = solve_triangular(L, Bt, lower=True, overwrite_b=True, check_finite=False)
                    Q[s0:e0] = Qt.T
            except np.linalg.LinAlgError:
                Q, _ = np.linalg.qr(Q, mode='reduced')
                return np.asarray(Q, dtype=np.float64)
        return Q

    def _orth(self, Z: np.ndarray) -> np.ndarray:
        if self.orth_method == 'qr':
            return self._orth_qr(Z)
        if self.orth_method == 'cholqr2':
            return self._orth_cholqr2(Z)
        raise ValueError("orth_method must be 'cholqr2' or 'qr'")

    def _subspace_solve(
        self,
        gamma: np.ndarray,
        H0: np.ndarray | None,
        *,
        tol: float | None = None,
        max_iter: int | None = None,
        outer_index: int = 0,
        trial_index: int = 0,
    ) -> tuple[np.ndarray, SubspaceInfo]:
        tol = self.inner_tol if tol is None else float(tol)
        max_iter = self.inner_max_iter if max_iter is None else int(max_iter)
        if H0 is None:
            rng = np.random.default_rng(self.random_state)
            H = self._orth(rng.standard_normal((self.n, self.c), dtype=np.float32))
        else:
            H = self._orth(np.asarray(H0, dtype=self.dtype))

        self.progress.emit('simplemkkm:fuse', None, 'Fusing sparse kernel operators', outer=outer_index, trial=trial_index)
        fused = FusedCenteredOperator(self.kernels, gamma, dtype=self.dtype, spmm_backend=self.spmm_backend, data_buffers=self._fused_data_buffers, fused_buffer_dir=self.fused_buffer_dir)

        if self.inner_solver == 'auto':
            # Candidate-only solver: deterministic, warm-started and strictly
            # bounded.  Line-search values steer the search but are never
            # allowed into the public trajectory; every proposed state is
            # recomputed by _strict_subspace_solve before acceptance.  This
            # avoids pathological LOBPCG/ARPACK runtimes on clustered trial
            # spectra while preserving the strict accepted-state invariant.
            high_c_cap = 5 if (self.c >= 64 or self.c / max(self.n, 1) >= 0.05) else self.search_power_steps
            steps = min(max_iter, self.search_power_steps, high_c_cap)
            rel = np.inf
            obj = np.nan
            self.progress.emit(
                'simplemkkm:inner', 0.0,
                f'Bounded warm-start search ({steps} orthogonal iterations): outer {outer_index}, trial {trial_index}',
                outer=outer_index, trial=trial_index, inner_solver='auto_bounded_power',
                fused_operator_bytes=fused.memory_bytes(),
            )
            # A loose search tolerance is intentional: these values only rank
            # candidate steps.  The strict accepted-state solver below remains
            # at acceptance_tol and is the sole source of public objectives.
            high_c = bool(self.c >= 64 or self.c / max(self.n, 1) >= 0.05)
            search_tol = max(float(tol), 1e-2 if high_c else 1e-3)
            it_done = 0
            for it in range(1, steps + 1):
                self.progress.control.check() if self.progress.control is not None else None
                H = self._orth(fused.matmat(H))
                AH = fused.matmat(H)
                T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
                it_done = it
                if rel <= search_tol:
                    break
            self.progress.emit(
                'simplemkkm:inner', 1.0,
                f'Bounded search complete: outer {outer_index}, iter {it_done}, residual {rel:.2e}',
                outer=outer_index, trial=trial_index, inner_iter=it_done, residual=rel,
                inner_objective=obj, inner_solver='auto_bounded_power',
                fused_operator_bytes=fused.memory_bytes(),
            )
            return H, SubspaceInfo(it_done, rel, obj, fused.memory_bytes(), 'auto_bounded_power')

        if self.inner_solver == 'eigsh':
            self.progress.emit('simplemkkm:inner', 0.0, f'ARPACK top-{self.c} solve: outer {outer_index}, trial {trial_index}', outer=outer_index, trial=trial_index, inner_solver='eigsh', fused_operator_bytes=fused.memory_bytes())
            def _mv64(x):
                return fused.matmat64(np.asarray(x, dtype=np.float64).reshape(self.n, 1))[:, 0]
            def _mm64(X):
                return fused.matmat64(np.asarray(X, dtype=np.float64))
            A64 = LinearOperator((self.n, self.n), matvec=_mv64, matmat=_mm64, dtype=np.dtype(np.float64))
            # Search-state ARPACK is deliberately bounded.  Some trial weights
            # create clustered spectra for which ARPACK can spend minutes even
            # though the value is used only to steer a line search.  Accepted
            # states are recomputed separately by the strict solver, so a
            # bounded search approximation is both cheaper and scientifically
            # safer than allowing one trial to stall the whole run.
            search_maxiter = min(int(max_iter), 40)
            used_fallback = False
            try:
                vals, He = eigsh(A64, k=self.c, which='LA', tol=max(float(tol)*0.1, 1e-8), maxiter=search_maxiter, ncv=min(self.n, max(2*self.c+20, self.c+2)))
                H = self._orth(np.asarray(He, dtype=self.dtype))
            except ArpackNoConvergence as exc:
                used_fallback = True
                He = getattr(exc, 'eigenvectors', None)
                if He is not None and getattr(He, 'ndim', 0) == 2 and He.shape[1] >= self.c:
                    H = self._orth(np.asarray(He[:, -self.c:], dtype=self.dtype))
                else:
                    # Five warm-started orthogonal-iteration steps give a
                    # deterministic bounded fallback for candidate evaluation.
                    for _ in range(5):
                        H = self._orth(fused.matmat(H))
            AH = fused.matmat(H)
            T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
            solver_name = 'eigsh_search_fallback' if used_fallback else 'eigsh'
            self.progress.emit('simplemkkm:inner', 1.0, f'ARPACK search complete: outer {outer_index}, residual {rel:.2e}', outer=outer_index, trial=trial_index, inner_iter=search_maxiter if used_fallback else 1, residual=rel, inner_objective=obj, inner_solver=solver_name, fused_operator_bytes=fused.memory_bytes())
            return H, SubspaceInfo(search_maxiter if used_fallback else 1, rel, obj, fused.memory_bytes(), solver_name)

        if self.inner_solver == 'lobpcg':
            # High-c fallback: block LOBPCG scales poorly in orthogonalization
            # when the requested subspace is large.  ARPACK/eigsh applies the
            # same implicit fused operator and is much faster in this regime.
            use_eigsh = bool(self.inner_solver == 'eigsh' or self.c >= 64 or self.c / max(self.n, 1) >= 0.05)
            if use_eigsh:
                self.progress.emit('simplemkkm:inner', 0.0, f'ARPACK top-{self.c} solve: outer {outer_index}, trial {trial_index}', outer=outer_index, trial=trial_index, inner_solver='eigsh_high_c', fused_operator_bytes=fused.memory_bytes())
                def _mv64(x):
                    return fused.matmat64(np.asarray(x, dtype=np.float64).reshape(self.n, 1))[:, 0]
                def _mm64(X):
                    return fused.matmat64(np.asarray(X, dtype=np.float64))
                A64 = LinearOperator((self.n, self.n), matvec=_mv64, matmat=_mm64, dtype=np.dtype(np.float64))
                vals, He = eigsh(A64, k=self.c, which='LA', tol=max(float(tol)*0.1, 1e-8), maxiter=max_iter, ncv=min(self.n, max(2*self.c+20, self.c+2)))
                H = self._orth(np.asarray(He, dtype=self.dtype))
                AH = fused.matmat(H)
                T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
                self.progress.emit('simplemkkm:inner', 1.0, f'ARPACK complete: outer {outer_index}, residual {rel:.2e}', outer=outer_index, trial=trial_index, inner_iter=1, residual=rel, inner_objective=obj, inner_solver='eigsh_high_c', fused_operator_bytes=fused.memory_bytes())
                return H, SubspaceInfo(1, rel, obj, fused.memory_bytes(), 'eigsh_high_c')
            self.progress.emit('simplemkkm:inner', 0.0, f'LOBPCG top-{self.c} solve: outer {outer_index}, trial {trial_index}', outer=outer_index, trial=trial_index, inner_solver='lobpcg', fused_operator_bytes=fused.memory_bytes())
            def _mv(x):
                return fused.matmat(np.asarray(x, dtype=self.dtype).reshape(self.n, 1))[:, 0]
            def _mm(X):
                return fused.matmat(np.asarray(X, dtype=self.dtype))
            A = LinearOperator((self.n, self.n), matvec=_mv, matmat=_mm, dtype=np.dtype(self.dtype))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lob_out = lobpcg(
                    A, np.asarray(H, dtype=self.dtype), largest=True, tol=tol, maxiter=max_iter,
                    retResidualNormsHistory=True, restartControl=20, verbosityLevel=0,
                )
            # SciPy may fall back to a dense small-problem eigensolver when n is
            # too small relative to the block size.  In that branch some SciPy
            # versions return only (eigenvalues, eigenvectors) even when
            # retResidualNormsHistory=True.  Treat residual history as optional;
            # the residual used for certification is computed explicitly below.
            if isinstance(lob_out, tuple) and len(lob_out) == 3:
                vals, Hlob, rhist = lob_out
            else:
                vals, Hlob = lob_out[:2]
                rhist = []
            H = self._orth(np.asarray(Hlob, dtype=self.dtype))
            AH = fused.matmat(H)
            T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
            try:
                nit = max(1, len(rhist) - 3)
            except Exception:
                nit = max_iter
            self.progress.emit('simplemkkm:inner', 1.0, f'LOBPCG complete: outer {outer_index}, iter {nit}, residual {rel:.2e}', outer=outer_index, trial=trial_index, inner_iter=nit, residual=rel, inner_objective=obj, inner_solver='lobpcg', fused_operator_bytes=fused.memory_bytes())
            # v0.6.5: never hide an expensive 100-step power fallback behind a
            # line-search cost evaluation.  Return the LOBPCG solution and its
            # measured residual truthfully; the outer optimizer records whether
            # the accepted iterate is numerically certified at inner_tol.
            return H, SubspaceInfo(nit, rel, obj, fused.memory_bytes(), 'lobpcg')

        if self.inner_solver != 'power':
            raise ValueError("inner_solver must be 'auto', 'lobpcg', 'eigsh', or 'power'")

        rel = np.inf
        obj = np.nan
        log_every = max(10, max_iter // 100)
        for it in range(1, max_iter + 1):
            self.progress.control.check() if self.progress.control is not None else None
            Y = fused.matmat(H)
            H = self._orth(Y)
            AH = fused.matmat(H)
            T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
            if it == 1 or it % log_every == 0 or rel <= tol or it == max_iter:
                self.progress.emit(
                    'simplemkkm:inner', min(1.0, it / max_iter),
                    f'Inner subspace: outer {outer_index}, iter {it}, residual {rel:.2e}',
                    outer=outer_index, trial=trial_index, inner_iter=it, residual=rel,
                    inner_objective=obj, inner_solver='power', fused_operator_bytes=fused.memory_bytes(),
                )
            if rel <= tol:
                return H, SubspaceInfo(it, rel, obj, fused.memory_bytes(), 'power')
        return H, SubspaceInfo(max_iter, rel, obj, fused.memory_bytes(), 'power')

    def _strict_subspace_solve(
        self, gamma: np.ndarray, H0: np.ndarray | None, *, outer_index: int, trial_index: int
    ) -> tuple[np.ndarray, SubspaceInfo]:
        """High-precision acceptance solve for the sparse engine.

        Search evaluations deliberately remain cheap/float32.  Accepted states
        are recomputed in float64 arithmetic against the same stored sparse
        operator coefficients, so the outer trajectory has one numerical
        meaning across iterations.  No float64 sparse copy is materialized.
        """
        tol = float(self.acceptance_tol)
        max_iter = int(max(self.acceptance_max_iter, self.inner_max_iter))
        if H0 is None:
            rng = np.random.default_rng(self.random_state)
            H = rng.standard_normal((self.n, self.c)).astype(np.float64)
        else:
            H = np.asarray(H0, dtype=np.float64)
        H = self._orth64_cholqr2(H)

        fused = FusedCenteredOperator(self.kernels, gamma, dtype=self.dtype, spmm_backend=self.spmm_backend, data_buffers=self._fused_data_buffers, fused_buffer_dir=self.fused_buffer_dir)
        self.progress.emit(
            'simplemkkm:acceptance', 0.0,
            f'Strict acceptance solve: outer {outer_index}, trial {trial_index}, tol={tol:.1e}',
            outer=outer_index, trial=trial_index, acceptance_tol=tol, precision='float64',
        )

        def _mv(x):
            return fused.matmat64(np.asarray(x, dtype=np.float64).reshape(self.n, 1))[:, 0]

        def _mm(X):
            return fused.matmat64(np.asarray(X, dtype=np.float64))

        A = LinearOperator((self.n, self.n), matvec=_mv, matmat=_mm, dtype=np.dtype(np.float64))

        # Validation-only small-n/high-c strict checker.  When c is a sizable
        # block (100Leaves: n=1600,c=100), both LOBPCG and ARPACK can spend a
        # disproportionate amount of time on certain trial spectra.  For these
        # small benchmark matrices only, materialize the *same supplied fused
        # sparse operator* in float64 and use a partial dense Hermitian solve.
        # This changes neither the representation nor the objective; it is not
        # used by the scalable path and is guarded by n<=3000,c>=64.
        if self.c >= 64 and self.n <= 3000 and fused.fast:
            Kd = np.zeros((self.n, self.n), dtype=np.float64)
            for g, op in zip(np.asarray(gamma, dtype=np.float64), self.kernels):
                if g == 0.0:
                    continue
                assert op.sparse_core is not None and op.d is not None and op.u is not None and op.s is not None
                w = float(g * g)
                Kd += w * op.sparse_core.toarray().astype(np.float64, copy=False)
                d = np.asarray(op.d, dtype=np.float64)
                u = np.asarray(op.u, dtype=np.float64)
                Kd -= (w / self.n) * (np.outer(d, u) + np.outer(u, d))
                Kd += (w * float(op.s) / (self.n * self.n)) * np.outer(d, d)
            Kd = 0.5 * (Kd + Kd.T)
            vals, H = eigh(
                Kd, subset_by_index=[self.n - self.c, self.n - 1],
                driver='evr', check_finite=False, overwrite_a=True,
            )
            H = np.asarray(H, dtype=np.float64)
            AH = fused.matmat64(H)
            T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
            self.progress.emit(
                'simplemkkm:acceptance', 1.0,
                f'Strict acceptance complete: outer {outer_index}, residual {rel:.2e}',
                outer=outer_index, trial=trial_index, acceptance_tol=tol,
                residual=rel, objective=obj, inner_solver='dense_eigh_validation',
                validation_only=True,
            )
            return H, SubspaceInfo(1, rel, obj, fused.memory_bytes(), 'dense_eigh_validation')

        # Large block LOBPCG becomes disproportionately expensive when c is a
        # nontrivial fraction of n (e.g. 100Leaves: n=1600, c=100).  In this
        # regime ARPACK/eigsh on the same implicit LinearOperator is typically
        # much faster and is still certified by the identical explicit
        # Frobenius residual below.  Search-state solves remain the bounded auto candidate solver;
        # this branch affects only the strict acceptance checker.
        solver_tol = max(tol * 0.05, 1e-12)
        nit = 0
        rel = np.inf
        obj = np.nan
        use_eigsh = bool(self.inner_solver == 'eigsh' or (self.inner_solver == 'auto' and (self.c >= 64 or self.c / max(self.n, 1) >= 0.05)))
        if use_eigsh:
            try:
                vals, Hlob = eigsh(
                    A, k=self.c, which='LA', tol=solver_tol, maxiter=max_iter,
                    ncv=min(self.n, max(2 * self.c + 20, self.c + 2)),
                )
                H = np.asarray(Hlob, dtype=np.float64)
                nit = 1
            except ArpackNoConvergence as exc:
                Hpartial = getattr(exc, 'eigenvectors', None)
                if Hpartial is not None and getattr(Hpartial, 'ndim', 0) == 2 and Hpartial.shape[1] >= self.c:
                    H = np.asarray(Hpartial[:, -self.c:], dtype=np.float64)
                    nit = max_iter
                else:
                    # Fall back to strict LOBPCG from the current warm start;
                    # acceptance still depends only on the explicit residual.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        lob_out = lobpcg(
                            A, H, largest=True, tol=solver_tol, maxiter=max_iter,
                            retResidualNormsHistory=True, restartControl=20, verbosityLevel=0,
                        )
                    H = np.asarray(lob_out[1], dtype=np.float64)
                    rhist = lob_out[2] if isinstance(lob_out, tuple) and len(lob_out) == 3 else []
                    nit = max(1, len(rhist) - 3) if hasattr(rhist, '__len__') else max_iter
            H = self._orth64_cholqr2(H)
            AH = fused.matmat64(H)
            T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
        else:
            for attempt in range(3):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    lob_out = lobpcg(
                        A, H, largest=True, tol=solver_tol, maxiter=max_iter,
                        retResidualNormsHistory=True, restartControl=20, verbosityLevel=0,
                    )
                if isinstance(lob_out, tuple) and len(lob_out) == 3:
                    vals, Hlob, rhist = lob_out
                else:
                    vals, Hlob = lob_out[:2]
                    rhist = []
                H = np.asarray(Hlob, dtype=np.float64)
                H = self._orth64_cholqr2(H)
                AH = fused.matmat64(H)
                T, rel, obj = _subspace_stats(H, AH, self.reduction_block_rows)
                try:
                    nit += max(1, len(rhist) - 3)
                except Exception:
                    nit += max_iter
                if rel <= tol:
                    break
                solver_tol = max(solver_tol * 0.1, 1e-14)
        self.progress.emit(
            'simplemkkm:acceptance', 1.0,
            f'Strict acceptance complete: outer {outer_index}, residual {rel:.2e}',
            outer=outer_index, trial=trial_index, inner_iter=nit, residual=rel,
            inner_objective=obj, acceptance_tol=tol, precision='float64',
            fused_operator_bytes=fused.memory_bytes(),
        )
        strict_solver_name = 'eigsh_strict' if use_eigsh else 'lobpcg_strict'
        return H, SubspaceInfo(nit, rel, obj, fused.memory_bytes(), strict_solver_name)

    def _objective_and_grad(self, gamma: np.ndarray, H: np.ndarray, *, strict: bool = False) -> tuple[float, np.ndarray]:
        q = np.empty(self.m, dtype=np.float64)
        for v, K in enumerate(self.kernels):
            if strict and hasattr(K, 'quadratic_trace64'):
                q[v] = K.quadratic_trace64(H)
            else:
                q[v] = K.quadratic_trace(H)
        obj = float(np.dot(gamma * gamma, q))
        grad = 2.0 * gamma * q
        return obj, grad

    def _strict_state(
        self, gamma: np.ndarray, H0: np.ndarray | None, *, outer: int, role: str
    ) -> tuple[np.ndarray, SubspaceInfo, float, np.ndarray, float]:
        t0 = time.perf_counter()
        Hs, info = self._strict_subspace_solve(gamma, H0, outer_index=outer, trial_index=-2)
        seconds = float(time.perf_counter() - t0)
        obj, grad = self._objective_and_grad(gamma, Hs, strict=True)
        row = {
            'outer': int(outer), 'role': str(role), 'objective': float(obj),
            'inner_iterations': int(info.n_iter), 'inner_residual': float(info.rel_residual),
            'seconds': seconds, 'acceptance_tol': float(self.acceptance_tol),
            'certified': bool(info.rel_residual <= self.acceptance_tol), 'solver': info.solver,
        }
        self.history_.acceptance_checks.append(row)
        self.progress.emit(
            'simplemkkm:acceptance_check', None,
            f"Strict {role}: J={obj:.8e}, residual={info.rel_residual:.2e}",
            **row,
        )
        return Hs, info, float(obj), grad, seconds

    def _acceptance_margin(self, objective: float) -> float:
        return float(max(self.acceptance_atol, self.acceptance_rtol * max(abs(float(objective)), 1.0)))

    @staticmethod
    def _official_direction(gamma: np.ndarray, grad: np.ndarray, numerical_precision: float) -> tuple[np.ndarray, int]:
        """Reduced-gradient direction used by the released SimpleMKKM MATLAB code.

        The released code normalizes the gradient, selects the largest-weight
        kernel as the base variable, subtracts its gradient component, projects
        onto the admissible cone, and enforces a zero-sum direction.
        """
        g = np.asarray(grad, dtype=np.float64).copy()
        ng = float(np.linalg.norm(g))
        if not np.isfinite(ng) or ng <= np.finfo(np.float64).eps:
            return np.zeros_like(gamma), int(np.argmax(gamma))
        g /= ng
        coord = int(np.argmax(gamma))
        g -= g[coord]
        desc = -g * ((gamma > 0.0) | (g < 0.0))
        desc[coord] = -float(np.sum(desc))
        # Tiny numerical noise can otherwise manufacture infeasible directions.
        desc[np.abs(desc) < numerical_precision] = 0.0
        return desc, coord

    @staticmethod
    def _max_feasible_step(gamma: np.ndarray, desc: np.ndarray) -> float:
        ind = desc < 0.0
        if not np.any(ind):
            return 0.0
        return float(np.min(-gamma[ind] / desc[ind]))

    def _cost_at(
        self, base_gamma: np.ndarray, step: float, desc: np.ndarray, H0: np.ndarray,
        *, outer: int, trial: int, phase: str,
    ) -> tuple[np.ndarray, np.ndarray, SubspaceInfo, float, float]:
        gt = np.asarray(base_gamma + float(step) * desc, dtype=np.float64)
        gt[np.abs(gt) < self.numerical_precision] = 0.0
        gt = np.maximum(gt, 0.0)
        sm = float(np.sum(gt))
        if sm <= 0:
            raise FloatingPointError('SimpleMKKM line search produced a zero weight vector')
        gt /= sm
        t0 = time.perf_counter()
        Ht, info = self._subspace_solve(gt, H0, outer_index=outer, trial_index=trial)
        seconds = time.perf_counter() - t0
        # A line-search trial needs only J(gamma), not the per-kernel gradient.
        # _subspace_solve has already computed Tr(H^T K_gamma H) from the fused
        # operator.  Recomputing q_r(H) for every base kernel here would add an
        # O(m n k c) pass to *every* golden-section trial and dominates the
        # high-c / many-kernel regime (e.g. 100Leaves RBF7).  Gradients are
        # still recomputed kernel-by-kernel for strict accepted states.
        cost = float(info.objective)
        row = {
            'outer': int(outer), 'trial': int(trial), 'phase': str(phase),
            'step': float(step), 'objective': float(cost),
            'inner_iterations': int(info.n_iter), 'inner_residual': float(info.rel_residual),
            'inner_seconds': float(seconds), 'certified': bool(info.rel_residual <= self.inner_tol), 'solver': info.solver,
        }
        self.history_.line_search.append(row)
        self.progress.emit(
            'simplemkkm:line_search', None,
            f"Outer {outer} {phase} trial {trial}: step={step:.3e}, J={cost:.8e}, residual={info.rel_residual:.2e}",
            **row,
        )
        return gt, Ht, info, float(cost), float(seconds)

    def _official_update(
        self, gamma: np.ndarray, H: np.ndarray, grad: np.ndarray, cost_old: float, *, outer: int
    ) -> tuple[np.ndarray, np.ndarray, SubspaceInfo, float, int, float, str]:
        """Port of the released ``simpleMKKMupdate.m`` line-search semantics.

        Large dense EVD is the only part intentionally replaced: each cost call
        uses the same sparse implicit top-c solver as the rest of this package.
        """
        gold = (np.sqrt(5.0) + 1.0) / 2.0
        sigma_new = np.asarray(gamma, dtype=np.float64).copy()
        desc, pivot = self._official_direction(sigma_new, grad, self.numerical_precision)
        stepmin = 0.0
        costmin = float(cost_old)
        costmax = 0.0
        Hbest = H
        infobest = SubspaceInfo(0, np.nan, cost_old, 0, 'none')
        best_seconds = 0.0
        trial = 0
        stepmax = self._max_feasible_step(sigma_new, desc)
        deltmax = stepmax
        if not np.isfinite(stepmax) or stepmax <= 0.0:
            return sigma_new, H, infobest, cost_old, trial, best_seconds, 'no_feasible_step'
        stepmax = min(stepmax, 0.1)

        # Projected-gradient expansion from the official implementation.
        projected_loops = 0
        while costmax < costmin and stepmax > np.finfo(np.float64).eps and projected_loops < 100:
            projected_loops += 1
            trial += 1
            gt, Ht, info, costmax, sec = self._cost_at(
                sigma_new, stepmax, desc, Hbest, outer=outer, trial=trial, phase='projected'
            )
            if costmax < costmin:
                costmin = costmax
                sigma_new = gt
                Hbest, infobest, best_seconds = Ht, info, sec
                sigma_new[np.abs(sigma_new) < self.numerical_precision] = 0.0
                sigma_new /= float(np.sum(sigma_new))
                desc = desc * ((sigma_new > self.numerical_precision) | (desc > 0.0))
                desc[pivot] = -float(np.sum(np.delete(desc, pivot)))
                stepmax = self._max_feasible_step(sigma_new, desc)
                deltmax = stepmax
                costmax = 0.0 if stepmax > 0.0 else costmin

        # Golden-section refinement exactly mirrors the released code's interval logic.
        if stepmax <= np.finfo(np.float64).eps:
            return sigma_new, Hbest, infobest, costmin, trial, best_seconds, 'projected_boundary'
        steps = [stepmin, stepmax]
        costs = [costmin, costmax]
        hs = [Hbest, Hbest]
        infos = [infobest, infobest]
        secs = [best_seconds, best_seconds]
        denom = max(abs(float(deltmax)), np.finfo(np.float64).eps)
        golden_loops = 0
        while (stepmax - stepmin) > self.golden_delta * denom and stepmax > np.finfo(np.float64).eps and golden_loops < 80:
            golden_loops += 1
            stepmedr = stepmin + (stepmax - stepmin) / gold
            stepmedl = stepmin + (stepmedr - stepmin) / gold
            trial += 1
            _, Hr, ir, cr, sr = self._cost_at(sigma_new, stepmedr, desc, Hbest, outer=outer, trial=trial, phase='golden_r')
            trial += 1
            _, Hl, il, cl, sl = self._cost_at(sigma_new, stepmedl, desc, Hbest, outer=outer, trial=trial, phase='golden_l')
            steps = [stepmin, stepmedl, stepmedr, stepmax]
            costs = [costmin, cl, cr, costmax]
            hs = [Hbest, Hl, Hr, Hbest]
            infos = [infobest, il, ir, infobest]
            secs = [best_seconds, sl, sr, best_seconds]
            coord = int(np.argmin(costs))
            if coord == 0:
                stepmax, costmax = stepmedl, cl
                Hbest, infobest, best_seconds = Hl, il, sl
            elif coord == 1:
                stepmax, costmax = stepmedr, cr
                Hbest, infobest, best_seconds = Hr, ir, sr
            elif coord == 2:
                stepmin, costmin = stepmedl, cl
                Hbest, infobest, best_seconds = Hl, il, sl
            else:
                stepmin, costmin = stepmedr, cr
                Hbest, infobest, best_seconds = Hr, ir, sr

        coord = int(np.argmin(costs))
        cost_new = float(costs[coord])
        step = float(steps[coord])
        Hsel, infosel, secsel = hs[coord], infos[coord], secs[coord]
        if cost_new < cost_old:
            sigma_new = sigma_new + step * desc
            sigma_new[np.abs(sigma_new) < self.numerical_precision] = 0.0
            sigma_new = np.maximum(sigma_new, 0.0)
            sigma_new /= float(np.sum(sigma_new))
            # Re-evaluate the selected final point if the selected H came from an interval endpoint
            # not corresponding to the exact updated sigma.  This keeps the Python state consistent.
            trial += 1
            _, Hsel, infosel, cost_new, secsel = self._cost_at(
                sigma_new, 0.0, np.zeros_like(desc), Hsel, outer=outer, trial=trial, phase='search_selected'
            )
            # With an iterative inner eigensolver, the re-evaluated Rayleigh
            # trace can differ from the line-search estimate by roundoff/solver
            # tolerance. Never accept an update whose re-evaluated objective is
            # higher than the previous iterate.
            if cost_new < cost_old:
                return sigma_new, Hsel, infosel, cost_new, trial, secsel, 'improved'
            return gamma.copy(), H, infobest, cost_old, trial, best_seconds, 'no_improvement'
        return gamma.copy(), H, infobest, cost_old, trial, best_seconds, 'no_improvement'

    def _record_iteration_state(self, accepted_iter: int, obj: float, gamma: np.ndarray, H: np.ndarray, sinfo: SubspaceInfo, reason: str) -> None:
        row = {
            'accepted_iter': int(accepted_iter),
            'objective': float(obj),
            'inner_iterations': int(sinfo.n_iter),
            'inner_residual': float(sinfo.rel_residual),
            'certified': bool(sinfo.rel_residual <= self.acceptance_tol),
            'acceptance_tol': float(self.acceptance_tol),
            'reason': str(reason), 'solver': sinfo.solver,
            'gamma': np.asarray(gamma, dtype=np.float64).tolist(),
        }
        if self.iteration_observer is not None:
            try:
                extra = self.iteration_observer(int(accepted_iter), np.asarray(H), np.asarray(gamma, dtype=np.float64), float(obj), sinfo)
                if extra:
                    row.update(dict(extra))
            except Exception as e:
                row['observer_error'] = f'{type(e).__name__}: {e}'
        self.history_.iteration_metrics.append(row)
        self.progress.emit(
            'simplemkkm:accepted', None,
            f'Accepted state {accepted_iter}: J={obj:.8e}, residual={sinfo.rel_residual:.2e}',
            **row,
        )

    def fit(self) -> 'SparseSimpleMKKM':
        """Fit with cheap search evaluations and strict accepted-state checks.

        v0.7.2 invariant: every point written to the accepted objective
        trajectory is evaluated at the same strict tolerance.  Loose LOBPCG
        values are allowed only inside the line search and can never directly
        become an accepted state.
        """
        gamma = np.full(self.m, 1.0 / self.m, dtype=np.float64)
        self.history_ = FitHistory()
        self.fit_status_ = 'running'
        self.termination_reason_ = ''
        converged = False
        accepted_iter = 0

        # The initial point is part of the public trajectory, therefore it must
        # already have the same numerical meaning as all later accepted points.
        H, sinfo, obj, grad, inner_sec = self._strict_state(gamma, None, outer=0, role='initial')
        initial_certified = bool(sinfo.rel_residual <= self.acceptance_tol)
        if self.history_.acceptance_checks:
            self.history_.acceptance_checks[-1].update({'accepted': initial_certified, 'margin': None, 'delta_objective': None})
        self._record_iteration_state(accepted_iter, obj, gamma, H, sinfo, 'initial')

        if not initial_certified:
            self.termination_reason_ = 'strict_initial_uncertified'
        else:
            for outer in range(1, self.outer_max_iter + 1):
                self.history_.objective.append(float(obj))
                self.history_.gamma.append(gamma.copy())
                self.history_.inner_iterations.append(int(sinfo.n_iter))
                self.history_.inner_residual.append(float(sinfo.rel_residual))
                self.history_.inner_seconds.append(float(inner_sec))
                self.progress.emit(
                    'simplemkkm:outer', (outer - 1) / self.outer_max_iter,
                    f'Outer {outer}: strict objective={obj:.8e}', outer=outer, objective=float(obj),
                    gamma=gamma.tolist(), inner_iterations=int(sinfo.n_iter),
                    inner_residual=float(sinfo.rel_residual), certified=True,
                    objective_kind='strict_accepted', acceptance_tol=float(self.acceptance_tol),
                )

                gamma_old = gamma.copy()
                gamma_prop, H_prop, sinfo_prop, obj_prop, trials, sec_prop, search_reason = self._official_update(
                    gamma, H, grad, obj, outer=outer
                )
                self.history_.line_search_trials.append(int(trials))
                delta_gamma = float(np.max(np.abs(gamma_prop - gamma_old)))

                # If the cheap official search cannot even propose a descent
                # point, the current *strict* state is the state we keep.
                if search_reason in {'no_feasible_step', 'no_improvement'} or delta_gamma <= 0.0:
                    self.termination_reason_ = search_reason
                    converged = True
                    break

                # Mandatory strict re-evaluation of the proposed point.  This is
                # the only gate through which a new state can enter the public
                # objective/gamma/quality trajectory.
                if bool(getattr(self, 'reuse_search_for_strict_acceptance', False)) and sinfo_prop.rel_residual <= self.acceptance_tol:
                    H_cand, info_cand = H_prop, sinfo_prop
                    obj_cand, grad_cand = self._objective_and_grad(gamma_prop, H_cand, strict=True)
                    sec_cand = float(sec_prop)
                    self.history_.acceptance_checks.append({
                        'outer': int(outer), 'role': 'candidate', 'objective': float(obj_cand),
                        'inner_iterations': int(info_cand.n_iter), 'inner_residual': float(info_cand.rel_residual),
                        'seconds': sec_cand, 'acceptance_tol': float(self.acceptance_tol),
                        'certified': True, 'reused_search_solution': True,
                    })
                else:
                    H_cand, info_cand, obj_cand, grad_cand, sec_cand = self._strict_state(
                        gamma_prop, H_prop, outer=outer, role='candidate'
                    )
                cand_certified = bool(info_cand.rel_residual <= self.acceptance_tol)
                margin = self._acceptance_margin(obj)
                delta_obj = float(obj_cand - obj)
                accept = bool(cand_certified and obj_cand < obj - margin)
                if self.history_.acceptance_checks:
                    self.history_.acceptance_checks[-1].update({
                        'old_objective': float(obj), 'delta_objective': delta_obj,
                        'margin': float(margin), 'accepted': accept,
                        'proposed_max_abs_gamma_change': delta_gamma,
                        'search_reason': str(search_reason),
                    })
                self.progress.emit(
                    'simplemkkm:acceptance_decision', None,
                    f"Outer {outer}: strict ΔJ={delta_obj:.3e}, margin={margin:.3e}, accepted={accept}",
                    outer=outer, old_objective=float(obj), candidate_objective=float(obj_cand),
                    delta_objective=delta_obj, margin=float(margin), accepted=accept,
                    candidate_certified=cand_certified, candidate_residual=float(info_cand.rel_residual),
                    proposed_max_abs_gamma_change=delta_gamma,
                )

                if not cand_certified:
                    self.termination_reason_ = 'strict_candidate_uncertified'
                    converged = False
                    break
                if not accept:
                    if obj_cand < obj:
                        # A decrease smaller than the declared numerical margin
                        # is intentionally not treated as scientific progress.
                        self.termination_reason_ = 'strict_numerical_floor'
                    else:
                        self.termination_reason_ = 'strict_no_improvement'
                    converged = True
                    break

                # Commit only the strict candidate.  The objective trajectory is
                # now monotone by construction up to the explicit margin.
                gamma, H, sinfo, obj, grad, inner_sec = (
                    np.asarray(gamma_prop, dtype=np.float64), H_cand, info_cand,
                    float(obj_cand), grad_cand, float(sec_cand)
                )
                accepted_iter += 1
                self._record_iteration_state(accepted_iter, obj, gamma, H, sinfo, 'strict_improved')
                if self.verbose:
                    print(
                        f'outer={outer:03d} strict J={self.history_.objective[-1]:.10e}->{obj:.10e} '
                        f'dg={delta_gamma:.3e} trials={trials} res={sinfo.rel_residual:.2e}'
                    )
                if delta_gamma <= self.outer_tol:
                    converged = True
                    self.termination_reason_ = 'weight_variation'
                    break
            else:
                self.termination_reason_ = 'outer_max_iter'

        # Keep the final accepted strict state only.
        self.gamma_ = gamma
        self.embedding_ = np.asarray(H, dtype=self.dtype if self.dtype == np.float64 else np.float32)
        certified = bool(sinfo.rel_residual <= self.acceptance_tol)
        if self.termination_reason_ in {'strict_initial_uncertified', 'strict_candidate_uncertified'}:
            self.fit_status_ = self.termination_reason_
        elif converged and certified:
            self.fit_status_ = 'converged'
        elif converged:
            self.fit_status_ = 'converged_acceptance_not_certified'
        else:
            self.fit_status_ = self.termination_reason_ or 'stopped'
        self.progress.emit(
            'simplemkkm:outer', 1.0,
            f'SimpleMKKM stopped: {self.fit_status_} ({self.termination_reason_})',
            objective=float(obj), gamma=gamma.tolist(), fit_status=self.fit_status_,
            termination_reason=self.termination_reason_, inner_residual=float(sinfo.rel_residual),
            certified=certified, acceptance_tol=float(self.acceptance_tol),
            accepted_updates=int(accepted_iter),
        )
        return self

    def fit_predict(
        self,
        *,
        discretization_backend: str = 'auto',
        discretization_block_rows: int = 262_144,
        discretization_max_iter: int = 20,
        discretization_tol: float = 1e-4,
        discretization_init_sample: int = 200_000,
        discretization_init_restarts: int = 5,
        discretization_init_local_iter: int = 8,
        labels_out: np.ndarray | None = None,
        faiss_gpu: bool = False,
    ) -> np.ndarray:
        """Fit SimpleMKKM and discretize the final spectral embedding.

        The default large-n path is our own bounded-memory full-data Lloyd
        implementation. It avoids a sklearn runtime dependency and scans all
        samples on every Lloyd pass. FAISS sampled k-means is an optional
        engineering backend, not the default algorithmic definition.
        """
        self.fit()
        self.progress.emit('postprocess:kmeans', 0.0, 'Discretizing final embedding')
        res = discretize_embedding(
            self.embedding_, self.c,
            backend=discretization_backend,
            block_rows=discretization_block_rows,
            max_iter=discretization_max_iter,
            tol=discretization_tol,
            init_sample=discretization_init_sample,
            init_restarts=discretization_init_restarts,
            init_local_iter=discretization_init_local_iter,
            random_state=self.random_state,
            labels_out=labels_out,
            faiss_gpu=faiss_gpu,
            progress=self.progress,
        )
        self.discretization_result_ = res
        self.postprocess_backend_ = res.backend
        self.progress.emit('postprocess:kmeans', 1.0, f'{res.backend} complete', inertia=res.inertia, n_iter=res.n_iter)
        return res.labels
