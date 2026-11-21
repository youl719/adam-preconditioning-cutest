import math
import time
from typing import Callable, Optional, Tuple

import torch


# =============================================================================
# Preconditioner strategies
# =============================================================================

class IdentityPreconditioner:
    """Preconditioner that leaves the system untouched."""

    def update(self, xk: torch.Tensor, gk: torch.Tensor, hvp_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return torch.ones_like(xk)


class HessianDiagPreconditioner:
    """
    Builds the diagonal of the Hessian by using Hessian-vector
    products with coordinate basis vectors.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def update(self, xk: torch.Tensor, gk: torch.Tensor, hvp_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        dim = xk.numel()
        diag = torch.zeros(dim, dtype=xk.dtype, device=xk.device)
        for i in range(dim):
            e_i = torch.zeros(dim, dtype=xk.dtype, device=xk.device)
            e_i[i] = 1.0
            diag[i] = hvp_fn(e_i)[i]
        return torch.clamp(diag, min=self.eps)


class AdamPreconditioner:
    """
    Uses the Adam variance term to build a diagonal preconditioner.
    """

    def __init__(self, alpha: float = 1e-3, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.alpha = alpha
        self.beta2 = beta2
        self.eps = eps
        self.v: Optional[torch.Tensor] = None
        self.k: int = 0

    def update(self, xk: torch.Tensor, gk: torch.Tensor, hvp_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(xk)

        self.v = self.beta2 * self.v + (1 - self.beta2) * gk.pow(2)
        self.k += 1

        bias_correction = 1.0 - self.beta2 ** self.k
        v_hat = self.v / bias_correction

        return self.alpha * (torch.sqrt(v_hat) + self.eps)


class AMSGradPreconditioner:
    """
    Uses the AMSGrad variance term (elementwise running maximum of the
    bias-corrected Adam v) to build a diagonal preconditioner.

    Bias correction is kept for apples-to-apples comparison with
    AdamPreconditioner; this is a (mild) deviation from the paper-standard
    AMSGrad (Reddi et al. 2018), which omits bias correction.
    """

    def __init__(self, alpha: float = 1e-3, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.alpha = alpha
        self.beta2 = beta2
        self.eps = eps
        self.v: Optional[torch.Tensor] = None
        self.v_max: Optional[torch.Tensor] = None
        self.k: int = 0

    def update(self, xk: torch.Tensor, gk: torch.Tensor, hvp_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(xk)
            self.v_max = torch.zeros_like(xk)

        self.v = self.beta2 * self.v + (1 - self.beta2) * gk.pow(2)
        self.k += 1

        bias_correction = 1.0 - self.beta2 ** self.k
        v_hat = self.v / bias_correction

        # AMSGrad: elementwise running maximum of bias-corrected v
        self.v_max = torch.maximum(self.v_max, v_hat)

        return self.alpha * (torch.sqrt(self.v_max) + self.eps)


# =============================================================================
# Inner solvers (CG and line search)
# =============================================================================

"""
Unpreconditioned CG and backwards Armijo taken from Lim & Roosta.
Created on Tue Oct 25 12:54:55 2022

@author: uqalim8
"""
def Ax(A, x):
    if callable(A):
        return A(x)
    else:
        return A @ x

def diag_preconditioned_CG(A, b, M_diag=None, tol=1e-2, maxite=100):
    """
    Conjugate gradient method with a diagonal preconditioner.
    """
    if M_diag is None:
        M_diag = torch.ones_like(b)
    
    x0 = torch.zeros_like(b) 
    rkm1 = b - Ax(A, x0)
    ykm1 = rkm1 / M_diag
    pk = ykm1.clone()
    xkm1 = x0
    k = 0
    
    while torch.norm(rkm1) > tol and k < maxite:
        Apk = Ax(A, pk)
        rk_1_dot_yk_1 = torch.dot(rkm1, ykm1)
        
        alpha = rk_1_dot_yk_1 / torch.dot(pk, Apk)
        xk = xkm1 + alpha * pk
        rk = rkm1 - alpha * Apk
        
        if torch.norm(rk) <= tol:
            k += 1
            xkm1 = xk
            break

        yk = rk / M_diag
        beta = torch.dot(rk, yk) / rk_1_dot_yk_1
        pk = yk + beta * pk
        rkm1 = rk
        ykm1 = yk
        xkm1 = xk
        k += 1
        
    return xkm1, k

def backwardArmijo(fun, xk, fk, gk, alpha, pk, beta, rho, maxite, minalpha=1e-18):
    betagp = torch.dot(gk, pk) * beta
    j = 1
    while fun(xk + alpha * pk) > fk + alpha * betagp and j < maxite:
        alpha *= rho
        j += 1
        if alpha < minalpha:
            alpha = minalpha
            break
    if j >= maxite:
        print("linesearch max exceeded!")
    return alpha, j


# =============================================================================
# 3. Main Newton-CG Solver
# =============================================================================

def _termination(k, g_norm, oracles, alpha, maxite, gradtol, max_oracles):
    return (k > maxite or g_norm < gradtol or oracles > max_oracles or alpha < 1e-18)

def newton_cg(
    fun,
    x0,
    preconditioner,
    alpha0=1.0,
    gradtol=1e-9,
    maxite=1000,
    max_oracles=300000,
    cg_min_tol=1e-13,
    inmaxite=1000,
    lineMaxite=500,
    lineBetaB=1e-4,
    lineRho=0.5,
    grad_cost=2,
    hvp_cost=2,
    func_cost=1,
    callback=None
):
    """
    Newton-CG solver with preconditioned CG.
    """
    xk = x0.clone()
    
    # Initial Evaluation
    fk, gk, hvp_fn = fun(xk, "012")
    g_norm = torch.linalg.norm(gk).item()

    k = 0
    oracles = grad_cost  # initial f,g call
    total_time = 0.0
    alpha = alpha0
    
    # Initialize loop variables for logging
    cg_iters = 0
    line_iters = 0
    cg_tol = 0.0
    M_diag = torch.ones_like(xk)

    # Initialize snapshot variables for Iter 0
    snap_g = gk
    snap_hvp = hvp_fn
    snap_M = M_diag

    # 1. Iteration 0 Log
    if callback:
        callback(locals(), action="log")

    while not _termination(k, g_norm, oracles, alpha, maxite, gradtol, max_oracles):
        t0 = time.time()
        
        # Dynamic CG tolerance from Numerical Optimization algorithm 7.1
        eta_k = min(0.5, math.sqrt(g_norm))
        cg_tol = max(eta_k * g_norm, cg_min_tol)
        
        # A. Preconditioner Update
        M_diag = preconditioner.update(xk, gk, hvp_fn)
        
        if isinstance(preconditioner, HessianDiagPreconditioner):
            oracles += x0.numel() * hvp_cost

        # Capture snapshot data before solving so dump matches legacy timing.
        snap_g, snap_hvp, snap_M = gk, hvp_fn, M_diag
        if callback:
            callback(locals(), action="dump")

        # B. Inner CG Solve
        pk, cg_iters = diag_preconditioned_CG(hvp_fn, -gk, M_diag, tol=cg_tol, maxite=inmaxite)

        # C. Line Search
        alpha, line_iters = backwardArmijo(
            lambda x: fun(x, "0"),
            xk, fk, gk,
            alpha0,
            pk,
            lineBetaB, lineRho, lineMaxite
        )
        
        # D. Update
        xk = xk + alpha * pk
        fk, gk, hvp_fn = fun(xk, "012")
        g_norm = torch.linalg.norm(gk).item()
        
        dt = time.time() - t0
        total_time += dt
        k += 1
        
        # Oracle Counting: Matches original 'oracle_calls' (2 + 2*cg + line)
        oracles += grad_cost + (cg_iters * hvp_cost) + (line_iters * func_cost)
        
        # 2. End of Iteration Log
        if callback:
            callback(locals(), action="log")

    return xk, locals()
 

# =============================================================================
# 4. Preconditioned Trust-Region Solver (Steihaug-CG)
# =============================================================================

def PreconditionedCGSteihaug(H, g, M_diag, delta, tol, maxite):
    """
    Preconditioned CG-Steihaug for Trust Region.

    Returns
    -------
    p        – step
    m0_mk    – predicted reduction
    reason   – termination reason string
    j        – CG iterations
    ax_calls – total number of Ax(H, ·) calls made (for oracle cross-checking)
    """
    z = torch.zeros_like(g)
    r = g.clone()
    
    # Apply preconditioner: y = M^-1 * r
    y = r / M_diag
    d = -y.clone()
    
    Bd = Ax(H, d)
    ax_calls = 1  # count the initial Ax call
    ry_val = torch.dot(r, y)
    
    j = 0
    while j < maxite:
        # Safeguard: if preconditioned residual inner product vanishes,
        # r is effectively zero — return current best solution.
        if abs(ry_val) < 1e-30:
            p = z
            Hp = Ax(H, p); ax_calls += 1
            m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
            return p, m0_mk, "Preconditioned residual vanished", j, ax_calls

        dBd = torch.dot(d, Bd)
        j += 1
        
        # 1. Negative Curvature
        if dBd <= 0:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            numerator = -dz + torch.sqrt(dz**2 - norm_d**2 * (norm_z**2 - delta**2))
            tau = numerator / norm_d**2
            p = z + tau * d
            Hp = Ax(H, p); ax_calls += 1
            m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
            return p, m0_mk, "Negative Curvature (Boundary)", j, ax_calls
        
        alpha = ry_val / dBd
        zp1 = z + alpha * d
        
        # 2. Exceeded Trust Region
        if torch.norm(zp1) >= delta:
            dz = torch.dot(d, z)
            norm_d, norm_z = torch.norm(d), torch.norm(z)
            discriminant = torch.clamp(dz**2 - norm_d**2 * (norm_z**2 - delta**2), min=0.0)
            numerator = -dz + torch.sqrt(discriminant)
            tau = numerator / norm_d**2
            p = z + tau * d
            Hp = Ax(H, p); ax_calls += 1
            m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
            return p, m0_mk, "Exceeded Trust Region", j, ax_calls

        z = zp1
        r = r + alpha * Bd
        
        # 3. Solved to tolerance
        if torch.norm(r) < tol:
            p = z
            Hp = Ax(H, p); ax_calls += 1
            m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
            return p, m0_mk, "Solved (Interior)", j, ax_calls
        
        # Apply preconditioner for the next step
        y = r / M_diag
        ry_next = torch.dot(r, y)

        # Safeguard: if next preconditioned residual inner product vanishes,
        # we've converged — return current solution.
        if abs(ry_next) < 1e-30:
            p = z
            Hp = Ax(H, p); ax_calls += 1
            m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
            return p, m0_mk, "Preconditioned residual vanished", j, ax_calls
        
        beta = ry_next / ry_val
        d = -y + beta * d
        ry_val = ry_next
        Bd = Ax(H, d)
        ax_calls += 1  # count each per-iteration Ax call
     
    p = z
    Hp = Ax(H, p); ax_calls += 1
    m0_mk = -torch.dot(g, p) - torch.dot(p, Hp) / 2
    return p, m0_mk, "Max Iterations reached", j, ax_calls


# =============================================================================
# 5. Trust-Region Newton with per-iteration logging
# =============================================================================

def trust_region_newton(
    fun, x0, preconditioner,
    initial_delta=1.0, max_delta=10.0,
    gradtol=1e-6, maxite=1000, max_oracles=300000,
    iteration_log: list = None,
):
    """
    Outer Trust-Region Loop with per-iteration oracle accounting.

    Parameters
    ----------
    iteration_log : list or None
        If a list is passed, a dict is appended for every outer iteration
        (including iteration 0 = initialisation).  Each dict contains:

            iter           – outer iteration number
            f_val          – objective value f(x_k)
            grad_norm      – ||∇f(x_k)||
            func_oracles   – function-eval oracle units spent THIS iteration
            grad_oracles   – gradient oracle units spent THIS iteration
            hvp_oracles    – HVP oracle units spent THIS iteration
            total_oracles  – cumulative oracle units up to and including this iter
            delta          – trust-region radius
            cg_iters       – inner CG iterations this step
            cg_reason      – CG termination reason
            step_accepted  – whether the step was accepted
    """
    # =========================================================================
    # KEY ASSUMPTIONS (read before modifying):
    #
    # 1. IMPLICIT ORACLE COUNTING — hvp_fn (returned by fun(xk, "012") or
    #    fun(xk, "12")) is a closure that auto-increments fun.hvp_units on
    #    every call.  PreconditionedCGSteihaug receives hvp_fn as a generic
    #    operator H; it does not know it is mutating external state.  All HVP
    #    costs inside CG are captured by the snapshot differencing (_snap /
    #    _delta_counters), NOT by explicit counting.  If hvp_fn is ever
    #    replaced with a plain matrix or a non-oracle callable, HVP costs
    #    will silently read as zero.
    #
    # 2. HESSDIAG "1 HVP" HACK — HessianDiagPreconditioner.update() calls
    #    hvp_fn n times (one per coordinate), but we override the recorded
    #    cost to a single hvp_cost unit.  This hack assumes the preconditioner
    #    is ONLY rebuilt on accepted steps.  If you add a rebuild on rejected
    #    steps, the hack will still fire and credit 1 HVP for redundant work
    #    at the same iterate.  Either guard with a flag or remove the hack. 
    # =========================================================================

    xk = x0.clone()
    delta = initial_delta

    # --- Helper to snapshot and compute deltas ---
    def _snap():
        return fun.snapshot_counters()

    def _delta_counters(before, after):
        return {
            "func_oracles": after["func_units"] - before["func_units"],
            "grad_oracles": after["grad_units"] - before["grad_units"],
            "hvp_oracles":  after["hvp_units"]  - before["hvp_units"],
        }

    # ---- ITERATION 0: Initialisation ----
    snap_before = _snap()

    fk, gk, hvp_fn = fun(xk, "012")
    g_norm = torch.linalg.norm(gk).item()

    # Build initial preconditioner.
    # For HessianDiag the n HVP calls go through hvp_fn which auto-increments
    # fun.hvp_units.  We then apply the "1 theoretical HVP" hack.
    if isinstance(preconditioner, HessianDiagPreconditioner):
        pre_precond = _snap()
        M_diag = preconditioner.update(xk, gk, hvp_fn)
        # Override: treat the whole diagonal build as 1 HVP cost
        actual_hvp_spent = fun.hvp_units - pre_precond["hvp_units"]
        fun.hvp_units -= (actual_hvp_spent - fun.hvp_cost)
    else:
        M_diag = preconditioner.update(xk, gk, hvp_fn)

    snap_after = _snap()
    deltas = _delta_counters(snap_before, snap_after)

    if iteration_log is not None:
        iteration_log.append({
            "iter":           0,
            "f_val":          fk,
            "grad_norm":      g_norm,
            "func_oracles":   deltas["func_oracles"],
            "grad_oracles":   deltas["grad_oracles"],
            "hvp_oracles":    deltas["hvp_oracles"],
            "total_oracles":  fun.oracle_units,
            "delta":          delta,
            "cg_iters":       0,
            "cg_reason":      "init",
            "step_accepted":  True,
        })

    k = 0

    while k < maxite and g_norm > gradtol and fun.oracle_units < max_oracles and delta > 1e-12:

        snap_before = _snap()

        # ---- INNER SOLVE ----
        cg_tol = 0.5 * g_norm
        snap_pre_cg = _snap()
        p, pred, reason, cg_iters, ax_calls = PreconditionedCGSteihaug(
            hvp_fn, gk, M_diag, delta, cg_tol, maxite=100
        )
        snap_post_cg = _snap()

        # Cross-check: CG's own Ax count should match oracle snapshot delta.
        # Each Ax call increments fun.hvp_units by fun.hvp_cost.
        expected_hvp_units = ax_calls * fun.hvp_cost
        actual_hvp_units = snap_post_cg["hvp_units"] - snap_pre_cg["hvp_units"]
        assert expected_hvp_units == actual_hvp_units, (
            f"CG Ax count mismatch: CG reported {ax_calls} Ax calls "
            f"({expected_hvp_units} hvp units) but oracle delta is "
            f"{actual_hvp_units} hvp units"
        )

        if pred <= 0:
            # Model breakdown — log and terminate
            snap_after = _snap()
            deltas = _delta_counters(snap_before, snap_after)
            if iteration_log is not None:
                iteration_log.append({
                    "iter":           k + 1,
                    "f_val":          fk,
                    "grad_norm":      g_norm,
                    "func_oracles":   deltas["func_oracles"],
                    "grad_oracles":   deltas["grad_oracles"],
                    "hvp_oracles":    deltas["hvp_oracles"],
                    "total_oracles":  fun.oracle_units,
                    "delta":          delta,
                    "cg_iters":       cg_iters,
                    "cg_reason":      f"Model breakdown ({reason})",
                    "step_accepted":  False,
                })
            break

        # ---- STEP EVALUATION ----
        fk_next = fun(xk + p, "0")
        actual = fk - fk_next
        rho = actual / pred

        # ---- TRUST REGION RADIUS UPDATE ----
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and ("Boundary" in reason or "Exceeded" in reason):
            delta = min(2.0 * delta, max_delta)

        # ---- STATE UPDATE ----
        step_accepted = rho > 0.1

        if step_accepted:
            xk = xk + p
            fk = fk_next
            gk, hvp_fn = fun(xk, "12")
            g_norm = torch.linalg.norm(gk).item()

            # Rebuild preconditioner at new point
            if isinstance(preconditioner, HessianDiagPreconditioner):
                pre_precond = _snap()
                M_diag = preconditioner.update(xk, gk, hvp_fn)
                actual_hvp_spent = fun.hvp_units - pre_precond["hvp_units"]
                fun.hvp_units -= (actual_hvp_spent - fun.hvp_cost)
            else:
                M_diag = preconditioner.update(xk, gk, hvp_fn)

        k += 1

        snap_after = _snap()
        deltas = _delta_counters(snap_before, snap_after)

        if iteration_log is not None:
            iteration_log.append({
                "iter":           k,
                "f_val":          fk,
                "grad_norm":      g_norm,
                "func_oracles":   deltas["func_oracles"],
                "grad_oracles":   deltas["grad_oracles"],
                "hvp_oracles":    deltas["hvp_oracles"],
                "total_oracles":  fun.oracle_units,
                "delta":          delta,
                "cg_iters":       cg_iters,
                "cg_reason":      reason,
                "step_accepted":  step_accepted,
            })

    return xk, {"k": k, "g_norm": g_norm}