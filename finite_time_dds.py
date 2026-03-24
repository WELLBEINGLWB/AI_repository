"""
Finite-Time Discrete Dynamical System for QP Solving (EIQP)

Implements the EIQP algorithm from:
  "EIQP: Execution-time-certified and Infeasibility-detecting QP Solver"
  L. Wu, W. Xiao, R. D. Braatz, IEEE TAC, 2025.
  DOI: 10.1109/TAC.2025.3631575

A finite-time discrete dynamical system is a system whose state evolves via
a discrete-time map F:

    x̄_{k+1} = F(x̄_k)

and is guaranteed to reach a target set in a *finite*, *certifiable* number
of steps N regardless of the problem data (only dimension-dependent).

The EIQP algorithm is such a system: starting from a known initial point, it
drives the complementarity gap and infeasibility residual to zero in exactly

    N = ceil( log((n+1)/ε) / (-log(1 - β/√(n+1))) )

iterations, where n is the problem dimension, ε is the target optimality
level, and β = √2 - 1 ≈ 0.4142.

Problem class
-------------
The solver handles convex Quadratic Programs (QPs) of the form:

    min   (1/2) z' Q z + c' z
    s.t.  A z ≥ b
          z ≥ 0

which arise in Model Predictive Control (MPC), Control Barrier Function
(CBF)-based QP, and many other engineering applications.

Algorithm overview
------------------
1. Convert QP → LCP (Linear Complementarity Problem) via KKT conditions.
2. Lift LCP → HLCP (Homogeneous LCP) by adding scalars τ, κ ≥ 0, enabling
   simultaneous infeasibility detection.
3. Scale and initialise: x̄⁰ = (e; 1), s̄⁰ = (e; 1).
4. Run N full-Newton-step IPM iterations (no line search required).
5. Decode: if τ > κ the QP is feasible and z* = x/τ; if κ > τ the QP is
   infeasible.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# QP → LCP conversion
# ---------------------------------------------------------------------------

def qp_to_lcp(Q, c, A, b):
    """Convert QP to LCP form.

    The convex QP
        min  (1/2) z'Qz + c'z
        s.t. Az ≥ b,  z ≥ 0
    has KKT conditions that can be written as a Linear Complementarity
    Problem (LCP):
        s = M x + q,   x ⊙ s = 0,   (x, s) ≥ 0
    where
        x = col(z, λ),
        M = [[Q, -A'], [A, 0]],
        q = col(c, -b).
    M has a positive-semidefinite symmetric part, as required by the HLCP
    theory.

    Parameters
    ----------
    Q : (nz, nz) array_like  – PSD cost matrix
    c : (nz,) array_like     – linear cost vector
    A : (nb, nz) array_like  – constraint matrix
    b : (nb,) array_like     – constraint rhs

    Returns
    -------
    M : (n, n) ndarray  where n = nz + nb
    q : (n,)   ndarray
    """
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    nz = Q.shape[0]
    nb = A.shape[0]

    M = np.block([
        [Q,                    -A.T              ],
        [A,  np.zeros((nb, nb))                  ],
    ])
    q = np.concatenate([c, -b])

    return M, q


# ---------------------------------------------------------------------------
# HLCP helper functions
# ---------------------------------------------------------------------------

def _psi(x, tau, M, q):
    """HLCP map  ψ(x, τ) = col(Mx + qτ,  -x'Mx/τ - x'q).

    Parameters
    ----------
    x   : (n,)   current x-iterate
    tau : float  current τ-iterate (> 0)
    M   : (n,n)  LCP matrix
    q   : (n,)   LCP vector

    Returns
    -------
    psi_bar : (n+1,)
    """
    Mx = M @ x
    psi_x = Mx + q * tau
    psi_tau = -(x @ Mx) / tau - (x @ q)
    return np.append(psi_x, psi_tau)


def _jac_psi(x, tau, M, q):
    """Jacobian Jψ(x, τ) of size (n+1) × (n+1).

    Jψ = [  M,                q        ]   (n rows)
         [ -(x'(M+M'))/τ - q',  x'Mx/τ² ]   (1 row)
    """
    n = len(x)
    J = np.zeros((n + 1, n + 1))

    J[:n, :n] = M                                           # top-left  : M
    J[:n, n] = q                                            # top-right : q
    J[n, :n] = -(x @ (M + M.T)) / tau - q                  # bottom-left
    J[n, n] = (x @ (M @ x)) / (tau ** 2)                   # bottom-right

    return J


# ---------------------------------------------------------------------------
# Exact iteration complexity
# ---------------------------------------------------------------------------

def exact_iteration_count(n, epsilon):
    """Return the exact number of EIQP iterations required.

    Theorem 3 of Wu et al. (2025):

        N = ceil( log((n+1)/ε) / (-log(1 - β/√(n+1))) )

    where β = √2 − 1 ≈ 0.4142.

    Parameters
    ----------
    n       : int   – LCP dimension (= nz + nb for QP with nz variables,
                       nb inequality constraints)
    epsilon : float – target optimality level (duality gap and infeasibility
                       residual norm both ≤ ε after N iterations)

    Returns
    -------
    N : int  – exact (not an upper bound) iteration count
    """
    beta = math.sqrt(2) - 1          # ≈ 0.414213
    eta = beta / math.sqrt(n + 1)    # per-iteration reduction exponent
    N = math.ceil(
        math.log((n + 1) / epsilon) / (-math.log(1.0 - eta))
    )
    return N


# ---------------------------------------------------------------------------
# Single Newton step (one application of the discrete map F)
# ---------------------------------------------------------------------------

def _newton_step(x, tau, s, kappa, M, q, eta, gamma):
    """Perform one full-Newton-step IPM iteration.

    Solves the linearised system
        ds̄ − Jψ(x̄) dx̄ = −η r̄
        diag(x̄) ds̄ + diag(s̄) dx̄ = γ μ̄ e − x̄ ⊙ s̄
    (no line search; α = 1) and returns updated iterates.

    Parameters
    ----------
    x, tau   : primal augmented state (x ∈ Rⁿ, τ ∈ R)
    s, kappa : dual augmented state   (s ∈ Rⁿ, κ ∈ R)
    M, q     : scaled LCP data
    eta      : step-size parameter η = β/√(n+1)
    gamma    : centering parameter γ = 1 − η

    Returns
    -------
    x_new, tau_new, s_new, kappa_new, mu_bar, r_bar_norm
    """
    n = len(x)
    x_bar = np.append(x, tau)
    s_bar = np.append(s, kappa)

    # Complementarity gap
    mu_bar = float(x_bar @ s_bar) / (n + 1)

    # Infeasibility residual
    psi_old = _psi(x, tau, M, q)
    r_bar = s_bar - psi_old

    # Jacobian
    J = _jac_psi(x, tau, M, q)

    # Assemble and solve Newton system:
    #   (diag(x̄) · Jψ + diag(s̄)) dx̄ = γ μ̄ e − x̄⊙s̄ + η diag(x̄) r̄
    e_bar = np.ones(n + 1)
    lhs = np.diag(x_bar) @ J + np.diag(s_bar)
    rhs = gamma * mu_bar * e_bar - x_bar * s_bar + eta * x_bar * r_bar

    dx_bar = np.linalg.solve(lhs, rhs)
    ds_bar = -eta * r_bar + J @ dx_bar

    # Update primal
    x_bar_new = x_bar + dx_bar
    x_new = x_bar_new[:n]
    tau_new = float(x_bar_new[n])

    # Update dual including nonlinear correction (guarantees r̄⁺ = (1−η)r̄)
    psi_new = _psi(x_new, tau_new, M, q)
    s_bar_new = s_bar + ds_bar + psi_new - psi_old - J @ dx_bar

    s_new = s_bar_new[:n]
    kappa_new = float(s_bar_new[n])

    return x_new, tau_new, s_new, kappa_new, mu_bar, float(np.linalg.norm(r_bar))


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class FiniteTimeDDS:
    """EIQP solver formulated as a finite-time discrete dynamical system.

    The state (x̄_k, s̄_k) evolves under the discrete map F:

        (x̄_{k+1}, s̄_{k+1}) = F(x̄_k, s̄_k)

    and converges to the QP solution (or infeasibility certificate) in
    exactly N = ceil(log((n+1)/ε) / (−log(1 − β/√(n+1)))) steps.

    Parameters
    ----------
    epsilon : float, optional
        Target optimality level ε (default 1e-8).  Both the duality gap and
        the infeasibility residual norm will be ≤ ε after N iterations.
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    def solve(self, Q, c, A, b):
        """Solve the convex QP.

        Parameters
        ----------
        Q : (nz, nz) array_like  – PSD cost matrix
        c : (nz,) array_like     – linear cost vector
        A : (nb, nz) array_like  – constraint matrix (rows: Aᵢ z ≥ bᵢ)
        b : (nb,) array_like     – constraint RHS

        Returns
        -------
        result : dict with keys
            'status'            : 'optimal' or 'infeasible'
            'solution'          : optimal z* (ndarray) if feasible, else None
            'objective'         : optimal cost (1/2)*z*'Qz* + c'*z* if feasible
            'iterations'        : number of iterations actually executed
            'planned_iterations': exact N predicted by Theorem 3
            'duality_gap'       : final complementarity gap x̄'s̄
            'residual_norm'     : final infeasibility residual ‖r̄‖
            'mu_history'        : list of complementarity gaps per iteration
            'residual_history'  : list of ‖r̄‖ per iteration
        """
        Q = np.asarray(Q, dtype=float)
        c = np.asarray(c, dtype=float)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        nz = Q.shape[0]

        # ---- Step 1: QP → LCP ----------------------------------------
        M_raw, q_raw = qp_to_lcp(Q, c, A, b)
        n = M_raw.shape[0]

        # ---- Step 2: Scaling (Eq. 28) -----------------------------------
        e = np.ones(n)
        Me = M_raw @ e
        sigma = max(1.0,
                    float(np.linalg.norm(Me + q_raw, np.inf)),
                    float(-e @ Me - e @ q_raw))
        M = M_raw / sigma
        q = q_raw / sigma

        # ---- Step 3: Algorithm parameters -------------------------------
        beta = math.sqrt(2) - 1          # ≈ 0.414213
        eta = beta / math.sqrt(n + 1)
        gamma = 1.0 - eta

        # ---- Step 4: Exact iteration count (Theorem 3) ------------------
        N = exact_iteration_count(n, self.epsilon)

        # ---- Step 5: Initialisation (Eq. 29) ----------------------------
        #   x̄⁰ = col(e, 1),  s̄⁰ = col(e, 1)
        x = np.ones(n)
        tau = 1.0
        s = np.ones(n)
        kappa = 1.0

        mu_history = []
        residual_history = []

        # ---- Step 6: N iterations of the discrete dynamical system ------
        for _ in range(N):
            x_bar = np.append(x, tau)
            s_bar = np.append(s, kappa)

            # Check early convergence
            psi_cur = _psi(x, tau, M, q)
            r_bar = s_bar - psi_cur
            gap = float(x_bar @ s_bar)
            res_norm = float(np.linalg.norm(r_bar))

            mu_history.append(gap / (n + 1))
            residual_history.append(res_norm)

            if gap <= self.epsilon and res_norm <= self.epsilon:
                break

            x, tau, s, kappa, _, _ = _newton_step(
                x, tau, s, kappa, M, q, eta, gamma
            )

        # Final metrics after last iteration
        x_bar = np.append(x, tau)
        s_bar = np.append(s, kappa)
        duality_gap = float(x_bar @ s_bar)
        psi_final = _psi(x, tau, M, q)
        residual_norm = float(np.linalg.norm(s_bar - psi_final))

        # ---- Step 7: Decode HLCP solution (Lemma 6) ---------------------
        if tau >= kappa:
            # QP is feasible; z* = x[:nz] / τ (un-scaled in decision space)
            z_opt = x[:nz] / tau
            obj = 0.5 * float(z_opt @ (Q @ z_opt)) + float(c @ z_opt)
            status = 'optimal'
        else:
            # QP is infeasible; x/κ is an infeasibility certificate
            z_opt = None
            obj = None
            status = 'infeasible'

        return {
            'status': status,
            'solution': z_opt,
            'objective': obj,
            'iterations': len(mu_history),
            'planned_iterations': N,
            'duality_gap': duality_gap,
            'residual_norm': residual_norm,
            'mu_history': mu_history,
            'residual_history': residual_history,
        }
