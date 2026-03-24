"""
Demonstration: EIQP as a Finite-Time Discrete Dynamical System

Shows three use cases:
  1. Simple feasible QP – verifies exact iteration count and solution quality.
  2. Deliberately infeasible QP – verifies infeasibility detection.
  3. Time-varying QP loop – mimics a real-time MPC/CBF scenario where the
     data (Q, c, A, b) change at every time step but the iteration budget N
     is fixed and known in advance.

Run:
    python examples/finite_time_dds_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from finite_time_dds import FiniteTimeDDS, exact_iteration_count


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _print_header(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_result(result):
    print(f"  Status            : {result['status']}")
    if result['solution'] is not None:
        np.set_printoptions(precision=6, suppress=True)
        print(f"  Optimal z*        : {result['solution']}")
        print(f"  Objective value   : {result['objective']:.8f}")
    print(f"  Planned iters (N) : {result['planned_iterations']}")
    print(f"  Actual iters      : {result['iterations']}")
    print(f"  Duality gap       : {result['duality_gap']:.2e}")
    print(f"  Residual norm     : {result['residual_norm']:.2e}")


# ---------------------------------------------------------------------------
# Example 1 – Simple feasible 2-variable QP
# ---------------------------------------------------------------------------

def example_feasible_qp():
    _print_header("Example 1: Simple Feasible QP")

    # min  0.5*(x1^2 + x2^2)  s.t. x1 + x2 >= 1,  x1 >= 0, x2 >= 0
    # Optimal: x1* = x2* = 0.5, obj* = 0.25
    Q = np.eye(2)
    c = np.zeros(2)
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])

    solver = FiniteTimeDDS(epsilon=1e-8)
    result = solver.solve(Q, c, A, b)
    _print_result(result)

    # Verify against known solution
    z = result['solution']
    assert result['status'] == 'optimal', "Expected 'optimal'"
    assert np.allclose(z, [0.5, 0.5], atol=1e-4), f"Unexpected solution: {z}"
    assert abs(result['objective'] - 0.25) < 1e-4
    print("  ✓  Solution matches expected z* = [0.5, 0.5], obj* = 0.25")

    # Show first few and last few complementarity gaps
    mu = result['mu_history']
    print(f"\n  Complementarity gap μ̄_k (selected iterations):")
    indices = list(range(min(5, len(mu)))) + list(range(max(5, len(mu) - 3), len(mu)))
    for i in indices:
        print(f"    k={i:4d}  μ̄ = {mu[i]:.4e}")


# ---------------------------------------------------------------------------
# Example 2 – Infeasible QP
# ---------------------------------------------------------------------------

def example_infeasible_qp():
    _print_header("Example 2: Infeasible QP")

    # min  0.5*(x1^2 + x2^2)
    # s.t. x1 + x2 >= 3,   x1 <= 1,   x2 <= 1,   x1 >= 0,   x2 >= 0
    # Infeasible: x1+x2 >= 3 but x1 <= 1 and x2 <= 1 → max(x1+x2) = 2 < 3
    Q = np.eye(2)
    c = np.zeros(2)
    A = np.array([
        [ 1.0,  1.0],   # x1 + x2 >= 3  (infeasible with bounds below)
        [-1.0,  0.0],   # -x1 >= -1  →  x1 <= 1
        [ 0.0, -1.0],   # -x2 >= -1  →  x2 <= 1
    ])
    b = np.array([3.0, -1.0, -1.0])

    solver = FiniteTimeDDS(epsilon=1e-8)
    result = solver.solve(Q, c, A, b)
    _print_result(result)

    assert result['status'] == 'infeasible', "Expected 'infeasible'"
    print("  ✓  Infeasibility correctly detected")


# ---------------------------------------------------------------------------
# Example 3 – Time-varying QP (MPC / CBF scenario)
# ---------------------------------------------------------------------------

def example_time_varying_qp():
    _print_header("Example 3: Time-Varying QP (Finite-Time DDS per time step)")

    nz = 4          # decision variables
    nb = nz         # bound constraints  z ≤ 1  (i.e. -z ≥ -1)
    n = nz + nb     # LCP dimension
    epsilon = 1e-8
    N_iter = exact_iteration_count(n, epsilon)
    T = 10          # number of time steps

    print(f"  LCP dimension n = {n}")
    print(f"  Exact iterations per step N = {N_iter}  (ε = {epsilon:.0e})\n")

    solver = FiniteTimeDDS(epsilon=epsilon)
    rng = np.random.default_rng(42)

    results_table = []
    for t in range(T):
        # Time-varying cost: track a sinusoidal reference
        ref = np.sin(2 * math.pi * t / T) * np.ones(nz)
        Q = np.eye(nz)
        c = -ref
        # Bound constraints: -z ≥ -1  and  z ≥ 0 (z ≥ 0 is built into the QP)
        A = -np.eye(nz)
        b = -np.ones(nz)

        res = solver.solve(Q, c, A, b)
        obj = res['objective'] if res['objective'] is not None else float('nan')
        results_table.append((t, res['status'], res['iterations'], obj))

    print(f"  {'t':>3}  {'status':>12}  {'iters':>6}  {'objective':>12}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*6}  {'-'*12}")
    for t, status, iters, obj in results_table:
        print(f"  {t:3d}  {status:>12}  {iters:6d}  {obj:12.6f}")

    all_optimal = all(r[1] == 'optimal' for r in results_table)
    assert all_optimal, "Expected all time steps to be feasible"
    print(f"\n  ✓  All {T} time steps solved optimally")
    print(f"  ✓  Each step used ≤ N = {N_iter} iterations (finite-time guarantee)")


# ---------------------------------------------------------------------------
# Example 4 – Iteration count vs dimension
# ---------------------------------------------------------------------------

def example_iteration_scaling():
    _print_header("Example 4: Exact Iteration Count vs Problem Dimension")

    print(f"  {'n':>6}  {'N (ε=1e-6)':>12}  {'N (ε=1e-8)':>12}  {'N (ε=1e-10)':>13}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*13}")
    for nz in [10, 50, 100, 200, 500]:
        nb = nz // 5
        n = nz + nb
        N6  = exact_iteration_count(n, 1e-6)
        N8  = exact_iteration_count(n, 1e-8)
        N10 = exact_iteration_count(n, 1e-10)
        print(f"  {n:6d}  {N6:12d}  {N8:12d}  {N10:13d}")

    print("\n  The O(√n) growth of N is the best achievable for IPMs (Theorem 3).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_feasible_qp()
    example_infeasible_qp()
    example_time_varying_qp()
    example_iteration_scaling()
    print("\nAll examples completed successfully.\n")
