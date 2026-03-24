"""
Unit tests for finite_time_dds.py

Tests cover:
  - exact_iteration_count: formula consistency and scaling
  - qp_to_lcp: correct M and q construction
  - FiniteTimeDDS.solve: feasible QPs, infeasible QP, solution quality
  - Finite-time guarantee: actual iterations ≤ planned N
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from finite_time_dds import (
    FiniteTimeDDS,
    exact_iteration_count,
    qp_to_lcp,
    _psi,
    _jac_psi,
)


# ---------------------------------------------------------------------------
# Tests for exact_iteration_count
# ---------------------------------------------------------------------------

class TestExactIterationCount:
    def test_positive(self):
        N = exact_iteration_count(10, 1e-8)
        assert N > 0

    def test_increases_with_dimension(self):
        N_small = exact_iteration_count(10, 1e-8)
        N_large = exact_iteration_count(200, 1e-8)
        assert N_large > N_small

    def test_increases_with_tighter_tolerance(self):
        N_loose = exact_iteration_count(50, 1e-4)
        N_tight = exact_iteration_count(50, 1e-12)
        assert N_tight > N_loose

    def test_formula_n10_epsilon1e6(self):
        # Manual computation: β = sqrt(2)-1, η = β/sqrt(11), N = ceil(...)
        beta = math.sqrt(2) - 1
        n = 10
        eta = beta / math.sqrt(n + 1)
        expected = math.ceil(math.log((n + 1) / 1e-6) / (-math.log(1 - eta)))
        assert exact_iteration_count(n, 1e-6) == expected

    def test_returns_int(self):
        assert isinstance(exact_iteration_count(20, 1e-8), int)


# ---------------------------------------------------------------------------
# Tests for qp_to_lcp
# ---------------------------------------------------------------------------

class TestQpToLcp:
    def test_dimensions(self):
        nz, nb = 3, 2
        Q = np.eye(nz)
        c = np.zeros(nz)
        A = np.ones((nb, nz))
        b = np.zeros(nb)
        M, q = qp_to_lcp(Q, c, A, b)
        n = nz + nb
        assert M.shape == (n, n)
        assert q.shape == (n,)

    def test_q_vector(self):
        Q = np.eye(2)
        c = np.array([1.0, 2.0])
        A = np.array([[1.0, 0.0]])
        b = np.array([0.5])
        M, q = qp_to_lcp(Q, c, A, b)
        np.testing.assert_allclose(q[:2], c)
        np.testing.assert_allclose(q[2:], -b)

    def test_m_structure(self):
        Q = np.array([[2.0, 0.0], [0.0, 3.0]])
        A = np.array([[1.0, 1.0]])
        c = np.zeros(2)
        b = np.zeros(1)
        M, _ = qp_to_lcp(Q, c, A, b)
        # Top-left = Q
        np.testing.assert_allclose(M[:2, :2], Q)
        # Top-right = -A'
        np.testing.assert_allclose(M[:2, 2:], -A.T)
        # Bottom-left = A
        np.testing.assert_allclose(M[2:, :2], A)
        # Bottom-right = 0
        np.testing.assert_allclose(M[2:, 2:], np.zeros((1, 1)))

    def test_symmetric_part_is_psd(self):
        rng = np.random.default_rng(0)
        nz, nb = 4, 3
        Qraw = rng.standard_normal((nz, nz))
        Q = Qraw @ Qraw.T + np.eye(nz)
        A = rng.standard_normal((nb, nz))
        c = rng.standard_normal(nz)
        b = rng.standard_normal(nb)
        M, _ = qp_to_lcp(Q, c, A, b)
        sym = (M + M.T) / 2
        eigvals = np.linalg.eigvalsh(sym)
        assert np.all(eigvals >= -1e-10), "Symmetric part of M is not PSD"


# ---------------------------------------------------------------------------
# Tests for _psi and _jac_psi
# ---------------------------------------------------------------------------

class TestPsiAndJacobian:
    def _simple_data(self):
        M = np.array([[2.0, -1.0], [-1.0, 2.0]])
        q = np.array([0.5, -0.5])
        return M, q

    def test_psi_shape(self):
        M, q = self._simple_data()
        x = np.ones(2)
        tau = 1.0
        p = _psi(x, tau, M, q)
        assert p.shape == (3,)

    def test_jac_psi_shape(self):
        M, q = self._simple_data()
        x = np.ones(2)
        tau = 1.0
        J = _jac_psi(x, tau, M, q)
        assert J.shape == (3, 3)

    def test_jac_psi_psd(self):
        # Jψ must be PSD in R^{n+1}_{++} (Lemma 3 of the paper)
        M = np.array([[3.0, 1.0], [1.0, 3.0]])
        q = np.array([0.1, 0.2])
        x = np.array([2.0, 3.0])
        tau = 1.5
        J = _jac_psi(x, tau, M, q)
        eigvals = np.linalg.eigvalsh((J + J.T) / 2)
        assert np.all(eigvals >= -1e-10), f"Jψ is not PSD, min eig={eigvals.min()}"

    def test_numerical_jacobian(self):
        M = np.array([[2.0, 0.5], [0.5, 3.0]])
        q = np.array([0.3, -0.2])
        x = np.array([1.5, 2.0])
        tau = 0.8
        J_analytical = _jac_psi(x, tau, M, q)

        # Numerical Jacobian via finite differences
        xbar = np.append(x, tau)
        n = len(x)
        eps = 1e-6
        J_numerical = np.zeros((n + 1, n + 1))
        for j in range(n + 1):
            xbar_p = xbar.copy(); xbar_p[j] += eps
            xbar_m = xbar.copy(); xbar_m[j] -= eps
            p_p = _psi(xbar_p[:n], xbar_p[n], M, q)
            p_m = _psi(xbar_m[:n], xbar_m[n], M, q)
            J_numerical[:, j] = (p_p - p_m) / (2 * eps)

        np.testing.assert_allclose(J_analytical, J_numerical, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests for FiniteTimeDDS.solve – feasible QPs
# ---------------------------------------------------------------------------

class TestFiniteTimeDDSSolveFeasible:
    def _solver(self, eps=1e-8):
        return FiniteTimeDDS(epsilon=eps)

    def test_simple_1d(self):
        # min 0.5*x^2  s.t. x >= 1, x >= 0  →  x* = 1
        Q = np.array([[1.0]])
        c = np.array([0.0])
        A = np.array([[1.0]])
        b = np.array([1.0])
        result = self._solver().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        np.testing.assert_allclose(result['solution'], [1.0], atol=1e-4)
        assert abs(result['objective'] - 0.5) < 1e-4

    def test_simple_2d(self):
        # min 0.5*(x1^2+x2^2)  s.t. x1+x2>=1, x>=0  →  x*=[0.5,0.5]
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        result = self._solver().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        np.testing.assert_allclose(result['solution'], [0.5, 0.5], atol=1e-4)
        assert abs(result['objective'] - 0.25) < 1e-4

    def test_returns_optimal_for_unconstrained_minimum_inside_feasible(self):
        # min 0.5*||x - x_ref||^2  s.t. x >= 0, x_ref is strictly positive
        nz = 3
        x_ref = np.array([1.0, 2.0, 3.0])
        Q = np.eye(nz)
        c = -x_ref
        A = np.eye(nz)       # x >= 0 (implicit in QP form, add explicit)
        b = np.zeros(nz)
        result = self._solver().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        np.testing.assert_allclose(result['solution'], x_ref, atol=1e-4)

    def test_iterations_within_planned_budget(self):
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        result = self._solver(eps=1e-6).solve(Q, c, A, b)
        assert result['iterations'] <= result['planned_iterations']

    def test_duality_gap_below_epsilon(self):
        Q = np.eye(3)
        c = np.array([-1.0, -2.0, -3.0])
        A = np.eye(3)
        b = np.zeros(3)
        eps = 1e-8
        result = self._solver(eps=eps).solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        assert result['duality_gap'] <= eps * 10   # allow small numerical slack

    def test_residual_norm_below_epsilon(self):
        Q = np.eye(3)
        c = np.array([-1.0, -2.0, -3.0])
        A = np.eye(3)
        b = np.zeros(3)
        eps = 1e-8
        result = self._solver(eps=eps).solve(Q, c, A, b)
        assert result['residual_norm'] <= eps * 10

    def test_mu_history_decreasing(self):
        Q = np.eye(4)
        c = np.zeros(4)
        A = np.eye(4)
        b = np.zeros(4)
        result = self._solver().solve(Q, c, A, b)
        mu = result['mu_history']
        # Complementarity gap should be strictly decreasing
        for i in range(1, len(mu)):
            assert mu[i] <= mu[i - 1] + 1e-14, f"mu increased at k={i}"

    def test_random_qp(self):
        rng = np.random.default_rng(7)
        nz = 5
        Qraw = rng.standard_normal((nz, nz))
        Q = Qraw @ Qraw.T + np.eye(nz) * 0.1
        c = rng.standard_normal(nz)
        A = np.eye(nz)
        b = np.zeros(nz)
        result = self._solver().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        assert result['duality_gap'] < 1e-5


# ---------------------------------------------------------------------------
# Tests for FiniteTimeDDS.solve – infeasible QP
# ---------------------------------------------------------------------------

class TestFiniteTimeDDSSolveInfeasible:
    def test_infeasible_2d(self):
        # x1+x2 >= 3 but x1 <= 1, x2 <= 1  →  infeasible
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([
            [ 1.0,  1.0],
            [-1.0,  0.0],
            [ 0.0, -1.0],
        ])
        b = np.array([3.0, -1.0, -1.0])
        result = FiniteTimeDDS(epsilon=1e-8).solve(Q, c, A, b)
        assert result['status'] == 'infeasible'
        assert result['solution'] is None
        assert result['objective'] is None

    def test_infeasible_iterations_within_budget(self):
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[ 1.0, 1.0], [-1.0, 0.0], [ 0.0, -1.0]])
        b = np.array([3.0, -1.0, -1.0])
        result = FiniteTimeDDS(epsilon=1e-8).solve(Q, c, A, b)
        assert result['iterations'] <= result['planned_iterations']


# ---------------------------------------------------------------------------
# Test: result dict has all expected keys
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_result_keys_feasible(self):
        Q = np.eye(2); c = np.zeros(2)
        A = np.array([[1.0, 1.0]]); b = np.array([1.0])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        required = {'status', 'solution', 'objective', 'iterations',
                    'planned_iterations', 'duality_gap', 'residual_norm',
                    'mu_history', 'residual_history'}
        assert required.issubset(result.keys())

    def test_result_keys_infeasible(self):
        Q = np.eye(2); c = np.zeros(2)
        A = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        b = np.array([3.0, -1.0, -1.0])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        required = {'status', 'solution', 'objective', 'iterations',
                    'planned_iterations', 'duality_gap', 'residual_norm',
                    'mu_history', 'residual_history'}
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Tests for input handling / robustness
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_list_inputs_accepted(self):
        # Solver should accept plain Python lists, not just ndarrays
        Q = [[1.0, 0.0], [0.0, 1.0]]
        c = [0.0, 0.0]
        A = [[1.0, 1.0]]
        b = [1.0]
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'

    def test_integer_inputs_promoted_to_float(self):
        Q = np.array([[2, 0], [0, 2]])   # integer dtype
        c = np.array([0, 0])
        A = np.array([[1, 1]])
        b = np.array([1])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'

    def test_epsilon_parameter_respected(self):
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        # Looser epsilon → fewer planned iterations
        r_loose = FiniteTimeDDS(epsilon=1e-4).solve(Q, c, A, b)
        r_tight = FiniteTimeDDS(epsilon=1e-12).solve(Q, c, A, b)
        assert r_loose['planned_iterations'] < r_tight['planned_iterations']

    def test_single_constraint(self):
        # 1-variable, 1-constraint: min 0.5*x^2  s.t. x >= 2  →  x* = 2
        Q = np.array([[1.0]])
        c = np.array([0.0])
        A = np.array([[1.0]])
        b = np.array([2.0])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        np.testing.assert_allclose(result['solution'], [2.0], atol=1e-4)
        assert abs(result['objective'] - 2.0) < 1e-4

    def test_many_constraints(self):
        # 4-variable box-constrained QP
        nz = 4
        Q = np.diag([1.0, 2.0, 3.0, 4.0])
        c = np.array([-1.0, -1.0, -1.0, -1.0])
        # z >= 0 (nz constraints) and z <= 2 (-z >= -2, nz constraints)
        A = np.vstack([np.eye(nz), -np.eye(nz)])
        b = np.concatenate([np.zeros(nz), -2.0 * np.ones(nz)])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        # Unconstrained optimum: Qz* + c = 0 → z*_i = -c_i/Q_ii.
        # With c_i = -1 for all i: z*_i = 1/Q_ii ∈ [0, 2] → no active bounds
        z_expected = np.array([1.0, 0.5, 1.0/3.0, 0.25])
        np.testing.assert_allclose(result['solution'], z_expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests for convergence rate (Theorem 3 verification)
# ---------------------------------------------------------------------------

class TestConvergenceRate:
    def test_residual_history_length_matches_iterations(self):
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert len(result['residual_history']) == result['iterations']
        assert len(result['mu_history']) == result['iterations']

    def test_residual_history_decreasing(self):
        Q = np.eye(3)
        c = np.array([-2.0, -1.0, -3.0])
        A = np.eye(3)
        b = np.zeros(3)
        result = FiniteTimeDDS().solve(Q, c, A, b)
        r = result['residual_history']
        for i in range(1, len(r)):
            assert r[i] <= r[i - 1] + 1e-12, f"residual increased at k={i}"

    def test_actual_iterations_match_planned_for_default_init(self):
        # Because of the fixed initialization (x=e, s=e), the algorithm uses
        # exactly N steps for non-trivial problems (no early exit).
        Q = np.eye(3)
        c = np.array([-1.0, -2.0, -3.0])
        A = np.eye(3)
        b = np.zeros(3)
        eps = 1e-8
        result = FiniteTimeDDS(epsilon=eps).solve(Q, c, A, b)
        assert result['iterations'] == result['planned_iterations']

    def test_theorem3_exact_count_feasible(self):
        # Verify that exactly N iterations are sufficient (feasible problem).
        nz, nb = 3, 3
        n = nz + nb
        eps = 1e-6
        N_expected = exact_iteration_count(n, eps)
        Q = np.eye(nz)
        c = np.array([-1.0, -1.0, -1.0])
        A = np.eye(nz)
        b = np.zeros(nz)
        result = FiniteTimeDDS(epsilon=eps).solve(Q, c, A, b)
        assert result['planned_iterations'] == N_expected
        assert result['iterations'] <= N_expected

    def test_theorem3_exact_count_infeasible(self):
        # Verify that exactly N iterations are sufficient (infeasible problem).
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        b = np.array([3.0, -1.0, -1.0])
        n = 2 + 3   # nz + nb
        eps = 1e-8
        N_expected = exact_iteration_count(n, eps)
        result = FiniteTimeDDS(epsilon=eps).solve(Q, c, A, b)
        assert result['planned_iterations'] == N_expected
        assert result['iterations'] <= N_expected


# ---------------------------------------------------------------------------
# Regression tests – known QP instances with analytically verified solutions
# ---------------------------------------------------------------------------

class TestKnownSolutions:
    def test_equality_constraint_via_two_inequalities(self):
        # min 0.5*(x1^2 + x2^2)  s.t. x1 + x2 = 1  (encoded as >= and <=)
        # Optimal: x1* = x2* = 0.5, obj* = 0.25
        Q = np.eye(2)
        c = np.zeros(2)
        A = np.array([
            [ 1.0,  1.0],   # x1 + x2 >= 1
            [-1.0, -1.0],   # -(x1+x2) >= -1  →  x1+x2 <= 1
        ])
        b = np.array([1.0, -1.0])
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        np.testing.assert_allclose(result['solution'], [0.5, 0.5], atol=1e-4)
        assert abs(result['objective'] - 0.25) < 1e-4

    def test_larger_known_optimum(self):
        # min 0.5*z'diag(w)z  s.t. z >= l  →  z*_i = max(0, l_i)
        # Choose l = [0.5, 1.0, 0.2, 0.8], w = [2, 1, 3, 4]
        w = np.array([2.0, 1.0, 3.0, 4.0])
        l = np.array([0.5, 1.0, 0.2, 0.8])
        Q = np.diag(w)
        c = np.zeros(4)
        A = np.eye(4)
        b = l
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        # Unconstrained optimum z*=0 violates z >= l for all positive l,
        # so active constraints z* = l.
        np.testing.assert_allclose(result['solution'], l, atol=1e-4)
        obj_expected = 0.5 * float(l @ (w * l))
        assert abs(result['objective'] - obj_expected) < 1e-4

    def test_objective_value_is_nonnegative_for_psd_Q_zero_c(self):
        rng = np.random.default_rng(99)
        nz = 6
        Qraw = rng.standard_normal((nz, nz))
        Q = Qraw @ Qraw.T   # PSD
        c = np.zeros(nz)
        A = np.eye(nz)
        b = np.zeros(nz)
        result = FiniteTimeDDS().solve(Q, c, A, b)
        assert result['status'] == 'optimal'
        assert result['objective'] >= -1e-6  # obj = 0.5*z'Qz >= 0
