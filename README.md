# AI_repository

## Publications

### EIQP: Execution-time-certified and Infeasibility-detecting QP Solver

**Paper:** [2026_IEEETAC_EIQP.pdf](2026_IEEETAC_EIQP.pdf)

**Authors:** Liang Wu, Wei Xiao, Richard D. Braatz

**Published in:** IEEE Transactions on Automatic Control (IEEE TAC), 2026

**Abstract:**
Solving real-time quadratic programming (QP) is a ubiquitous task in control engineering, such as in model predictive control and control barrier function-based QP. This article considers convex QP (including linear programming) and adopts its homogeneous formulation to achieve infeasibility detection. Exploiting this homogeneous formulation, this article proposes a novel infeasible interior-point method (IPM) algorithm with the best theoretical O(√n) iteration complexity that feasible IPM algorithms enjoy. The iteration complexity is proved to be exact (rather than an upper bound), simple to calculate, and data independent, making it appealing to certify the execution time of online time-varying convex QPs. The proposed algorithm is simple to implement without requiring a line search procedure (uses the full Newton step), and its C-code implementation (offering MATLAB, Julia, and Python interfaces) and numerical examples are provided.

**Keywords:** Quadratic programming, execution time certificate, infeasibility detection, model predictive control, control barrier function

---

## Key Contributions of the EIQP Algorithm

### 1. Unified Execution-Time Certificate with Infeasibility Detection
EIQP is the first QP solver to simultaneously provide an **execution-time certificate** *and* **infeasibility detection** for general convex QP (including LP). Prior works either certified execution time but assumed feasibility, or detected infeasibility but lacked certified iteration complexity. EIQP closes this gap by adopting a **homogeneous LCP (HLCP) formulation** of the QP's KKT conditions, which always has a solution (is asymptotically feasible) regardless of whether the original QP is feasible or infeasible.

### 2. Infeasible IPM Achieving O(√n) Iteration Complexity
By exploiting the homogeneous formulation, EIQP proposes an **infeasible interior-point method (IPM)** that attains the same O(√n) iteration complexity traditionally reserved only for feasible IPMs. This is a theoretical breakthrough: infeasible IPMs normally achieve only O(n) complexity, but EIQP's homogeneous structure enables the better O(√n) bound even when starting from an infeasible point.

### 3. Exact, Data-Independent Iteration Count
The iteration complexity is **exact** (not merely an upper bound) and **data-independent** (depends only on problem dimension n and the desired optimality level ε), given by:

```
N = ⌈ log((n+1)/ε) / (−log(1 − 0.414213/√(n+1))) ⌉
```

This closed-form formula makes it straightforward to pre-compute the worst-case number of iterations — and hence the worst-case execution time — for any online time-varying QP before solving it. No other general-purpose QP solver with infeasibility detection has achieved this property.

### 4. Full Newton Step Without Line Search
EIQP uses the **full Newton step** (step size α = 1) at every iteration, eliminating the need for a line search procedure. This keeps each iteration computationally simple (a fixed number of floating-point operations), which is essential for reliably bounding execution time.

### 5. Data-Independent Scaling and Initialization
A novel **scaling strategy** (normalizing the LCP data matrix and vector) combined with a fixed **initialization** (x₀ = e, τ₀ = 1, s₀ = e, κ₀ = 1) ensures that both the initial complementarity gap and the initial infeasibility residual norm are bounded by (n+1), independent of the problem data. This is what makes the iteration count formula fully data-independent.

### 6. Open-Source C Implementation with Multi-Language Interfaces
EIQP provides a **C-code implementation** with interfaces for **MATLAB**, **Julia**, and **Python**, making it accessible across the most common scientific computing environments. The solver requires only standard LAPACK/BLAS routines.

### 7. Demonstrated on Real-Time Control Applications
The algorithm is validated on two representative real-time control benchmarks:
- **Model Predictive Control (MPC)** for an AFTI-16 aircraft pitch-angle tracking problem, certifying execution times well below the 50 ms sampling period for prediction horizons up to Nₚ = 10.
- **CLF-HOCBF-QP** for an Adaptive Cruise Control (ACC) problem with nonlinear safety constraints, where EIQP achieves the smallest standard deviation in execution time across all compared solvers (OSQP, SCSv3, Clarabel, Quadprog), enabling certified operation at up to 13.3 kHz.

### Summary Table

| Property | EIQP | Typical Feasible IPM | Typical Infeasible IPM | OSQP / Active-Set |
|---|---|---|---|---|
| Infeasibility detection | ✅ | ❌ | ✅ | ✅ / partial |
| Iteration complexity | O(√n) | O(√n) | O(n) | Not certified |
| Exact (not upper bound) | ✅ | ❌ | ❌ | ❌ |
| Data-independent | ✅ | ❌ | ❌ | ❌ |
| Line-search free | ✅ | ❌ | ❌ | ✅ |
| General convex QP + LP | ✅ | ✅ | ✅ | ✅ |
