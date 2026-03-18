# AI_repository

## Publications

### EIQP: Execution-time-certified and Infeasibility-detecting QP Solver

**Paper:** [2026_IEEETAC_EIQP.pdf](2026_IEEETAC_EIQP.pdf)

**Authors:** Liang Wu, Wei Xiao, Richard D. Braatz

**Published in:** IEEE Transactions on Automatic Control (IEEE TAC), 2026

**Abstract:**
Solving real-time quadratic programming (QP) is a ubiquitous task in control engineering, such as in model predictive control and control barrier function-based QP. This article considers convex QP (including linear programming) and adopts its homogeneous formulation to achieve infeasibility detection. Exploiting this homogeneous formulation, this article proposes a novel infeasible interior-point method (IPM) algorithm with the best theoretical O(√n) iteration complexity that feasible IPM algorithms enjoy. The iteration complexity is proved to be exact (rather than an upper bound), simple to calculate, and data independent, making it appealing to certify the execution time of online time-varying convex QPs. The proposed algorithm is simple to implement without requiring a line search procedure (uses the full Newton step), and its C-code implementation (offering MATLAB, Julia, and Python interfaces) and numerical examples are provided.

**Keywords:** Quadratic programming, execution time certificate, infeasibility detection, model predictive control, control barrier function
