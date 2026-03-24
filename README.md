# AI_repository

## Finite-Time Discrete Dynamical Systems for QP Solving (EIQP)

This repository accompanies the paper:

> **EIQP: Execution-time-certified and Infeasibility-detecting QP Solver**  
> L. Wu, W. Xiao, R. D. Braatz  
> *IEEE Transactions on Automatic Control*, 2025.  
> DOI: [10.1109/TAC.2025.3631575](https://doi.org/10.1109/TAC.2025.3631575)

### What is a finite-time discrete dynamical system?

A **finite-time discrete dynamical system** is a discrete-time map

```
x̄_{k+1} = F(x̄_k)
```

that is guaranteed to reach a target set in a **finite, certifiable** number
of steps N — independent of the problem data (only dimension-dependent).

The EIQP algorithm is exactly such a system. Starting from a known initial
point, each iteration is one application of the map F (a full Newton step),
and the algorithm converges to the optimal QP solution *or* to an
infeasibility certificate in **exactly**

```
N = ceil( log((n+1)/ε) / (−log(1 − (√2−1)/√(n+1))) )
```

iterations, where n is the problem dimension and ε is the target optimality
level. This is the best-known O(√n) complexity, and it is an *exact* count
rather than an upper bound.

### Files

| File | Description |
|------|-------------|
| `finite_time_dds.py` | Core implementation: EIQP solver as a finite-time discrete dynamical system |
| `examples/finite_time_dds_demo.py` | Demonstrations: feasible QP, infeasible QP, time-varying QP, iteration scaling |
| `tests/test_finite_time_dds.py` | Unit tests (38 tests) |
| `2026_IEEETAC_EIQP.pdf` | Full paper |

### Quick start

```python
import numpy as np
from finite_time_dds import FiniteTimeDDS

# min  0.5*(x1² + x2²)   s.t.  x1 + x2 ≥ 1,  x ≥ 0
Q = np.eye(2)
c = np.zeros(2)
A = np.array([[1.0, 1.0]])
b = np.array([1.0])

solver = FiniteTimeDDS(epsilon=1e-8)
result = solver.solve(Q, c, A, b)

print(result['status'])             # 'optimal'
print(result['solution'])           # [0.5, 0.5]
print(result['planned_iterations']) # exact N iterations guaranteed
```

Run the demo:

```bash
python examples/finite_time_dds_demo.py
```

Run the tests:

```bash
python -m pytest tests/test_finite_time_dds.py -v
```
