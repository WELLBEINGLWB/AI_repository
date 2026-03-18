# AI_repository

## EIQP Algorithm — Comprehensive Technical Summary

This repository contains a detailed PDF document that summarises the **Enhanced Interior-point Quadratic Programming (EIQP)** algorithm step by step.

### 📄 Document

**[EIQP_Algorithm_Summary.pdf](./EIQP_Algorithm_Summary.pdf)**

The document covers:

1. **Introduction & Motivation** — Where and why EIQP is used (MPC, SVMs, robotics, portfolio optimisation)
2. **Mathematical Problem Formulation** — Canonical QP form, variable definitions, KKT conditions
3. **Algorithm Overview** — Predictor–corrector (Mehrotra) strategy
4. **Step-by-Step Algorithm Description**
   - Step 1: Problem Formulation
   - Step 2: Initialise Variables (Mehrotra heuristic / warm start)
   - Step 3: Evaluate Objective & Constraints (residuals, duality gap)
   - Step 4: Compute the Search Direction (predictor & corrector phases)
   - Step 5: Line Search & Variable Update (fraction-to-boundary rule)
   - Step 6: Update Barrier Parameter (adaptive μ schedule)
   - Step 7: Convergence Check & Output
5. **Flowchart Diagram** — Visual summary of the iteration loop
6. **Convergence Properties** — Rate, complexity, global convergence
7. **Practical Insights & Implementation Tips** — Sparse factorisation, scaling, regularisation, warm starting
8. **Worked Example** — 2-variable QP traced through 3 EIQP iterations to optimality
9. **Summary Table** — All 7 steps at a glance
10. **References** — Key academic papers and solver documentation
