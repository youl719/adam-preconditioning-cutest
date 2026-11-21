# Adam Preconditioning for Trust-Region Newton-CG on CUTEst

An empirical optimization study benchmarking adaptive diagonal preconditioners (**Adam** and **AMSGrad**) against **Identity** (unpreconditioned) and **Hessian Diagonal** preconditioning inside a Trust-Region Newton-CG (Steihaug-CG) solver across the unconstrained **CUTEst** test suite.

## Overview & Motivation
Second-order Trust-Region Newton-CG methods rely on solving a sequence of quadratic subproblems via Conjugate Gradient (CG). While exact Hessian-diagonal (Jacobi) preconditioning can accelerate CG, it can be expensive to compute or sensitive to indefiniteness and clamping. This project evaluates whether lightweight, gradient-based adaptive preconditioners (Adam and bias-corrected AMSGrad) improve convergence and oracle efficiency across a broad suite of nonlinear unconstrained optimization problems from CUTEst.

## Methodology
* **Solver:** Trust-Region Newton with Preconditioned Steihaug-CG (`solver.py`), supporting negative curvature detection and trust-region boundary truncation.
* **Preconditioners Compared:**
  * `Identity`: Standard unpreconditioned Steihaug-CG baseline.
  * `HessDiag`: Exact Hessian diagonal clamped at $\epsilon = 10^{-6}$ (accounted as 1 theoretical HVP cost per accepted step).
  * `Adam`: Exponentially weighted moving average of squared gradients with bias correction.
  * `AMSGrad`: Elementwise running maximum of the bias-corrected second-moment estimate.
* **Oracle Cost Accounting (`cutest_oracle.py`):**
  * Function evaluation: `1` unit
  * Gradient evaluation: `2` units
  * Hessian-vector product (HVP): `4` units
* **Benchmarking Protocols:**
  * **Gradient-Norm Convergence (`cutest_main.py`):** Single pass at tight tolerance ($\Vert{}\nabla f(x)\Vert{} \le 10^{-6}$, `maxite=1000`), logging full per-iteration history to extract coarser tolerances ($10^{-2}, 10^{-4}$).
  * **Function-Value Convergence (`cutest_main_fval.py`):** Runs with `gradtol=0.0` to extract the best achievable objective value ($f_{\text{best}}$) per problem for function-value performance profiles.

## Key Findings
* **Overall Performance:** ADAM and AMSGrad preconditioners did not consistently or noticeably outperform Identity, though they were much better than the HessianDiag preconditioner.
* **Inadequacy of diagonal preconditioners for ill-conditioned problems:** This may be attributable to the fact that the tested problems are ill-conditioned and contain signficant cross interactions, leading to a diagonal preconditioner being inadequate.


> **Full Report:** See [`report/`](report/) for the complete mathematical formulation, experimental setup, and analysis.

## Repository Structure

```text
adam-preconditioning-cutest/
├── cutest_problems.json                # Unconstrained CUTEst benchmark suite & sifparams
├── cutest_oracle.py                    # PyCUTEst wrapper with fine-grained oracle tracking
├── solver.py                           # Preconditioners, Steihaug-CG, and Trust-Region Newton
├── cutest_main.py                      # Main benchmark runner (gradtol = 1e-6)
├── cutest_main_fval.py                 # Extended benchmark runner (gradtol = 0.0 for f_best)
├── performance_profile_build.py        # Builds grad-norm performance profile data
├── performance_profile_build_fval.py   # Builds f-value performance profile data
├── performance_profile_plot.py         # Plots grad-norm performance profiles
├── performance_profile_plot_fval.py    # Plots f-value performance profiles
├── inspect_hessian.py                  # Spectral & clamping diagnostics at x_0
├── results/                            # Generated performance profile plots
├── report/                             # Final research report (PDF)
├── requirements.txt
└── LICENSE
```

## Setup & Reproduction
> **Environment Note:** `pycutest` requires **Linux** or **WSL2 (Windows Subsystem for Linux)** with the CUTEst/SIFDecode libraries installed.

1. Clone the repository in Linux/WSL:
   ```bash
   git clone [https://github.com/youl719/adam-preconditioning-cutest.git](https://github.com/youl719/adam-preconditioning-cutest.git)
   cd adam-preconditioning-cutest
   ```
2. Install dependencies (after configuring CUTEst/PyCUTEst):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the benchmarks and generate performance profiles:
   ```bash
   python cutest_main.py
   python performance_profile_build.py
   python performance_profile_plot.py
   ```
   *(For function-value profiles, run the corresponding `*_fval.py` scripts.)*

## Attribution
This repository builds upon the solver architecture and preconditioning framework originally developed in [finn-mead/adam-preconditioning](https://github.com/finn-mead/adam-preconditioning), extending the methodology to benchmark Trust-Region Newton-CG across the CUTEst optimization test suite.
