"""
Run Trust-Region Newton-CG benchmarks WITHOUT gradient-norm stopping.

The solver exits only when:
    - max iterations reached
    - max oracle calls reached
    - trust region shrinks to zero

This produces the longest possible runs so that we can extract the best
achievable f-value (f_best) per problem across all preconditioners.

Output directory: cutest_results_fval/
"""

import csv
import json
import os
import time
import torch
import pycutest

from cutest_oracle import CutestOracle
from solver import (
    IdentityPreconditioner,
    HessianDiagPreconditioner,
    AdamPreconditioner,
    AMSGradPreconditioner,
    trust_region_newton,
)

CSV_COLUMNS = [
    "iter",
    "f_val",
    "grad_norm",
    "func_oracles",
    "grad_oracles",
    "hvp_oracles",
    "total_oracles",
    "delta",
    "cg_iters",
    "cg_reason",
    "step_accepted",
]


def load_problems(json_path: str, category: str = "UNCONSTRAINED") -> list:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get(category, [])


def write_iteration_log(log: list, filepath: str) -> None:
    """Write the per-iteration log list to a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in log:
            writer.writerow(row)


def run_benchmarks(
    json_path: str = "cutest_problems.json",
    max_problems: int = None,
    maxite: int = 1000,
    output_dir: str = "cutest_results_fval",
):
    """
    Run Trust-Region Newton-CG benchmarks with gradtol=0 so that the
    gradient-norm stopping condition is effectively disabled.

    The solver will only exit via max iterations, max oracle calls,
    or trust-region collapse. This gives us the best possible f-values
    for building function-value-based performance profiles.
    """
    problems = load_problems(json_path)
    device = torch.device("cpu")

    os.makedirs(output_dir, exist_ok=True)

    num_problems = max_problems if max_problems else len(problems)
    print(f"Loaded {len(problems)} problems.  Benchmarking {num_problems} with gradtol=0 (disabled), maxite={maxite}.")
    print(f"Per-iteration CSVs will be saved to: {os.path.abspath(output_dir)}/\n")
    print(
        f"{'Problem':<15} | {'Precond':<10} | {'Iters':<6} | "
        f"{'Oracles':<8} | {'f_val':<14} | {'g_norm':<12} | {'Time (s)':<8}"
    )
    print("-" * 90)

    for prob_info in problems[:num_problems]:
        prob_name = prob_info["name"]
        sif_params = prob_info.get("sifparams", None)

        try:
            oracle = CutestOracle(problem_name=prob_name, sifparams=sif_params)
            x0_original = oracle.get_initial_point().to(device)

            preconditioners = {
                "Identity": IdentityPreconditioner(),
                "HessDiag": HessianDiagPreconditioner(eps=1e-6),
                "Adam": AdamPreconditioner(),
                "AMSGrad": AMSGradPreconditioner(),
            }

            for p_name, precond in preconditioners.items():
                oracle.reset_counters()
                x0 = x0_original.clone()

                iteration_log: list[dict] = []

                t_start = time.time()
                x_opt, logs = trust_region_newton(
                    fun=oracle,
                    x0=x0,
                    preconditioner=precond,
                    gradtol=0.0,          # <-- disabled
                    maxite=maxite,
                    iteration_log=iteration_log,
                )
                t_end = time.time()

                iters = logs.get("k", 0)
                g_norm = logs.get("g_norm", float("inf"))
                f_val = iteration_log[-1]["f_val"] if iteration_log else float("inf")
                oracles = oracle.oracle_units

                csv_name = f"{prob_name}_{p_name}.csv"
                csv_path = os.path.join(output_dir, csv_name)
                write_iteration_log(iteration_log, csv_path)

                display_prob = prob_name if p_name == "Identity" else ""
                print(
                    f"{display_prob:<15} | {p_name:<10} | {iters:<6} | "
                    f"{oracles:<8} | {f_val:<14.6e} | {g_norm:<12.2e} | {t_end - t_start:<8.4f}"
                )

            print("-" * 90)

        except Exception as e:
            print(
                f"{prob_name:<15} | {'FAILED':<10} | {'-':<6} | "
                f"{'-':<8} | {'-':<14} | {'-':<12} | {'-':<8}"
            )
            print(f"  -> Error: {str(e)[:120]}")
            print("-" * 90)


if __name__ == "__main__":
    run_benchmarks(maxite=1000)