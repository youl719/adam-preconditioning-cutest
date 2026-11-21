"""
Build performance profiles from CUTEst optimization logs.

Expected directory layout (flat, from the updated cutest_main.py):
    cutest_results/
        AKIVA_Adam.csv
        AKIVA_HessDiag.csv
        AKIVA_Identity.csv
        ALLINITU_Adam.csv
        ...

Each CSV has columns:
    iter, f_val, grad_norm, func_oracles, grad_oracles, hvp_oracles,
    total_oracles, delta, cg_iters, cg_reason, step_accepted

The script scans each CSV to find the first iteration where grad_norm
drops to or below each tolerance threshold (1e-2, 1e-4, 1e-6), and
reads the total_oracles at that row. If the threshold is never reached,
the run is marked as failed (Inf).

Additionally, for HessDiag, a "free preconditioner" variant is computed
by subtracting 4 oracle units for each accepted step (including iter 0),
reflecting the cost if the diagonal Hessian build were free.

Output:
    cutest_performance_profiles_gradnorm/                          — 3-solver profiles
    cutest_performance_profiles_gradnorm_hessianprediag_no_cost/   — 4-solver profiles (includes HessDiagFree)

Usage:
    python performance_profile_build.py

    Run from the directory containing cutest_results/, i.e.:
        ~/adam-preconditioning/
"""

import os
import csv
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "cutest_results")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_gradnorm")
OUTPUT_DIR_FREE = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_gradnorm_hessianprediag_no_cost")
PROBLEMS_JSON = os.path.join(SCRIPT_DIR, "cutest_problems.json")

PRECONDITIONERS = ["Adam", "AMSGrad", "HessDiag", "Identity"]
PRECONDITIONERS_FREE = ["Adam", "AMSGrad", "HessDiag", "HessDiagFree", "Identity"]
TOLERANCES = [1e-02, 1e-04, 1e-06]

HESSDIAG_PRECOND_COST = 4  # oracle units per preconditioner rebuild

INF_LABEL = "Inf"  # Label for failed (non-converged) runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_problem_names(json_path: str) -> list[str]:
    """Return the ordered list of problem names from the JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [entry["name"] for entry in data["UNCONSTRAINED"]]


def extract_oracle_counts(csv_path: str, tolerances: list[float]) -> dict[float, int | None]:
    """
    Scan a single iteration log CSV and, for each tolerance, return the
    total_oracles at the first row where grad_norm <= tolerance.

    Returns a dict mapping tolerance -> total_oracles (or None if never reached).
    """
    result = {tol: None for tol in tolerances}
    remaining = set(tolerances)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not remaining:
                break
            grad_norm = float(row["grad_norm"])
            total_oracles = int(row["total_oracles"])
            satisfied = [tol for tol in remaining if grad_norm <= tol]
            for tol in satisfied:
                result[tol] = total_oracles
                remaining.discard(tol)

    return result


def extract_oracle_counts_free(csv_path: str, tolerances: list[float]) -> dict[float, int | None]:
    """
    Like extract_oracle_counts, but for HessDiag with free preconditioner.

    Subtracts HESSDIAG_PRECOND_COST oracle units for each accepted step
    (including iter 0) seen up to and including the convergence row.
    """
    result = {tol: None for tol in tolerances}
    remaining = set(tolerances)
    accepted_count = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not remaining:
                break
            if row["step_accepted"].strip() == "True":
                accepted_count += 1
            grad_norm = float(row["grad_norm"])
            total_oracles = int(row["total_oracles"])
            adjusted = total_oracles - HESSDIAG_PRECOND_COST * accepted_count
            satisfied = [tol for tol in remaining if grad_norm <= tol]
            for tol in satisfied:
                result[tol] = adjusted
                remaining.discard(tol)

    return result


def build_profile_for_tol(
    tol: float,
    problem_names: list[str],
    oracle_data: dict[str, dict[str, dict[float, int | None]]],
    preconditioners: list[str],
) -> list[dict]:
    """
    Build the performance profile table for one tolerance level.

    oracle_data is structured as:
        oracle_data[problem][preconditioner][tolerance] -> total_oracles or None

    Returns a list of row dicts with keys for each preconditioner,
    Fastest, and ratio columns.
    """
    rows = []

    for prob in problem_names:
        row = {"Problem": prob}
        oracles = {}

        for prec in preconditioners:
            val = oracle_data.get(prob, {}).get(prec, {}).get(tol, None)
            oracles[prec] = val
            row[prec] = val if val is not None else INF_LABEL

        converged_vals = [v for v in oracles.values() if v is not None]

        if converged_vals:
            fastest = min(converged_vals)
            row["Fastest"] = fastest

            for prec in preconditioners:
                if oracles[prec] is not None:
                    row[f"{prec}_ratio"] = round(oracles[prec] / fastest, 4)
                else:
                    row[f"{prec}_ratio"] = INF_LABEL
        else:
            row["Fastest"] = INF_LABEL
            for prec in preconditioners:
                row[f"{prec}_ratio"] = INF_LABEL

        rows.append(row)

    return rows


def write_profile_csv(
    tol: float,
    rows: list[dict],
    preconditioners: list[str],
    output_dir: str,
) -> str:
    """Write the performance profile to a CSV file. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    tol_label = f"{tol:.0e}"
    out_path = os.path.join(output_dir, f"performance_profile_{tol_label}.csv")

    fieldnames = (
        ["Problem"]
        + preconditioners
        + ["Fastest"]
        + [f"{p}_ratio" for p in preconditioners]
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        print("       Run this script from the directory containing cutest_results/")
        return

    if not os.path.isfile(PROBLEMS_JSON):
        print(f"ERROR: Problems JSON not found: {PROBLEMS_JSON}")
        print("       Place cutest_problems.json next to this script.")
        return

    problem_names = load_problem_names(PROBLEMS_JSON)
    print(f"Loaded {len(problem_names)} problem names from {PROBLEMS_JSON}\n")

    # --- Single pass: read each CSV once, extract all tolerances ---
    # oracle_data[problem][preconditioner][tolerance] -> total_oracles or None
    oracle_data: dict[str, dict[str, dict[float, int | None]]] = {}

    for prob in problem_names:
        oracle_data[prob] = {}
        for prec in PRECONDITIONERS:
            csv_path = os.path.join(RESULTS_DIR, f"{prob}_{prec}.csv")
            if not os.path.isfile(csv_path):
                oracle_data[prob][prec] = {tol: None for tol in TOLERANCES}
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = {tol: None for tol in TOLERANCES}
                continue
            try:
                oracle_data[prob][prec] = extract_oracle_counts(csv_path, TOLERANCES)
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = extract_oracle_counts_free(
                        csv_path, TOLERANCES
                    )
            except Exception as e:
                print(f"  WARNING: Could not read {csv_path}: {e}")
                oracle_data[prob][prec] = {tol: None for tol in TOLERANCES}
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = {tol: None for tol in TOLERANCES}

    # --- Build and write standard 3-solver profiles ---
    print("=" * 60)
    print("Standard profiles (3 solvers)")
    print("=" * 60)
    for tol in TOLERANCES:
        tol_label = f"{tol:.0e}"
        print(f"Building profile for gradtol = {tol_label} ...")
        rows = build_profile_for_tol(tol, problem_names, oracle_data, PRECONDITIONERS)
        out_path = write_profile_csv(tol, rows, PRECONDITIONERS, OUTPUT_DIR)

        n_total = len(rows)
        converged_counts = {prec: sum(1 for r in rows if r[prec] != INF_LABEL) for prec in PRECONDITIONERS}
        print(f"  Written to: {out_path}")
        print(f"  Problems: {n_total}")
        for prec in PRECONDITIONERS:
            print(f"    {prec:>12s}: {converged_counts[prec]:>3d}/{n_total} converged")
        print()

    # --- Build and write 4-solver profiles (with HessDiagFree) ---
    print("=" * 60)
    print("Free-preconditioner profiles (4 solvers)")
    print("=" * 60)
    for tol in TOLERANCES:
        tol_label = f"{tol:.0e}"
        print(f"Building profile for gradtol = {tol_label} ...")
        rows = build_profile_for_tol(tol, problem_names, oracle_data, PRECONDITIONERS_FREE)
        out_path = write_profile_csv(tol, rows, PRECONDITIONERS_FREE, OUTPUT_DIR_FREE)

        n_total = len(rows)
        converged_counts = {prec: sum(1 for r in rows if r[prec] != INF_LABEL) for prec in PRECONDITIONERS_FREE}
        print(f"  Written to: {out_path}")
        print(f"  Problems: {n_total}")
        for prec in PRECONDITIONERS_FREE:
            print(f"    {prec:>16s}: {converged_counts[prec]:>3d}/{n_total} converged")
        print()

    print("Done.")


if __name__ == "__main__":
    main()