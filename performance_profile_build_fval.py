"""
Build performance profiles from CUTEst optimization logs using the
function-value reduction criterion:

    f(x_k) <= f_best + tau * (f(x_0) - f_best)

where f_best is the best (lowest) final f-value achieved by ANY
preconditioner on that problem, and tau varies across {1e-2, 1e-4, 1e-6}.

Expected directory layout:
    cutest_results_fval/
        AKIVA_Adam.csv
        AKIVA_HessDiag.csv
        AKIVA_Identity.csv
        ...

Each CSV has columns:
    iter, f_val, grad_norm, func_oracles, grad_oracles, hvp_oracles,
    total_oracles, delta, cg_iters, cg_reason, step_accepted

Phase 1: Scan all CSVs to find f_best per problem (minimum f_val across
          all preconditioners and all iterations).

Phase 2: For each CSV, find the first iteration where
          f_val <= f_best + tau * (f_x0 - f_best)
          and record total_oracles at that row.

Additionally, for HessDiag, a "free preconditioner" variant is computed
by subtracting 4 oracle units for each accepted step (including iter 0).

Output:
    cutest_performance_profiles_fval/                          — 3-solver profiles
    cutest_performance_profiles_fval_hessianprediag_no_cost/   — 4-solver profiles

Usage:
    python performance_profile_build_fval.py
"""

import os
import csv
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "cutest_results_fval")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_fval")
OUTPUT_DIR_FREE = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_fval_hessianprediag_no_cost")
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


def get_f_best_and_f0(csv_paths: list[str]) -> tuple[float, float]:
    """
    Scan multiple CSV files for the same problem and return:
        - f_best: the minimum f_val seen across ALL iterations and ALL files
        - f_x0:   the f_val at iter 0 (taken from the first file that has it)

    f_x0 should be the same across all preconditioners since they share
    the same starting point.
    """
    f_best = float("inf")
    f_x0 = None

    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fval = float(row["f_val"])
                f_best = min(f_best, fval)
                if f_x0 is None and int(row["iter"]) == 0:
                    f_x0 = fval

    return f_best, f_x0


def extract_oracle_counts_fval(
    csv_path: str,
    tolerances: list[float],
    f_best: float,
    f_x0: float,
) -> dict[float, int | None]:
    """
    Scan a single iteration log CSV and, for each tolerance tau, return
    the total_oracles at the first row where:

        f_val <= f_best + tau * (f_x0 - f_best)

    Returns a dict mapping tau -> total_oracles (or None if never reached).

    Edge case: if f_x0 == f_best (starting point is already optimal),
    the criterion becomes f_val <= f_best, so any row achieving f_best
    will satisfy all tolerances.
    """
    result = {tol: None for tol in tolerances}
    remaining = set(tolerances)

    # Precompute thresholds
    thresholds = {tol: f_best + tol * (f_x0 - f_best) for tol in tolerances}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not remaining:
                break
            f_val = float(row["f_val"])
            total_oracles = int(row["total_oracles"])
            satisfied = [tol for tol in remaining if f_val <= thresholds[tol]]
            for tol in satisfied:
                result[tol] = total_oracles
                remaining.discard(tol)

    return result


def extract_oracle_counts_fval_free(
    csv_path: str,
    tolerances: list[float],
    f_best: float,
    f_x0: float,
) -> dict[float, int | None]:
    """
    Like extract_oracle_counts_fval, but for HessDiag with free preconditioner.

    Subtracts HESSDIAG_PRECOND_COST oracle units for each accepted step
    (including iter 0) seen up to and including the convergence row.
    """
    result = {tol: None for tol in tolerances}
    remaining = set(tolerances)
    accepted_count = 0

    thresholds = {tol: f_best + tol * (f_x0 - f_best) for tol in tolerances}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not remaining:
                break
            if row["step_accepted"].strip() == "True":
                accepted_count += 1
            f_val = float(row["f_val"])
            total_oracles = int(row["total_oracles"])
            adjusted = total_oracles - HESSDIAG_PRECOND_COST * accepted_count
            satisfied = [tol for tol in remaining if f_val <= thresholds[tol]]
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
        print("       Run cutest_main_fval.py first to generate results.")
        return

    if not os.path.isfile(PROBLEMS_JSON):
        print(f"ERROR: Problems JSON not found: {PROBLEMS_JSON}")
        print("       Place cutest_problems.json next to this script.")
        return

    problem_names = load_problem_names(PROBLEMS_JSON)
    print(f"Loaded {len(problem_names)} problem names from {PROBLEMS_JSON}\n")

    # -----------------------------------------------------------------
    # Phase 1: Compute f_best and f_x0 for each problem
    # -----------------------------------------------------------------
    print("Phase 1: Computing f_best per problem (across all preconditioners)...")
    f_best_map: dict[str, float] = {}
    f_x0_map: dict[str, float] = {}

    for prob in problem_names:
        csv_paths = [
            os.path.join(RESULTS_DIR, f"{prob}_{prec}.csv")
            for prec in PRECONDITIONERS
        ]
        f_best, f_x0 = get_f_best_and_f0(csv_paths)

        if f_x0 is None:
            print(f"  WARNING: No iter-0 data found for {prob}, skipping.")
            continue

        f_best_map[prob] = f_best
        f_x0_map[prob] = f_x0

    print(f"  Computed f_best for {len(f_best_map)} problems.\n")

    # -----------------------------------------------------------------
    # Phase 2: Extract oracle counts using f-value criterion
    # -----------------------------------------------------------------
    print("Phase 2: Extracting oracle counts with f-value criterion...")
    oracle_data: dict[str, dict[str, dict[float, int | None]]] = {}

    for prob in problem_names:
        if prob not in f_best_map:
            continue

        f_best = f_best_map[prob]
        f_x0 = f_x0_map[prob]
        oracle_data[prob] = {}

        for prec in PRECONDITIONERS:
            csv_path = os.path.join(RESULTS_DIR, f"{prob}_{prec}.csv")
            if not os.path.isfile(csv_path):
                oracle_data[prob][prec] = {tol: None for tol in TOLERANCES}
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = {tol: None for tol in TOLERANCES}
                continue
            try:
                oracle_data[prob][prec] = extract_oracle_counts_fval(
                    csv_path, TOLERANCES, f_best, f_x0
                )
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = extract_oracle_counts_fval_free(
                        csv_path, TOLERANCES, f_best, f_x0
                    )
            except Exception as e:
                print(f"  WARNING: Could not read {csv_path}: {e}")
                oracle_data[prob][prec] = {tol: None for tol in TOLERANCES}
                if prec == "HessDiag":
                    oracle_data[prob]["HessDiagFree"] = {tol: None for tol in TOLERANCES}

    # Use only problems that have f_best computed
    valid_problems = [p for p in problem_names if p in f_best_map]
    print(f"  Processed {len(valid_problems)} problems.\n")

    # -----------------------------------------------------------------
    # Phase 3: Build and write profiles
    # -----------------------------------------------------------------

    # --- Standard 3-solver profiles ---
    print("=" * 60)
    print("Standard profiles (3 solvers) — f-value criterion")
    print("=" * 60)
    for tol in TOLERANCES:
        tol_label = f"{tol:.0e}"
        print(f"Building profile for tau = {tol_label} ...")
        rows = build_profile_for_tol(tol, valid_problems, oracle_data, PRECONDITIONERS)
        out_path = write_profile_csv(tol, rows, PRECONDITIONERS, OUTPUT_DIR)

        n_total = len(rows)
        converged_counts = {prec: sum(1 for r in rows if r[prec] != INF_LABEL) for prec in PRECONDITIONERS}
        print(f"  Written to: {out_path}")
        print(f"  Problems: {n_total}")
        for prec in PRECONDITIONERS:
            print(f"    {prec:>12s}: {converged_counts[prec]:>3d}/{n_total} converged")
        print()

    # --- 4-solver profiles (with HessDiagFree) ---
    print("=" * 60)
    print("Free-preconditioner profiles (4 solvers) — f-value criterion")
    print("=" * 60)
    for tol in TOLERANCES:
        tol_label = f"{tol:.0e}"
        print(f"Building profile for tau = {tol_label} ...")
        rows = build_profile_for_tol(tol, valid_problems, oracle_data, PRECONDITIONERS_FREE)
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