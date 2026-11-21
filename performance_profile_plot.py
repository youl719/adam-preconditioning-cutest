"""
Plot Dolan-Moré performance profiles from performance profile CSVs.

Generates 6 plots total:
    - 3 standard plots (Adam, HessDiag, Identity) at tolerances 1e-2, 1e-4, 1e-6
    - 3 free-preconditioner plots (Adam, HessDiag, HessDiagFree, Identity)

Expected input directories (relative to this script):
    cutest_performance_profiles_gradnorm/
        performance_profile_1e-02.csv
        performance_profile_1e-04.csv
        performance_profile_1e-06.csv
    cutest_performance_profiles_gradnorm_hessianprediag_no_cost/
        performance_profile_1e-02.csv
        performance_profile_1e-04.csv
        performance_profile_1e-06.csv

Output: PNG plots saved to cutest_performance_profile_plots/

Usage:
    python performance_profile_plot.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAD_DIR = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_gradnorm")
GRAD_FREE_DIR = os.path.join(SCRIPT_DIR, "cutest_performance_profiles_gradnorm_hessianprediag_no_cost")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "cutest_performance_profile_plots")

PRECONDITIONERS = ["Adam", "AMSGrad", "HessDiag", "Identity"]
PRECONDITIONERS_FREE = ["Adam", "AMSGrad", "HessDiag", "HessDiagFree", "Identity"]
TOLERANCES = ["1e-02", "1e-04", "1e-06"]

INF_LABEL = "Inf"

# Serif fonts (no LaTeX install required)
plt.rc("font", family="serif")
plt.rc("mathtext", fontset="dejavuserif")

# Font sizes tuned for presentations (visible from back of room)
LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 14
TICK_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
LINEWIDTH = 2.8

# Colourblind-safe palette (Okabe-Ito-inspired), each colour clearly distinct
# under deuteranopia and protanopia. Paired with varied linestyles so curves
# are still distinguishable on a B&W printout or washed-out projector.
COLORS = {
    "Adam":         "#0072B2",  # strong blue
    "AMSGrad":      "#CC79A7",  # reddish purple
    "HessDiag":     "#D55E00",  # vermillion (orange-red)
    "HessDiagFree": "#000000",  # black
    "Identity":     "#009E73",  # bluish green
}
LINESTYLES = {
    "Adam":         "-",
    "AMSGrad":      "-",
    "HessDiag":     "-",
    "HessDiagFree": "-",
    "Identity":     "-",
}
LABELS = {
    "Adam": "Adam",
    "AMSGrad": "AMSGrad",
    "HessDiag": "HessDiag",
    "HessDiagFree": "HessDiag (free precond)",
    "Identity": "Identity",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ratios(csv_path: str, preconditioners: list[str]) -> dict[str, list[float]]:
    """
    Load ratio columns from a performance profile CSV.

    Returns a dict mapping preconditioner name to a list of ratios.
    Inf entries are kept as np.inf.
    """
    ratios = {p: [] for p in preconditioners}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for prec in preconditioners:
                val = row[f"{prec}_ratio"]
                if val == INF_LABEL:
                    ratios[prec].append(np.inf)
                else:
                    ratios[prec].append(float(val))

    return ratios


def compute_profile(ratios: list[float], tau_values: np.ndarray) -> np.ndarray:
    """
    Compute the performance profile curve.

    For each tau in tau_values, returns the fraction of problems
    where the ratio <= tau.
    """
    ratios_arr = np.array(ratios)
    n = len(ratios_arr)
    fractions = np.array([np.sum(ratios_arr <= tau) / n for tau in tau_values])
    return fractions


def plot_profile(
    ratios: dict[str, list[float]],
    preconditioners: list[str],
    title: str,
    output_path: str,
):
    """
    Generate and save a single performance profile plot.
    """
    all_finite = []
    for prec in preconditioners:
        all_finite.extend([r for r in ratios[prec] if np.isfinite(r)])

    if not all_finite:
        print(f"  WARNING: No finite ratios found, skipping plot: {title}")
        return

    max_ratio = max(all_finite)
    x_max = max(2, 2 ** np.ceil(np.log2(max_ratio + 0.1)))

    tau_values = np.concatenate([
        [1.0],
        np.logspace(0, np.log10(x_max), 1000)[1:]
    ])

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))

    for prec in preconditioners:
        fractions = compute_profile(ratios[prec], tau_values)
        ax.step(
            tau_values, fractions,
            where="post",
            label=LABELS[prec],
            color=COLORS[prec],
            linestyle=LINESTYLES[prec],
            linewidth=LINEWIDTH,
            solid_capstyle="round",
            dash_capstyle="round",
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))
    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel(r"Performance ratio", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"Fraction of problems solved", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    legend = ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="lower right",
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.8)

    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    # Save both PDF (for LaTeX report / Beamer) and PNG (for PowerPoint/Keynote)
    base, _ = os.path.splitext(output_path)
    pdf_path = base + ".pdf"
    png_path = base + ".png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Standard 3-solver profiles ---
    if os.path.isdir(GRAD_DIR):
        print("Plotting standard grad_norm performance profiles (3 solvers):")
        for tol in TOLERANCES:
            csv_path = os.path.join(GRAD_DIR, f"performance_profile_{tol}.csv")
            if not os.path.isfile(csv_path):
                print(f"  Skipping (not found): {csv_path}")
                continue

            ratios = load_ratios(csv_path, PRECONDITIONERS)
            title = rf"Performance profile: $\|\nabla f\| \leq {tol}$"
            out_path = os.path.join(OUTPUT_DIR, f"profile_gradnorm_{tol}.png")
            plot_profile(ratios, PRECONDITIONERS, title, out_path)
        print()
    else:
        print(f"WARNING: Grad norm directory not found: {GRAD_DIR}\n")

    # --- 4-solver profiles (with HessDiagFree) ---
    if os.path.isdir(GRAD_FREE_DIR):
        print("Plotting free-preconditioner grad_norm performance profiles (4 solvers):")
        for tol in TOLERANCES:
            csv_path = os.path.join(GRAD_FREE_DIR, f"performance_profile_{tol}.csv")
            if not os.path.isfile(csv_path):
                print(f"  Skipping (not found): {csv_path}")
                continue

            ratios = load_ratios(csv_path, PRECONDITIONERS_FREE)
            title = rf"Performance profile: $\|\nabla f\| \leq {tol}$ (free precond)"
            out_path = os.path.join(OUTPUT_DIR, f"profile_gradnorm_hessianprediag_no_cost_{tol}.png")
            plot_profile(ratios, PRECONDITIONERS_FREE, title, out_path)
        print()
    else:
        print(f"WARNING: Free-preconditioner directory not found: {GRAD_FREE_DIR}\n")

    print("Done.")


if __name__ == "__main__":
    main()