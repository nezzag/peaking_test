"""
Testing the standard slope test vs. the Mahalanobis test

For each of 4 noise methods × 27 trend combinations (3-year test period,
each year can decline at 0%, 1%, or 2%), runs:
  - standard slope bootstrap test
  - Mahalanobis test

and collects the p-values into a single DataFrame.
"""

import sys
import os
import itertools

import numpy as np
import pandas as pd

# Scripts directory is expected to be the working directory when run,
# or we add it to the path so imports resolve correctly.
sys.path.insert(0, os.path.dirname(__file__))

from peak_tester import EmissionsPeakTest
from mahalanobis_test import MahalanobisTest
from helper_functions import HiddenPrints

# ── Configuration ───────────────────────────────────────────────────────────

HIST_DATA = "gcb_hist_co2.csv"
REGION = "WLD"
YEAR_RANGE = range(1970, 2025)

# Test years (one period beyond historical data)
TEST_YEARS = [2025, 2026, 2027]

# Noise characterisation methods to evaluate
METHODS = {
    "hamilton": {},
    "linear_w_autocorrelation": {},
    "broken_trend": {},
    "spline": {"n_knots": 15},
}

# Per-year decline rates to explore (0 %, 1 %, 2 %)
DECLINE_RATES = [0.00, 0.01, 0.02]

N_BOOTSTRAP = 10_000
NULL_HYPOTHESIS = "zero_trend"
BOOTSTRAP_METHOD = "ar_bootstrap"

OUTPUT_PATH = "../notebooks/results/emissions_grid_pvalues_2020included.csv"

# Set True to load an existing CSV instead of re-running the full grid
REUSE_RESULTS = True

# ── Helpers ──────────────────────────────────────────────────────────────────


def build_test_data(base_value: float, rates: tuple) -> list:
    """Return list of (year, value) tuples for the 3 test years.

    Each year's value declines from the previous year by the corresponding
    rate in *rates* (r1, r2, r3 applied to years 2025, 2026, 2027).
    """
    values = []
    prev = base_value
    for year, rate in zip(TEST_YEARS, rates):
        val = prev * (1.0 - rate)
        values.append((year, val))
        prev = val
    return values


def decline_label(rates: tuple) -> str:
    """Human-readable label, e.g. '0%-1%-2%'."""
    return "-".join(f"{int(r*100)}%" for r in rates)


# ── Main grid loop ───────────────────────────────────────────────────────────


def run_grid() -> pd.DataFrame:
    # Initialise and load historical data once
    base_test = EmissionsPeakTest()
    base_test.load_historical_data(HIST_DATA, region=REGION, year_range=YEAR_RANGE)

    base_value = float(base_test.historical_data.iloc[-1, 1])
    print(f"Base emissions value (last historical year): {base_value:.1f} Mt CO₂")

    # All 3^3 = 27 rate combinations
    rate_combos = list(itertools.product(DECLINE_RATES, repeat=3))

    records = []

    total = len(METHODS) * len(rate_combos)
    done = 0

    for method_name, method_kwargs in METHODS.items():
        for rates in rate_combos:
            done += 1
            r1, r2, r3 = rates
            label = decline_label(rates)
            avg_decline = np.mean(rates) * 100
            print(
                f"[{done:3d}/{total}] method={method_name:25s}  rates={label}", end="  "
            )

            test_data = build_test_data(base_value, rates)

            # Build a fresh tester for each combination so state never leaks
            tester = EmissionsPeakTest()
            tester.load_historical_data(HIST_DATA, region=REGION, year_range=YEAR_RANGE)
            tester.set_test_data(test_data)

            slope_p = np.nan
            mahal_p = np.nan

            try:
                with HiddenPrints():
                    tester.characterize_noise(
                        method=method_name,
                        include_test_data=False,
                        # ignore_years=[2020],
                        **method_kwargs,
                    )
                    tester.create_noise_generator()

                    # ── Slope / bootstrap test ──────────────────────────────
                    tester.run_complete_bootstrap_test(
                        n_bootstrap=N_BOOTSTRAP,
                        null_hypothesis=NULL_HYPOTHESIS,
                        bootstrap_method=BOOTSTRAP_METHOD,
                    )
                    slope_p = tester.bootstrap_results["p_value_one_tail"]

                    # ── Mahalanobis test ────────────────────────────────────
                    mahal = MahalanobisTest(tester)
                    mahal_results = mahal.run_test(
                        n_bootstrap=N_BOOTSTRAP,
                        null_hypothesis=NULL_HYPOTHESIS,
                        bootstrap_method=BOOTSTRAP_METHOD,
                    )
                    mahal_p = mahal_results["p_value"]

                print(f"slope_p={slope_p:.3f}  mahal_p={mahal_p:.3f}")

            except Exception as exc:
                print(f"ERROR: {exc}")

            records.append(
                {
                    "method": method_name,
                    "decline_y1_pct": int(r1 * 100),
                    "decline_y2_pct": int(r2 * 100),
                    "decline_y3_pct": int(r3 * 100),
                    "avg_decline_pct": round(avg_decline, 2),
                    "decline_label": label,
                    "slope_p_value": slope_p,
                    "mahal_p_value": mahal_p,
                }
            )

    df = pd.DataFrame(records)
    return df


# ── Analysis ─────────────────────────────────────────────────────────────────


def categorise_decline(avg_pct: float) -> str:
    """Map average decline (in %) to a named case matching the grid thresholds."""
    if avg_pct > 1:
        return "strong"
    elif avg_pct > 0.35:
        return "weak"
    elif avg_pct > 0:
        return "very_weak"
    else:
        return "none"


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return a (method × decline_case) summary table.

    Columns
    -------
    avg_slope_p   : mean slope-test p-value across scenarios in that cell
    avg_mahal_p   : mean Mahalanobis p-value
    median_ratio  : median of element-wise (mahal_p / slope_p)
    """
    df = results.copy()
    df["average_slope"] = df[
        ["decline_y1_pct", "decline_y2_pct", "decline_y3_pct"]
    ].mean(axis=1)
    df["decline_case"] = df["average_slope"].map(categorise_decline)
    df["ratio"] = df["mahal_p_value"] / df["slope_p_value"]

    summary = df.groupby(["method", "decline_case"]).agg(
        avg_slope_p=("slope_p_value", "mean"),
        avg_mahal_p=("mahal_p_value", "mean"),
        median_ratio=("ratio", "mean"),
    )

    # Enforce a meaningful row order: none < weak < strong
    case_order = pd.CategoricalDtype(
        ["none", "very_weak", "weak", "strong"], ordered=True
    )
    summary = summary.reset_index()
    summary["decline_case"] = summary["decline_case"].astype(case_order)
    summary = summary.sort_values(["decline_case"]).set_index(
        ["decline_case", "method"]
    )

    return summary


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))  # ensure relative CSV paths work

    out = os.path.join(os.path.dirname(__file__), OUTPUT_PATH)

    if REUSE_RESULTS and os.path.exists(out):
        print(f"Loading existing results from {out}")
        results = pd.read_csv(out)
    else:
        results = run_grid()
        results.to_csv(out, index=False)
        print(f"\nSaved to {out}")

    summary = summarise_results(results)

    print("\n" + "=" * 70)
    print("P-VALUE SUMMARY BY METHOD AND DECLINE CASE")
    print(
        "(avg_slope_p / avg_mahal_p = mean p-values; median_ratio = median(mahal/slope))"
    )
    print("=" * 70)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
