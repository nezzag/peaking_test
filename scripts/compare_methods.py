"""
Compare Methods
==================================

Command-line wrapper around EmissionsPeakTest.compare_methods(): for a given
emissions series, test each trend decomposition method's noise (autocorrelation,
ARCH, normality, with GESD outlier exclusion), and select one of the methods that
pass. See EmissionsPeakTest.compare_methods for details.

Usage:
    python scripts/compare_methods.py --data gcb_hist_co2.csv --region WLD \
        --variable "Emissions|CO2|Fossil" --start 1970 --end 2024 \
        [--selection priority|backtest]

From Python, method selection runs in characterize_noise():
    tester = EmissionsPeakTest()
    tester.load_historical_data(...)
    tester.characterize_noise()               # method="priority" by default, or "backtest"
    tester.method_comparison                  # per-method results
    tester.selected_method, tester.selected_ignore_years
or directly with tester.compare_methods(selection=...).
"""

import argparse
import contextlib
import io

import pandas as pd

from peak_tester import METHODS, SELECTION_OPTIONS, EmissionsPeakTest


def compare_methods(tester: EmissionsPeakTest, **kwargs) -> pd.DataFrame:
    """Kept for backwards compatibility: same as tester.compare_methods(**kwargs)."""
    return tester.compare_methods(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="gcb_hist_co2.csv",
                        help="CSV file in data/processed/")
    parser.add_argument("--region", default="WLD")
    parser.add_argument("--variable", default="Emissions|CO2|Fossil")
    parser.add_argument("--start", type=int, default=1970, help="First historical year")
    parser.add_argument("--end", type=int, default=2024, help="Last historical year (inclusive)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-exclusions", type=int, default=3)
    parser.add_argument("--n-protected", type=int, default=3,
                        help="Number of final historical years that are never auto-excluded")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--selection", choices=SELECTION_OPTIONS, default="priority",
                        help="How to choose among passing methods (backtest is slow)")
    parser.add_argument("--n-test-years", type=int, default=5,
                        help="Length of the test window used in the backtest")
    parser.add_argument("--output", help="Optional CSV path for the results table")
    args = parser.parse_args()

    tester = EmissionsPeakTest()
    with contextlib.redirect_stdout(io.StringIO()):
        tester.load_historical_data(
            args.data, region=args.region, variable=args.variable,
            year_range=range(args.start, args.end + 1),
        )
        results = tester.compare_methods(
            selection=args.selection, methods=args.methods,
            alpha=args.alpha, max_exclusions=args.max_exclusions,
            n_protected=args.n_protected, n_test_years=args.n_test_years,
        )

    print(f"\n{args.region} | {args.variable} | {args.start}-{args.end}\n")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(tester.summarise_method_comparison())

    if args.output:
        results.drop(columns=["noise", "bt_windows"], errors="ignore").to_csv(args.output)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
