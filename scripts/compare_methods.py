"""
Compare Methods
==================================

For a given emissions series, decompose into trend + noise using each of the
trend methods available in EmissionsPeakTest, and check whether the resulting
noise satisfies the assumptions of the peak test:

    1. No autocorrelation (Ljung-Box) - either in the residuals directly, or in
       the innovations once an AR(1) process has been removed
    2. No heteroskedasticity (ARCH LM test)
    3. Normality (Jarque-Bera)

If ARCH or normality fail, outlier years are detected with a generalized ESD
(Rosner) test and removed one at a time (most extreme first), refitting the
trend each time, up to `max_exclusions` years. Years in the last
`n_protected` historical years are never removed automatically - they are
reported as warnings for the user to decide on. If the next year to remove is
adjacent to another flagged/excluded year with a residual of the same sign, this
is treated as a structural break rather than outliers, and the method fails.

One of the methods that pass is then selected, by one of:
    - "priority" (default): the first passing method in METHOD_PRIORITY
    - "backtest": the best rolling-origin calibration backtest of the method's
      noise model (see `backtest_method`) - slower, ~10 s per passing method
    - "manual": a named method, which must pass the tests

Usage:
    python scripts/compare_methods.py --data gcb_hist_co2.csv --region WLD \
        --variable "Emissions|CO2|Fossil" --start 1970 --end 2024 \
        [--selection priority|backtest|manual] [--method hp]

Or from Python:
    tester = EmissionsPeakTest()
    tester.load_historical_data(...)
    results = compare_methods(tester)

"""

# =================================
# Module Imports
# =================================

import argparse
import contextlib
import io
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import acf

from peak_tester import AR1_PVALUE_THRESHOLD, EmissionsPeakTest

warnings.filterwarnings("ignore")

METHODS = [
    "linear",
    "linear_w_autocorrelation",
    "broken_trend",
    "hp",
    "hamilton",
    "spline",
    "lowess",
]

# Order of preference used by selection="priority": the first passing method in this
# list is selected
# Ratinale: methods with explicit structure come first, then the flexible smoothers 
# A more in-depth analysis via backtest showed they tend to absorb noise), 
# with Hamilton last given its open issues.
METHOD_PRIORITY = [
    "broken_trend",
    "linear_w_autocorrelation",
    "linear",
    "hp",
    "spline",
    "lowess",
    "hamilton",
]

SELECTION_OPTIONS = ("priority", "backtest", "manual")

# =================================
# A) Gap-aware statistical tests
# =================================
# Excluding a year leaves a gap in the series. These helpers work on a series
# reindexed to a continuous range of years (NaN in the gaps), so that lags are
# never computed across a removed year.


def _to_full_index(series: pd.Series) -> pd.Series:
    """Reindex a year-indexed series to a continuous range of years."""
    series = series.sort_index()
    return series.reindex(range(int(series.index.min()), int(series.index.max()) + 1))


def _n_lags(series: pd.Series) -> int:
    """Number of lags for Ljung-Box / ARCH: min(10, n/5)."""
    return max(1, min(10, series.notna().sum() // 5))


def ljung_box(series: pd.Series, lags: Optional[int] = None) -> Dict:
    """
    Ljung-Box test for autocorrelation, ignoring pairs that span a gap.

    Returns p-values at lag 1 and at `lags`. The series fails if either is
    significant (lag 1 catches simple AR(1) structure that a joint test over
    many lags can dilute).
    """
    full = _to_full_index(series)
    lags = lags or _n_lags(full)
    n = full.notna().sum()
    rho = acf(full.values, nlags=lags, missing="conservative", fft=False)[1:]
    q = n * (n + 2) * np.cumsum(rho**2 / (n - np.arange(1, lags + 1)))
    pvalues = stats.chi2.sf(q, np.arange(1, lags + 1))
    return {"p_lag1": pvalues[0], "p_lagL": pvalues[-1], "lags": lags}


def arch_test(series: pd.Series, lags: Optional[int] = None) -> float:
    """
    Engle's ARCH LM test: regress squared residuals on their own lags.
    Rows where any lag falls in a gap are dropped. Returns the p-value.
    """
    full = _to_full_index(series)
    lags = lags or _n_lags(full)
    sq = (full - full.mean()) ** 2
    df = pd.concat([sq.shift(i).rename(i) for i in range(lags + 1)], axis=1).dropna()
    y = df[0].values
    X = np.column_stack([np.ones(len(df)), df.drop(columns=0).values])
    model = OLS(y, X).fit()
    lm = len(df) * model.rsquared
    return stats.chi2.sf(lm, lags)


def fit_ar1(residuals: pd.Series) -> Dict:
    """
    Fit r_t = c + phi * r_{t-1} + e_t, using only consecutive-year pairs.
    Returns phi and the innovations e_t (indexed by year t).
    """
    full = _to_full_index(residuals)
    df = pd.concat([full.rename("y"), full.shift(1).rename("x")], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    model = OLS(df["y"].values, X).fit()
    innovations = pd.Series(index=df.index, data=model.resid)
    return {"phi": model.params[1], "innovations": innovations}


def normality_test(series: pd.Series) -> Dict:
    """
    Jarque-Bera test, plus its two components so that a failure can be
    attributed to skewness or to heavy tails (excess kurtosis).
    """
    x = series.dropna().values
    n = len(x)
    skew = stats.skew(x)
    ex_kurt = stats.kurtosis(x)  # Fisher definition: normal = 0
    jb = stats.jarque_bera(x)
    return {
        "p": jb.pvalue,
        "skew": skew,
        "excess_kurtosis": ex_kurt,
        "jb_skew_component": n / 6 * skew**2,
        "jb_kurtosis_component": n / 24 * ex_kurt**2,
    }


def gesd(series: pd.Series, max_k: int = 3, alpha: float = 0.05) -> List[int]:
    """
    Generalized ESD test (Rosner, 1983) for up to `max_k` outliers.

    Iteratively removes the most extreme point (by ordinary z-score on the
    remaining data) and compares it with a critical value lambda_i. The number
    of outliers is the largest i for which R_i > lambda_i.

    Returns flagged index labels, most extreme first.
    """
    x = series.dropna().copy()
    n = len(x)
    candidates, n_outliers = [], 0
    for i in range(1, min(max_k, n - 2) + 1):
        deviations = (x - x.mean()).abs()
        idx = deviations.idxmax()
        R = deviations[idx] / x.std(ddof=1)
        p = 1 - alpha / (2 * (n - i + 1))
        t = stats.t.ppf(p, n - i - 1)
        lam = (n - i) * t / np.sqrt((n - i - 1 + t**2) * (n - i + 1))
        candidates.append(idx)
        if R > lam:
            n_outliers = i
        x = x.drop(idx)
    return [int(c) for c in candidates[:n_outliers]]


# =================================
# B) Evaluate a single method
# =================================


def _diagnose(
    tester: EmissionsPeakTest,
    method: str,
    excluded: List[int],
    alpha: float,
    method_kwargs: Dict,
) -> Dict:
    """Fit the trend (excluding `excluded` years) and run all diagnostics."""
    with contextlib.redirect_stdout(io.StringIO()):
        _, trend, _ = tester._calculate_residuals(
            method, ignore_years=excluded or None, **method_kwargs
        )
    # Recompute residuals as data - trend so the sign convention is the same for all methods
    emissions = tester.historical_data["emissions"]
    residuals = (emissions.loc[trend.index] - trend).dropna()

    # Use AR(1) on the same lag-1 rule as EmissionsPeakTest. Whether AR(1) was
    # sufficient is then checked by the multi-lag Ljung-Box test on the noise
    lb_resid = ljung_box(residuals)
    ar1_used = lb_resid["p_lag1"] < AR1_PVALUE_THRESHOLD

    if ar1_used:
        ar = fit_ar1(residuals)
        phi, noise = ar["phi"], ar["innovations"]
        lb_noise = ljung_box(noise)
    else:
        phi, noise, lb_noise = None, residuals, lb_resid

    return {
        "residuals": residuals,
        "noise": noise,  # the series the ARCH and normality tests apply to
        "ar1_used": ar1_used,
        "phi": phi,
        "lb_resid_p": min(lb_resid["p_lag1"], lb_resid["p_lagL"]),
        "lb_noise_p": min(lb_noise["p_lag1"], lb_noise["p_lagL"]),
        "arch_p": arch_test(noise),
        "normality": normality_test(noise),
    }


def _adjacent_years(year: int, noise_years, excluded: List[int]) -> List[int]:
    """
    The nearest years before and after `year`, among years that are either in the
    noise series or already excluded. Years dropped only because they lag across
    an excluded year (e.g. 1993 after excluding 1992 in a lag-based method) are
    skipped over, so 1992 and 1994 still count as adjacent.
    """
    available = sorted(set(int(y) for y in noise_years) | set(excluded))
    before = [y for y in available if y < year]
    after = [y for y in available if y > year]
    return ([before[-1]] if before else []) + ([after[0]] if after else [])


def _status(d: Dict, alpha: float) -> str:
    """Classify a diagnostics dict into a pass / failure-reason label."""
    if d["lb_noise_p"] < alpha or (d["ar1_used"] and abs(d["phi"]) >= 1):
        return "fail_autocorrelation"
    if d["arch_p"] < alpha:
        return "fail_arch"
    norm = d["normality"]
    if norm["p"] < alpha:
        heavy_tails = (
            norm["excess_kurtosis"] > 0
            and norm["jb_kurtosis_component"] >= norm["jb_skew_component"]
        )
        return "fail_normality_heavy_tails" if heavy_tails else "fail_normality_skewed"
    return "pass"


def evaluate_method(
    tester: EmissionsPeakTest,
    method: str,
    alpha: float = 0.05,
    max_exclusions: int = 3,
    n_protected: int = 3,
    method_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Run the diagnostics for one method, removing GESD-flagged outlier years
    (up to `max_exclusions`) if ARCH or normality fail.

    Outlier removal is not attempted for autocorrelation failures, since
    remaining autocorrelation points to a mis-specified trend, not outliers.
    """
    method_kwargs = method_kwargs or {}
    hist_years = tester.historical_data.index
    protected = set(hist_years[-n_protected:]) if n_protected > 0 else set()

    excluded: List[int] = []
    excluded_signs: Dict[int, float] = {}  # sign of each excluded year's noise when it was flagged
    flagged_protected: List[int] = []
    break_years: List[int] = []

    d = _diagnose(tester, method, excluded, alpha, method_kwargs)
    initial = d
    status = _status(d, alpha)

    while status in ("fail_arch", "fail_normality_heavy_tails", "fail_normality_skewed"):
        budget = max_exclusions - len(excluded)
        if budget <= 0:
            break
        flagged = gesd(d["noise"], max_k=budget, alpha=alpha)
        flagged_protected += [y for y in flagged if y in protected and y not in flagged_protected]
        eligible = [y for y in flagged if y not in protected]
        if not eligible:
            break
        candidate = int(eligible[0])

        # Check whether the candidate is next to another flagged/excluded year with the
        # same sign. That points to a sustained shift the trend can't follow (a structural
        # break), not a one-off outlier, so years shouldn't be removed. A one-off shock
        # followed by a rebound (e.g. 2020/2021) has opposite signs and is not caught here.
        signs = {**excluded_signs, **{y: np.sign(d["noise"][y]) for y in flagged}}
        neighbours = _adjacent_years(candidate, d["noise"].index, excluded)
        same_sign = [y for y in neighbours if signs.get(y) == signs[candidate]]
        if same_sign:
            status = "fail_structural_break"
            break_years = sorted({candidate, *same_sign})
            break

        excluded.append(candidate)
        excluded_signs[candidate] = signs[candidate]
        d = _diagnose(tester, method, sorted(excluded), alpha, method_kwargs)
        status = _status(d, alpha)

    if status == "pass" and excluded:
        status = "pass_with_exclusions"

    messages = []
    if break_years:
        messages.append(
            f"possible structural break around {break_years[0]}-{break_years[-1]} "
            "(consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints"
        )
    if flagged_protected:
        messages.append(
            f"{sorted(flagged_protected)} flagged as outlier(s) but within the last "
            f"{n_protected} historical years - not excluded, review manually"
        )
    warning = "; ".join(messages) or None

    return {
        "method": method,
        "status": status,
        "passes": status.startswith("pass"),
        # Years removed; if the method still fails these are the years that were tried
        "years_excluded": sorted(excluded),
        "flagged_protected": sorted(flagged_protected),
        "break_years": break_years,
        "n": len(d["noise"]),
        "ar1_used": d["ar1_used"],
        "phi": d["phi"],
        "lb_p": d["lb_noise_p"],
        "arch_p": d["arch_p"],
        "jb_p": d["normality"]["p"],
        "skew": d["normality"]["skew"],
        "excess_kurtosis": d["normality"]["excess_kurtosis"],
        "initial_status": _status(initial, alpha),
        "initial_lb_p": initial["lb_noise_p"],
        "initial_arch_p": initial["arch_p"],
        "initial_jb_p": initial["normality"]["p"],
        "warning": warning,
        "noise": d["noise"],
    }


# =================================
# C) Calibration backtest
# =================================
# The peak test only takes the noise (sigma, phi) from the decomposition method - the null
# trajectory itself is the recent historical trend. So methods are ranked on how well their
# noise describes what actually happened next: at each origin year T, characterise the noise
# on data up to T, run the test's own null simulation ("recent_trend") for the window
# T+1..T+N, and score the realised slope against the simulated slope distribution.


def _detrended_sd(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Residual standard deviation around a straight line fitted to each row of `values`
    (shape (n_samples, n_years) or (n_years,)), with n - 2 degrees of freedom.
    """
    X = np.column_stack([np.ones(len(years)), years - years.mean()])
    hat = X @ np.linalg.pinv(X)
    resid = values - values @ hat.T
    return np.sqrt((resid**2).sum(axis=-1) / (len(years) - 2))


def _simulate_noise(tester: EmissionsPeakTest, n_years: int, n_samples: int) -> np.ndarray:
    """Draw noise paths from the tester's generator, choosing AR(1)/white noise as the test does."""
    use_ar = tester.autocorr_params["has_autocorr"] and tester.autocorr_params["is_stationary"]
    if use_ar:
        return np.array([tester.noise_generator(n_years, initial_value=None) for _ in range(n_samples)])
    return np.array([tester.noise_generator(n_years) for _ in range(n_samples)])


def _recent_trend(emissions: pd.Series, n_years: int = 5) -> float:
    """Slope of a linear fit to the last `n_years` of emissions (as in load_historical_data)."""
    recent = emissions.tail(n_years)
    return np.polyfit(recent.index.values, recent.values, 1)[0]


def backtest_method(
    tester: EmissionsPeakTest,
    method: str,
    excluded: List[int],
    n_test_years: int = 5,
    min_train_years: int = 20,
    n_bootstrap: int = 1000,
    method_kwargs: Optional[Dict] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Rolling-origin calibration backtest of one method's noise model. Returns the
    scores for each origin; aggregate them with `summarise_backtest`.

    For each origin T (with at least `min_train_years` of data up to T, and
    `n_test_years` of data after it), the method is fitted on years <= T (ignoring
    any of `excluded` that fall in that range), and the bootstrap null distribution
    of slopes for T+1..T+N is generated exactly as in the peak test, under the
    "recent_trend" null. Two things are scored:

    1. Scatter (used for ranking): the standard deviation of actual emissions around
       a straight line fitted to T+1..T+N, against the distribution of the same
       statistic for simulated noise. This is set by the same sigma/phi that set the
       test's slope uncertainty, but does not depend on the trend being right.
        - scatter_log_score: log density of the realised scatter (KDE on log scale)
        - scatter_ratio: realised / median simulated scatter (>1: noise too narrow)

    2. Slope (diagnostic): the realised slope against the bootstrap null slopes.
        - pit: share of simulated slopes <= realised slope, i.e. the peak test's
          one-tailed p-value at that origin
        - slope_log_score: log density of the realised slope (normal approximation)
       Realised slopes include real changes in trend (e.g. the 2000s acceleration),
       so this mixes noise calibration with how well the recent-trend null holds.
    """
    method_kwargs = method_kwargs or {}
    emissions = tester.historical_data["emissions"]
    years = emissions.index
    origins = [T for T in years
               if (T - years[0] + 1) >= min_train_years and T + n_test_years <= years[-1]]

    rows = []
    for T in origins:
        bt = EmissionsPeakTest()
        bt.region, bt.variable, bt.unit = tester.region, tester.variable, tester.unit
        bt.historical_data = tester.historical_data.loc[:T].copy()
        bt.recent_historical_trend = _recent_trend(bt.historical_data["emissions"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bt.characterize_noise(
                    method=method, noise_type="normal",
                    ignore_years=[y for y in excluded if y <= T] or None, **method_kwargs,
                )
                bt.create_noise_generator()
                bt.set_test_data(tester.historical_data.loc[T + 1 : T + n_test_years].copy())
                # Same random numbers for every method at a given origin, so differences
                # between methods are not down to simulation noise
                np.random.seed(seed + int(T))
                slopes = bt._generate_bootstrap_slopes(n_bootstrap, "recent_trend", "ar_bootstrap")
                sim_sd = _detrended_sd(_simulate_noise(bt, n_test_years, n_bootstrap),
                                       bt.test_data.index.values.astype(float))
        except Exception:
            # e.g. a method that cannot be fitted on a short training sample - skip this origin
            continue

        realised = bt.test_slope
        realised_sd = _detrended_sd(bt.test_data["emissions"].values,
                                    bt.test_data.index.values.astype(float))
        rows.append({
            "origin": T,
            "scatter_log_score": stats.gaussian_kde(np.log(sim_sd))(np.log(realised_sd))[0],
            "scatter_ratio": realised_sd / np.median(sim_sd),
            "pit": np.mean(slopes <= realised),
            "slope_log_score": stats.norm.logpdf(realised, slopes.mean(), slopes.std()),
        })

    windows = pd.DataFrame(rows, columns=["origin", "scatter_log_score", "scatter_ratio",
                                          "pit", "slope_log_score"]).set_index("origin")
    windows["scatter_log_score"] = np.log(windows["scatter_log_score"])
    return windows


def summarise_backtest(windows: pd.DataFrame) -> Dict:
    """Aggregate per-origin backtest scores (from backtest_method) into summary metrics."""
    pit = windows["pit"]
    return {
        "bt_scatter_log_score": windows["scatter_log_score"].mean(),
        # Typical realised / predicted scatter: ~1 if sigma and phi are right
        "bt_scatter_ratio": np.exp(np.log(windows["scatter_ratio"]).mean()),
        "bt_slope_log_score": windows["slope_log_score"].mean(),
        # Share of realised slopes inside the central 90% of the null distribution
        "bt_slope_coverage90": ((pit >= 0.05) & (pit <= 0.95)).mean(),
        # Share of windows where the test would have declared a significant decline
        # at p < 0.1 under the recent-trend null
        "bt_rate_p10": (pit < 0.1).mean(),
        "bt_n_windows": len(windows),
    }


def select_method(
    results: pd.DataFrame, selection: str = "priority", manual_method: Optional[str] = None
) -> Optional[str]:
    """
    Choose a method from those that pass the diagnostic tests. Returns None if none pass.

        - "priority": the first passing method in METHOD_PRIORITY
        - "backtest": the passing method with the best backtest scatter log score
          (requires the bt_* columns from compare_methods(selection="backtest"))
        - "manual": `manual_method`, which must be one of the passing methods
    """
    passing = results.index[results["passes"]].tolist()
    if not passing:
        return None

    if selection == "priority":
        ranked = [m for m in METHOD_PRIORITY if m in passing]
        # Methods missing from METHOD_PRIORITY come last, in the order they were run
        return (ranked + [m for m in passing if m not in ranked])[0]

    if selection == "backtest":
        return results.loc[passing, "bt_scatter_log_score"].idxmax()

    if selection == "manual":
        if manual_method not in passing:
            raise ValueError(
                f"Manually selected method '{manual_method}' does not pass the diagnostic "
                f"tests. Passing methods: {passing}"
            )
        return manual_method

    raise ValueError(f"selection must be one of {SELECTION_OPTIONS}")


# =================================
# D) Compare all methods
# =================================


def compare_methods(
    tester: EmissionsPeakTest,
    methods: List[str] = METHODS,
    alpha: float = 0.05,
    max_exclusions: int = 3,
    n_protected: int = 3,
    method_kwargs: Optional[Dict[str, Dict]] = None,
    selection: str = "priority",
    manual_method: Optional[str] = None,
    n_test_years: int = 5,
) -> pd.DataFrame:
    """
    Evaluate each method on the historical data loaded in `tester`, and select one of
    the methods that pass (see `select_method` for the `selection` options). The
    calibration backtest is only run for selection="backtest", as it is slow.

    Returns a DataFrame indexed by method. The `noise` column holds the final
    residual/innovation series tested for each method, and (for backtest) `bt_windows`
    the per-origin backtest scores. The selected method is in `results.attrs["selected"]`.
    """
    if tester.historical_data is None:
        raise ValueError("Must load historical data first")
    if selection not in SELECTION_OPTIONS:
        raise ValueError(f"selection must be one of {SELECTION_OPTIONS}")
    method_kwargs = method_kwargs or {}
    backtest = selection == "backtest"

    rows = []
    for method in methods:
        try:
            rows.append(
                evaluate_method(
                    tester, method, alpha, max_exclusions, n_protected,
                    method_kwargs.get(method, {}),
                )
            )
        except Exception as e:
            rows.append({"method": method, "status": "error", "passes": False, "warning": str(e)})

    if backtest:
        passing_rows = [row for row in rows if row["passes"]]
        for row in passing_rows:
            row["bt_windows"] = backtest_method(
                tester, row["method"], row["years_excluded"], n_test_years=n_test_years,
                method_kwargs=method_kwargs.get(row["method"], {}),
            )
        # Score every method on the same set of windows, so the scores are comparable
        if passing_rows:
            common = passing_rows[0]["bt_windows"].index
            for row in passing_rows[1:]:
                common = common.intersection(row["bt_windows"].index)
            for row in passing_rows:
                row.update(summarise_backtest(row["bt_windows"].loc[common]))

    results = pd.DataFrame(rows).set_index("method")
    results.attrs["selection"] = selection
    results.attrs["selected"] = select_method(results, selection, manual_method)

    if not results["passes"].any():
        warnings.warn(
            f"No method passes all diagnostic tests for {tester.region} "
            f"({tester.variable}). Peak test results should be treated with caution.",
            stacklevel=2,
        )
    return results


def summarise(results: pd.DataFrame) -> str:
    """Human-readable summary table of compare_methods() output."""
    cols = ["status", "years_excluded", "ar1_used", "phi", "lb_p", "arch_p", "jb_p",
            "skew", "excess_kurtosis", "initial_status", "warning"]
    table = results.reindex(columns=cols).copy()
    for c in ["phi", "lb_p", "arch_p", "jb_p", "skew", "excess_kurtosis"]:
        table[c] = table[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    table["warning"] = table["warning"].fillna("")

    passing = results.index[results["passes"]].tolist()
    lines = [table.to_string(), ""]
    if not passing:
        lines.append("WARNING: no method passes all tests.")
        return "\n".join(lines)

    lines.append("Methods passing all tests:")
    if "bt_scatter_log_score" in results:
        bt_cols = {
            "bt_scatter_log_score": "scatter_score",
            "bt_scatter_ratio": "scatter_ratio",
            "bt_slope_coverage90": "slope_cov90",
            "bt_rate_p10": "rate_p<0.1",
        }
        bt = results.loc[passing, ["years_excluded", *bt_cols]].rename(columns=bt_cols)
        bt = bt.sort_values("scatter_score", ascending=False)
        lines.append(bt.to_string(float_format=lambda v: f"{v:.3f}"))
        lines.append(
            "  scatter_score: backtest log score, higher is better (used for selection)\n"
            "  scatter_ratio: realised / predicted scatter in 5-yr windows (~1 ideal, >1 noise too narrow)\n"
            "  slope_cov90 / rate_p<0.1: slope calibration under the recent-trend null (0.90 / 0.10 ideal;\n"
            "  affected by real trend changes, not just noise)"
        )
    else:
        for m in passing:
            excl = results.loc[m, "years_excluded"]
            lines.append(f"  - {m}" + (f" (excluding {excl})" if excl else ""))

    selected = results.attrs.get("selected")
    if selected:
        excl = results.loc[selected, "years_excluded"]
        lines.append(
            f"\nSuggested method ({results.attrs.get('selection')}): {selected}"
            + (f" (excluding {excl})" if excl else "")
        )
    return "\n".join(lines)


# =================================
# E) Command line
# =================================


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
    parser.add_argument("--method", help="Method to use with --selection manual")
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

    results = compare_methods(
        tester, methods=args.methods, alpha=args.alpha,
        max_exclusions=args.max_exclusions, n_protected=args.n_protected,
        selection=args.selection, manual_method=args.method, n_test_years=args.n_test_years,
    )

    print(f"\n{args.region} | {args.variable} | {args.start}-{args.end}\n")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(summarise(results))

    if args.output:
        results.drop(columns=["noise", "bt_windows"], errors="ignore").to_csv(args.output)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
