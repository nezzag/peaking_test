"""
Peak Tester
==================================

A statistical test to determine if an emissions series has peaked
based on N years of declining emissions data, and accounting for the
noise in historical data (which is assumed to continue)

Usage:
    # Initialize the test
    peak_test = EmissionsPeakTest()
    
    # Load your data
    peak_test.load_historical_data('path/to/your/data.csv')

    # Characterize noise. By default (method="priority") this tests the trend
    # decomposition methods and uses the first (by priority) whose noise passes the
    # diagnostic tests, excluding any outlier years identified (or warns if none passes)
    peak_test.characterize_noise()  # or method="backtest"
    print(peak_test.summarise_method_comparison())
    # ...or use a specific method; its test results are reported, pass or fail
    peak_test.characterize_noise(method='hp')
    
    # Set test data
    test_data = [(2021, 36500), (2022, 35800), (2023, 35100)]
    peak_test.set_test_data(test_data)
    
    # Run the test
    results = peak_test.run_bootstrap_test(n_bootstrap=1000)
    
    # Visualize results
    peak_test.plot_analysis()
    
    # Get interpretation
    interpretation = peak_test.interpret_results()

Authors: Neil Grant and Claire Fyson
"""

# =================================
# Module Imports
# =================================

import contextlib
import io
from pathlib import Path
import warnings
from typing import Tuple, Dict, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from pandas_indexing import isin
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.interpolate import make_splrep
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# An AR(1) noise process is used when a lag-1 Ljung-Box test on the residuals has a
# p-value below this threshold. Used both when characterising the noise and when
# selecting a method, so both make the same choice
AR1_PVALUE_THRESHOLD = 0.1

# Trend decomposition methods compared when selecting a method
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
# Rationale: methods with explicit structure come first, then the flexible smoothers
# (a more in-depth analysis via backtest showed they tend to absorb noise),
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

# Ways of selecting a method from those passing the diagnostic tests
SELECTION_OPTIONS = ("priority", "backtest")


class MethodSelectionWarning(UserWarning):
    """No decomposition method passes the diagnostic tests, or a failing method is used."""


# Always shown, even though other warnings are silenced above
warnings.filterwarnings("always", category=MethodSelectionWarning)


# =================================
# Statistical tests used to select a method
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


def _diagnostic_status(d: Dict, alpha: float) -> str:
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


def _detrended_sd(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Residual standard deviation around a straight line fitted to each row of `values`
    (shape (n_samples, n_years) or (n_years,)), with n - 2 degrees of freedom.
    """
    X = np.column_stack([np.ones(len(years)), years - years.mean()])
    hat = X @ np.linalg.pinv(X)
    resid = values - values @ hat.T
    return np.sqrt((resid**2).sum(axis=-1) / (len(years) - 2))


def _recent_trend(emissions: pd.Series, n_years: int = 5) -> float:
    """Slope of a linear fit to the last `n_years` of emissions (as in load_historical_data)."""
    recent = emissions.tail(n_years)
    return np.polyfit(recent.index.values, recent.values, 1)[0]


def _summarise_backtest(windows: pd.DataFrame) -> Dict:
    """Aggregate per-origin backtest scores (from _backtest_method) into summary metrics."""
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


def select_method(results: pd.DataFrame, selection: str = "priority") -> Optional[str]:
    """
    Choose a method from those that pass the diagnostic tests. Returns None if none pass.

        - "priority": the first passing method in METHOD_PRIORITY
        - "backtest": the passing method with the best backtest scatter log score
          (requires the bt_* columns from compare_methods(selection="backtest"))
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

    raise ValueError(f"selection must be one of {SELECTION_OPTIONS}")


def summarise_method_comparison(results: pd.DataFrame) -> str:
    """Human-readable summary table of EmissionsPeakTest.compare_methods() output."""
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
# Define Key Class
# =================================


class EmissionsPeakTest:
    """
    A comprehensive statistical test for detecting emissions peaks.

    This class implements a bootstrap hypothesis test to determine if observed
    declining emissions represent a statistically significant trend beyond
    random year-to-year fluctuations.

    Attributes:
        TODO: Update this list with the latest set of attributes
        region (str): ISO code for the region you want to analyse
        historical_data (pd.DataFrame): Historical emissions data
        test_data (pd.DataFrame): Recent emissions data showing potential decline
        emissions_data (pd.DataFrame): A composite of historical emissions data and potential test data
        that we are analysing for decline. In the case that emissions have declined in the real world,
        then emissions_data and historical_data would be equivalent
        noise_params (Dict): Parameters of the fitted noise distribution
        noise_generator (Callable): Function to generate noise samples
        bootstrap_results (Dict): Results from the bootstrap hypothesis test
        residuals (np.ndarray): Residuals from trend fitting
        method_comparison (pd.DataFrame): Diagnostic test results for each trend method
            (from compare_methods(), run by characterize_noise() when selecting a method)
        selected_method (str): Method selected from those passing the tests. None if no
            method passes
        selected_ignore_years (list): Outlier years excluded for the selected method
    """

    def __init__(self):
        """
        Initialize the emissions peak test.
        """
        self.region: Optional[str] = None
        self.variable: Optional[str] = None
        self.historical_data: Optional[pd.DataFrame] = None
        self.emissions_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.recent_historical_trend: Optional[float] = None
        self.noise_generator: Optional[Callable] = None
        self.bootstrap_results: Optional[Dict] = None
        self.residuals: Optional[pd.Series] = None
        self.trend: Optional[pd.Series] = None
        self.trend_info: Optional[Dict] = None
        self.method_comparison: Optional[pd.DataFrame] = None
        self.selected_method: Optional[str] = None
        self.selected_ignore_years: List[int] = []

    def __getstate__(self) -> Dict:
        """Drop the unpicklable closure before serialisation."""
        state = self.__dict__.copy()
        state["noise_generator"] = None
        return state

    def __setstate__(self, state: Dict) -> None:
        """Restore state and rebuild the noise generator from saved params."""
        self.__dict__.update(state)
        if self.autocorr_params is not None:
            self.create_noise_generator()

    # =================================
    # A) Loading data
    # =================================

    def load_historical_data(
        self,
        data_source: Union[str, pd.DataFrame],
        region: str = "WLD",
        variable: str = "Emissions|CO2|Total",
        year_range: range = range(1970, 2025),
    ) -> "EmissionsPeakTest":
        """
        Load historical emissions data.

        Args:
            data_source: Path to CSV file or pandas DataFrame
            region: ISO3 code of the region
            variable: Variable to load
            year_range: Years of historical data to keep

        Returns:
            Self for method chaining
        """
        if self.region is None:
            try:
                self.region = region
            except Exception as e:
                raise ValueError(
                    "No region selected - provide ISO3 code for region of interest"
                )
            
        if self.variable is None:
            try:
                self.variable = variable
            except Exception as e:
                raise ValueError(
                    "No variable selected - provide variable of interest"
                )

        if isinstance(data_source, str):
            # Load from file, which should be stored in the data folder
            data_path = (
                Path(__file__).resolve().parent / f"../data/processed/{data_source}"
            )

            try:
                data = pd.read_csv(data_path, index_col=[0, 1, 2, 3, 4])
            except Exception as e:
                raise ValueError(f"Could not load data from {data_path}: {e}")

        elif isinstance(data_source, pd.DataFrame):
            assert data_source.index.names == [
                "region",
                "region_name",
                "variable",
                "unit",
            ]
            data = data_source.copy()

        else:
            raise ValueError("data_source must be a file path or DataFrame")

        data = data.loc[isin(region=self.region)]
        data = data.loc[isin(variable=self.variable)]
        # self.variable = data.pix.unique("variable")[0]
        self.unit = data.pix.unique("unit")[0]
        # Cast into long-form and rename
        data.columns = data.columns.astype(int)
        data = pd.Series(index=data.columns, data=data.values.squeeze()).reset_index()
        data.columns = ["year", "emissions"] # TODO: Make this more flexible to allow for different column names
        data = data.set_index("year")

        # Filter to the selected years
        self.historical_data = data.loc[data.index.isin(year_range)]

        # Validate data
        self._validate_historical_data()

        # calculate recent trend (excluding most recent year)
        recent_data = self.historical_data.tail(5)
        X_recent = recent_data.index.values.reshape(-1, 1) # extract years for regression
        y_recent = recent_data["emissions"].values
        model_recent = LinearRegression()
        model_recent.fit(X_recent, y_recent)
        self.recent_historical_trend = model_recent.coef_[0]

        print(
            f"Loaded historical data: {self.historical_data.index.min()}-{self.historical_data.index.max()}"
        )
        print(f"Data points: {len(self.historical_data)}")

        # Any previous method comparison was for other data
        self.method_comparison = None
        self.selected_method = None
        self.selected_ignore_years = []

        return self

    def _validate_historical_data(self) -> None:
        """Validate the loaded historical data."""
        if self.historical_data is None:
            raise ValueError("No historical data loaded")

        if len(self.historical_data) < 10:
            raise ValueError("Need at least 10 years of historical data")

        if self.historical_data["emissions"].isna().any():
            raise ValueError("Historical data contains missing values")

        if (self.historical_data["emissions"] < 0).any():
            raise ValueError("Historical emissions cannot be negative")

    # =================================
    # A2) Selecting a trend decomposition method
    # =================================

    def compare_methods(
        self,
        selection: str = "priority",
        methods: List[str] = METHODS,
        alpha: float = 0.05,
        max_exclusions: int = 3,
        n_protected: int = 3,
        method_kwargs: Optional[Dict[str, Dict]] = None,
        n_test_years: int = 5,
    ) -> pd.DataFrame:
        """
        Decompose the historical data with each trend method, test whether the noise
        satisfies the assumptions of the peak test, and select a method that does.

        For each method the noise is tested for:
            1. Autocorrelation (Ljung-Box) - either in the residuals directly, or in the
               innovations once an AR(1) process has been removed
            2. Heteroskedasticity (ARCH LM test)
            3. Normality (Jarque-Bera)

        If ARCH or normality fail, outlier years are detected with a generalized ESD
        (Rosner) test and removed one at a time (most extreme first), refitting the trend
        each time, up to `max_exclusions` years. Years in the last `n_protected`
        historical years are never removed automatically - they are reported as
        warnings. If the next year to remove is adjacent to another flagged/excluded year
        with a residual of the same sign, this is treated as a structural break rather
        than outliers, and the method fails.

        One of the passing methods is then selected:
            - "priority": the first passing method in METHOD_PRIORITY
            - "backtest": the best rolling-origin calibration backtest of the method's
              noise model (see _backtest_method) - slow, ~10 s per passing method

        Called by characterize_noise(method="priority"/"backtest"); call it directly to
        inspect the results or change the settings.

        Args:
            selection: "priority" or "backtest"
            methods: Methods to compare
            alpha: Significance level for the diagnostic and outlier tests
            max_exclusions: Maximum number of outlier years removed per method
            n_protected: Number of final historical years that are never removed
            method_kwargs: Optional {method: kwargs} passed to the decomposition
            n_test_years: Length of the test window used in the backtest

        Returns:
            DataFrame indexed by method (also stored as self.method_comparison). The
            `noise` column holds the final residual/innovation series tested for each
            method, and (for backtest) `bt_windows` the per-origin backtest scores. The
            selected method and its excluded years are stored as self.selected_method
            and self.selected_ignore_years.
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first")
        if selection not in SELECTION_OPTIONS:
            raise ValueError(f"selection must be one of {SELECTION_OPTIONS}")
        method_kwargs = method_kwargs or {}

        rows = []
        for method in methods:
            try:
                rows.append(
                    self._evaluate_method(
                        method, alpha, max_exclusions, n_protected, method_kwargs.get(method, {})
                    )
                )
            except Exception as e:
                rows.append({"method": method, "status": "error", "passes": False, "warning": str(e)})

        if selection == "backtest":
            passing_rows = [row for row in rows if row["passes"]]
            for row in passing_rows:
                row["bt_windows"] = self._backtest_method(
                    row["method"], row["years_excluded"], n_test_years=n_test_years,
                    method_kwargs=method_kwargs.get(row["method"], {}),
                )
            # Score every method on the same set of windows, so the scores are comparable
            if passing_rows:
                common = passing_rows[0]["bt_windows"].index
                for row in passing_rows[1:]:
                    common = common.intersection(row["bt_windows"].index)
                for row in passing_rows:
                    row.update(_summarise_backtest(row["bt_windows"].loc[common]))

        results = pd.DataFrame(rows).set_index("method")
        results.attrs["selection"] = selection
        results.attrs["selected"] = select_method(results, selection)

        self.method_comparison = results
        self.selected_method = results.attrs["selected"]
        self.selected_ignore_years = (
            list(results.loc[self.selected_method, "years_excluded"]) if self.selected_method else []
        )

        if self.selected_method is None:
            warnings.warn(
                f"No trend decomposition method passes the diagnostic tests (autocorrelation, "
                f"ARCH, normality) for {self.region} ({self.variable}). Peak test results "
                "should be treated with caution; pass a method name to characterize_noise() to "
                "use one anyway. See summarise_method_comparison() for details.",
                MethodSelectionWarning,
                stacklevel=2,
            )
        else:
            passing = results.index[results["passes"]].tolist()
            print(
                f"Method selection ({selection}): {self.selected_method}"
                + (f", excluding {self.selected_ignore_years}" if self.selected_ignore_years else "")
            )
            print(f"  Methods passing all tests: {', '.join(passing)}")
            warning = results.loc[self.selected_method, "warning"]
            if isinstance(warning, str):
                print(f"  Note: {warning}")

        return results

    def summarise_method_comparison(self) -> str:
        """Human-readable summary of the last compare_methods() results."""
        if self.method_comparison is None:
            raise ValueError("Run compare_methods() first")
        return summarise_method_comparison(self.method_comparison)

    def _diagnose_method(self, method: str, excluded: List[int], method_kwargs: Dict) -> Dict:
        """Fit the trend (excluding `excluded` years) and run all diagnostics."""
        with contextlib.redirect_stdout(io.StringIO()):
            _, trend, _ = self._calculate_residuals(
                method, ignore_years=excluded or None, **method_kwargs
            )
        # Recompute residuals as data - trend so the sign convention is the same for all methods
        emissions = self.historical_data["emissions"]
        residuals = (emissions.loc[trend.index] - trend).dropna()

        # Use AR(1) on the same lag-1 rule as _analyze_autocorrelation. Whether AR(1) was
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

    def _evaluate_method(
        self,
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
        hist_years = self.historical_data.index
        protected = set(hist_years[-n_protected:]) if n_protected > 0 else set()

        excluded: List[int] = []
        excluded_signs: Dict[int, float] = {}  # sign of each excluded year's noise when it was flagged
        flagged_protected: List[int] = []
        break_years: List[int] = []

        d = self._diagnose_method(method, excluded, method_kwargs)
        initial = d
        status = _diagnostic_status(d, alpha)

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
            d = self._diagnose_method(method, sorted(excluded), method_kwargs)
            status = _diagnostic_status(d, alpha)

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
            "initial_status": _diagnostic_status(initial, alpha),
            "initial_lb_p": initial["lb_noise_p"],
            "initial_arch_p": initial["arch_p"],
            "initial_jb_p": initial["normality"]["p"],
            "warning": warning,
            "noise": d["noise"],
        }

    def _simulate_noise(self, n_years: int, n_samples: int) -> np.ndarray:
        """Draw noise paths from the noise generator, choosing AR(1)/white noise as the test does."""
        use_ar = self.autocorr_params["has_autocorr"] and self.autocorr_params["is_stationary"]
        if use_ar:
            return np.array([self.noise_generator(n_years, initial_value=None) for _ in range(n_samples)])
        return np.array([self.noise_generator(n_years) for _ in range(n_samples)])

    def _backtest_method(
        self,
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
        scores for each origin; aggregate them with _summarise_backtest.

        The peak test only takes the noise (sigma, phi) from the decomposition method -
        the null trajectory itself is the recent historical trend. So methods are ranked
        on how well their noise describes what actually happened next.

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
        emissions = self.historical_data["emissions"]
        years = emissions.index
        origins = [T for T in years
                   if (T - years[0] + 1) >= min_train_years and T + n_test_years <= years[-1]]

        rows = []
        for T in origins:
            bt = type(self)()
            bt.region, bt.variable, bt.unit = self.region, self.variable, self.unit
            bt.historical_data = self.historical_data.loc[:T].copy()
            bt.recent_historical_trend = _recent_trend(bt.historical_data["emissions"])
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    bt._fit_noise(
                        method=method, noise_type="normal",
                        ignore_years=[y for y in excluded if y <= T] or None, **method_kwargs,
                    )
                    bt.create_noise_generator()
                    bt.set_test_data(self.historical_data.loc[T + 1 : T + n_test_years].copy())
                    # Same random numbers for every method at a given origin, so differences
                    # between methods are not down to simulation noise
                    np.random.seed(seed + int(T))
                    slopes = bt._generate_bootstrap_slopes(n_bootstrap, "recent_trend", "ar_bootstrap")
                    sim_sd = _detrended_sd(bt._simulate_noise(n_test_years, n_bootstrap),
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

    # =================================
    # B) Characterising the noise in the data
    # =================================

    def characterize_noise(
        self,
        method: str = "priority",
        noise_type: str = "t-dist",
        t_df: Optional[float] = None,
        ignore_years: Optional[list] = None,
        include_test_data: bool = False,
        **kwargs,
    ) -> "EmissionsPeakTest":
        """
        Characterize the noise distribution in historical emissions data.

        The trend decomposition method is chosen with `method`, which is one of:
            - "priority" (default): test every method's noise (see compare_methods) and
              use the first passing method in METHOD_PRIORITY
            - "backtest": test every method, and use the passing method with the best
              calibration backtest (slow, ~10 s per passing method)
            - a method name from METHODS (e.g. "hp"): use that method. Its noise is still
              tested and the results reported, with a warning if it does not pass
        Unless ignore_years is given, the outlier years identified in testing are
        excluded (for a named method, only if it passes). A "priority" or "backtest"
        comparison already run on this data is reused - run compare_methods() first to
        change its settings.

        Args:
            method: "priority", "backtest" or a method name (see above)
            noise_type: 'normal' or 't-dist' or 'empirical' for noise distribution type
            t_df: Optional fixed degrees of freedom for t-distribution
            ignore_years: Years to exclude, overriding the outlier years identified in
                testing. Pass [] to exclude no years
            include_test_data: Whether or not to include the test data in the system or not
            **kwargs: Additional arguments for specific methods. These include:
                - fraction: float = 0.3. Use in lowess method, change to reduce/increase the width of
                dataset over which lowess is calculated
                - forced_breakpoints: list[int]. use in broken_trend to force certain breakpoints

        Returns:
            Self for method chaining
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first")

        method, ignore_years = self._resolve_method(method, ignore_years, kwargs)
        return self._fit_noise(method, noise_type, t_df, ignore_years, include_test_data, **kwargs)

    def _fit_noise(
        self,
        method: str,
        noise_type: str = "t-dist",
        t_df: Optional[float] = None,
        ignore_years: Optional[list] = None,
        include_test_data: bool = False,
        **kwargs,
    ) -> "EmissionsPeakTest":
        """Fit the noise for a given method and ignored years, without any method testing."""
        if include_test_data:
            if self.emissions_data is None:
                raise ValueError("To run noise parameterisation with `include_test_data=True`" \
                "you must first provide test data")

        self.residuals, self.trend, self.trend_info = self._calculate_residuals(
            method, ignore_years, include_test_data, **kwargs
        )

        print(f"using {noise_type} distribution to calculate noise")
        self.autocorr_params = self._analyze_autocorrelation(noise_type=noise_type, t_df=t_df)

        print("Noise characterization complete:")
        print(f"  Method used: {method}")

        return self

    def _resolve_method(
        self, method: str, ignore_years: Optional[list], method_kwargs: Dict
    ) -> Tuple[str, Optional[list]]:
        """
        Work out the method and ignored years for characterize_noise(), running the
        diagnostic tests (see characterize_noise for the options).
        """
        if method in SELECTION_OPTIONS:
            # Reuse an existing comparison with the same selection (getattr, as objects
            # pickled before method selection existed lack this attribute)
            comparison = getattr(self, "method_comparison", None)
            if comparison is None or comparison.attrs.get("selection") != method:
                self.compare_methods(selection=method)

            if self.selected_method is None:
                raise ValueError(
                    "No trend decomposition method passes the diagnostic tests - pass a "
                    f"method name (one of {METHODS}) to use one anyway. See "
                    "summarise_method_comparison() for details"
                )
            if ignore_years is None and self.selected_ignore_years:
                ignore_years = list(self.selected_ignore_years)
                print(f"Ignoring outlier years identified in testing: {ignore_years}")
            return self.selected_method, ignore_years

        if method not in METHODS:
            raise ValueError(
                f"method must be one of {list(SELECTION_OPTIONS)} or a method in {METHODS}, "
                f"not '{method}'"
            )

        # A named method: test it and report the results, but use it either way
        if ignore_years is None:
            # Test as in compare_methods, including outlier exclusion
            result = self._evaluate_method(method, method_kwargs=method_kwargs)
            if result["passes"] and result["years_excluded"]:
                ignore_years = list(result["years_excluded"])
        else:
            # Test with exactly the years the user chose to ignore
            d = self._diagnose_method(method, list(ignore_years), method_kwargs)
            status = _diagnostic_status(d, 0.05)
            result = {
                "method": method, "status": status, "passes": status.startswith("pass"),
                "years_excluded": sorted(ignore_years), "ar1_used": d["ar1_used"],
                "phi": d["phi"], "lb_p": d["lb_noise_p"], "arch_p": d["arch_p"],
                "jb_p": d["normality"]["p"], "skew": d["normality"]["skew"],
                "excess_kurtosis": d["normality"]["excess_kurtosis"], "warning": None,
                "noise": d["noise"],
            }

        results = pd.DataFrame([result]).set_index("method")
        results.attrs["selection"] = method
        results.attrs["selected"] = method
        self.method_comparison = results
        self.selected_method = method
        self.selected_ignore_years = list(ignore_years or [])

        print(
            f"Diagnostic tests for {method}: {'PASS' if result['passes'] else 'FAIL'} "
            f"({result['status']})"
            + (f", excluding {ignore_years}" if ignore_years else "")
            + f" - Ljung-Box p={result['lb_p']:.3f}, ARCH p={result['arch_p']:.3f}, "
            f"Jarque-Bera p={result['jb_p']:.3f}"
        )
        if isinstance(result.get("warning"), str):
            print(f"  Note: {result['warning']}")
        if not result["passes"]:
            warnings.warn(
                f"Method '{method}' does not pass the diagnostic tests ({result['status']}) "
                f"for {self.region} ({self.variable}); it is used anyway. Use "
                'method="priority" to select a method that passes.',
                MethodSelectionWarning,
                stacklevel=3,
            )

        return method, ignore_years

    def _calculate_residuals(
        self, method: str, ignore_years: list = None, include_test_data: bool = False, **kwargs
    ) -> Tuple[pd.Series, Dict]:
        """Decomposes a timeseries into trend + residuals

        Args:
            method: str: Decide the method of TSA to use to extract results
            ignore_years: Years to exclude
            **kwargs: Additional arguments for specific methods

        Returns:
            residuals: pd.Series: Residuals for the timeseries
            trend: pd.Series: Trend of the timeseries
            trend_info: Dict: Dictionary with key data on the trend

        """
        residuals = pd.Series()
        trend = pd.Series()
        trend_info = {}

        if include_test_data:
            # Take the blended data of historical + test data
            hist_data = self.emissions_data.copy()
        else:
            # Just take the historical data alone
            hist_data = self.historical_data.copy()


        if ignore_years:
            hist_data = hist_data.loc[~hist_data.index.isin(ignore_years)]

        years = hist_data.index.values
        emissions = hist_data["emissions"].values

        if method == "lowess":
            # lowess smoothing
            frac = kwargs.get(
                "fraction", 0.3
            )  # Smoothing parameter -> default option 0.3
            smoothed = lowess(emissions, years, frac=frac, return_sorted=False)
            residuals = emissions - smoothed
            residuals = pd.Series(index=years, data=residuals)
            trend = pd.Series(index=years, data = smoothed)
            trend_info = {
                "method": "lowess",
                "parameters": {"fraction": frac},
            }

        elif method == "linear":
            X = np.column_stack([np.ones(len(years)), years])
            model = OLS(emissions, X).fit()
            trend = pd.Series(index=years, data = model.fittedvalues)
            residuals = pd.Series(index=years, data=emissions - trend.values)
            trend_info = {
                "method": method,
                "parameters": model.params,
                "standard_errors": model.bse,
            }

        elif method == "linear_w_autocorrelation":
            # Create lagged variable(s) for autoregressive component. Lag by calendar year
            # (not position) so that a year following an ignored year has no lag, rather
            # than being lagged on the year before the gap
            hist_data["emissions_lag"] = hist_data["emissions"].reindex(
                hist_data.index - 1).values

            # Drop rows with NaN values (first year, and any year following an ignored year)
            hist_data = hist_data.dropna()

            # Prepare features (X) and target (y)
            feature_cols = ["emissions_lag"]
            X = hist_data[feature_cols].assign(year=hist_data.index)
            y = hist_data["emissions"]

            # Fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Make predictions
            y_pred = model.predict(X)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)

            trend = pd.Series(index=y.index, data=y_pred)
            residuals = pd.Series(index=y.index, data=(y - trend.values).values)

            #TODO: Think about whether enforcing 0 residual for the first year is dangerous
            # trend = trend.reindex(years)
            # trend.loc[years[0]] = emissions[0]
            # residuals = residuals.reindex(years).dropna()
            # residuals.loc[years[0]] = 0

            trend_info = {
                "method": method,
                "parameters": {
                    "intercept": model.intercept_,
                    "trend": model.coef_[0],
                    "autocorrelation": model.coef_[1],
                    "r2": r2,
                    "rmse": rmse,
                },
                "model": model,
            }

        elif method == "broken_trend":

            # Get forced breakpoints if specified
            forced_bps = kwargs.get("forced_breakpoints", None)

            def segments_fit(X, Y, max_bps, forced_breakpoints=None):
                """Function that performs a broken trend analysis on a timeseries

                Args:
                    X: np.array: X-values of the time series
                    Y: np.array: Y-values of the time series
                    max_bps: int: Max number of break points in the time series
                    forced_breakpoints: list: Years where breakpoints must occur

                Returns:
                    x_predict: np.array: X-values of the break points in the time series
                    y_predict: np.array: Y-values of the break points in the time series
                """
                xmin = X.min()
                xmax = X.max()

                n = len(X)

                aic_ = float("inf")
                bic_ = float("inf")
                r_ = None

                # Store forced breakpoints for later merging
                forced_bps_to_add = None
                if forced_breakpoints is not None and len(forced_breakpoints) > 0:
                    # Convert to integers to ensure they're whole years
                    forced_bps_to_add = np.array(sorted(forced_breakpoints), dtype=int)

                for count in range(1, max_bps + 1):

                    seg = np.full(count - 1, (xmax - xmin) / count)

                    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
                    py_init = np.array(
                        [Y[np.abs(X - x) < (xmax - xmin) * 0.1].mean() for x in px_init]
                    )

                    def func(p):
                        seg = p[: count - 1]
                        py = p[count - 1 :]
                        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
                        return px, py

                    def err(p):  # This is RSS / n
                        px, py = func(p)
                        Y2 = np.interp(X, px, py)
                        return np.mean((Y - Y2) ** 2)

                    r = optimize.minimize(
                        err, x0=np.r_[seg, py_init], method="Nelder-Mead"
                    )

                    # Compute AIC/ BIC.
                    aic = n * np.log(err(r.x)) + 4 * count
                    bic = n * np.log(err(r.x)) + 2 * count * np.log(n)

                    # Use AIC / BIC to determine if we want to add another breakpoint and test again
                    if (bic < bic_) & (aic < aic_):
                        # If so, then repeat with an additional break point
                        r_ = r
                        aic_ = aic
                        bic_ = bic
                    else:  # Stop and return the last function
                        count = count - 1
                        break

                # Get the automatically found breakpoints
                px_auto, py_auto = func(r_.x)

                # If we have forced breakpoints, merge them with the automatic ones
                if forced_bps_to_add is not None:
                    # Combine all breakpoints (auto + forced)
                    all_bps = np.unique(np.concatenate([px_auto, forced_bps_to_add]))

                    # Now fit linear regressions in each segment defined by all breakpoints
                    y_values = []

                    for i in range(len(all_bps) - 1):
                        # Get data for this segment
                        if i == 0:
                            mask = (X >= all_bps[i]) & (X <= all_bps[i+1])
                        else:
                            mask = (X > all_bps[i]) & (X <= all_bps[i+1])

                        X_seg = X[mask]
                        Y_seg = Y[mask]

                        if len(X_seg) >= 2:
                            # Fit linear regression for this segment
                            model = LinearRegression()
                            model.fit(X_seg.reshape(-1, 1), Y_seg)

                            # Predict y-values at the breakpoints
                            y_start = model.predict([[all_bps[i]]])[0]
                            y_end = model.predict([[all_bps[i+1]]])[0]

                            # Add start point (unless it's not the first segment)
                            if i == 0:
                                y_values.append(y_start)
                            # Always add end point
                            y_values.append(y_end)
                        elif len(X_seg) == 1:
                            # Only one point in segment - use its value
                            if i == 0:
                                y_values.append(Y_seg[0])
                            y_values.append(Y_seg[0])
                        else:
                            # No points in segment - interpolate linearly
                            if i == 0 and len(y_values) == 0:
                                # First segment, use first data point
                                y_values.append(Y[0])
                            # Interpolate from previous point
                            if len(y_values) > 0:
                                y_values.append(y_values[-1])
                            else:
                                y_values.append(Y[0])

                    return all_bps, np.array(y_values)
                else:
                    # No forced breakpoints, return automatic result
                    return px_auto, py_auto

            max_bps = int(
                np.ceil(len(emissions) / 5)
            )  # Shortest trendlength is 5 years
            print(f"Max breakpoints: {max_bps}")
            breakpoints, y_breakpoint = segments_fit(years, emissions, max_bps, forced_breakpoints=forced_bps)

            trend = pd.Series(index=breakpoints, data=y_breakpoint)
            trend = (
                trend.reindex(set(years) | set(trend.index)).sort_index().interpolate(method='index')
            )
            trend = trend.reindex(years)

            residuals = pd.Series(index=years, data=emissions - trend.values)
            trend_info = {
                "method": "broken-trend analysis",
                "number_of_breakpoints": len(breakpoints),
                "breakpoints": breakpoints
            }

        elif method in ["hp", "hodrick_prescott"]:
            # HP filter
            cycle, trend = hpfilter(
                emissions, lamb=100
            )  # 100 is typical for annual data
            residuals = pd.Series(index=years, data=emissions - trend)
            trend = pd.Series(trend, index=years)
            trend_info = {
                "method": "Hodrick-Prescott filter",
                "parameters": {"lambda": 100},
            }

        elif method == "hamilton":
            """Fit Hamilton's regression-based method"""

            # Horizon ahead at which we calculate y, can be specified, otherwise default 4
            h = kwargs.get("h", 4)
            # Length of timeseries that we use to predict y -> this is fixed by Hamilton's method
            p = 4
            n = len(years)
            y = emissions.copy()

            if n <= p:
                return None

            # Build regression matrix
            # Both X and Y are reduced in size by p + h
            # (you can't forecast before p+h: into the data, as you don't have sufficient data)
            # Lags are taken by calendar year (not position), so that no regression row
            # spans an ignored year - those rows are dropped instead
            full = pd.Series(index=years, data=y).reindex(range(int(years[0]), int(years[-1]) + 1))
            lagged = pd.concat(
                [full.rename("Y")] + [full.shift(lag).rename(lag) for lag in range(p, 0, -1)],
                axis=1,
            )
            lagged = lagged.loc[lagged.index >= years[0] + p + h].dropna()
            X = lagged.drop(columns="Y").values
            Y = lagged["Y"].values

            # Fit regression
            model = OLS(Y, X).fit()

            # Extract data
            residuals = pd.Series(index=lagged.index, data=model.resid)
            trend = pd.Series(index=lagged.index, data=model.fittedvalues)

            # Fill initial periods (Neil: I removed this as there is no real way to backfill residuals)
            # if max_lag > 0:
            #     trend_slope = (model.fittedvalues[1] - model.fittedvalues[0]) if len(model.fittedvalues) > 1 else 0
            #     for i in range(max_lag):
            #         full_trend[i] = model.fittedvalues[0] - trend_slope * (max_lag - i)
            #         full_cycle[i] = y[i] - full_trend[i]

            trend_info = {
                "method": "Hamilton (2018)",
                "parameters": f"h={h} (forecast horizon), p={p} (lags)",
                "r_squared": model.rsquared,
                "model": model,
            }

        elif method == "spline":

            # Nest decides the number of "knots" in the spline
            # If not provided, we use len(emissions)/3 (so a knot every 3 years)
            # (at least 8, the minimum for a cubic spline, so short series don't fail)
            n_knots = kwargs.get("n_knots", max(8, int(np.ceil(len(emissions) / 3))))

            s = kwargs.get("s", len(years))
    
            # Use automatic knot selection via smoothing
            spline_model = make_splrep(x=years, y=emissions, s=s, nest=n_knots)

            trend = pd.Series(index=years, data=spline_model(years))
            residuals = pd.Series(index=years, data=(spline_model(years) - emissions))

            # Get the actual knots that were used
            knots = len(spline_model.t)

            trend_info = {"method": "spline",
                          "number_of_knots": knots,
                          "s": s}

        else:
            raise ValueError(
                "Method must be one of 'lowess' // 'linear' // ' linear_w_autocorrelation' // + "
                "'broken_trend' // 'hp' // 'hamilton' //  // 'spline'"
            )

        return residuals, trend, trend_info

    def _analyze_autocorrelation(
        self, noise_type: str = "t-dist", t_df: Optional[float] = None
    ) -> Dict:
        """
        Analyze autocorrelation in temporally-ordered residuals to check for any remaining structure.
        
        Arguments:
            noise_type: type of noise distribution ('normal', 't-dist', 'empirical')
            t_df: Optional fixed degrees of freedom for t-distribution.
                If None, will be fitted from data when noise_type = 't-dist'

        Returns:
            Self for method chaining

        """

        # Reindex to a continuous range of years (NaN for ignored years), so that
        # lag-1 pairs that span an ignored year are skipped rather than treated as adjacent
        residuals = self.residuals.sort_index()
        full = residuals.reindex(
            range(int(residuals.index.min()), int(residuals.index.max()) + 1))

        # Calculate lag-1 autocorrelation via the acf function, and the probability of
        # autocorrelation via a lag-1 Ljung-Box test
        n = full.notna().sum()
        phi = acf(full.values, nlags=1, missing="conservative", fft=False)[1] # Ignore AR(0), i.e. no lag applied at all (which will be 1)
        q_stat = n * (n + 2) * phi**2 / (n - 1)
        p_autocorr = stats.chi2.sf(q_stat, 1)

        # Simple AR(1) model fitting, on consecutive-year pairs only
        pairs = pd.concat([full.rename("y"), full.shift(1).rename("x")], axis=1).dropna()
        if len(pairs) > 1:
            y = pairs["y"].values
            X = pairs["x"].values.reshape(-1, 1)

            ar_model = (
                LinearRegression()
            )  # TODO: Currently has non-zero intercept, which is weird...
            ar_model.fit(X, y)
            phi_fitted = ar_model.coef_[0]

            # Check that manual fit gives similar to the ACF results – up to a 10% tolerance is accepted
            try:
                assert np.isclose(phi_fitted, phi, rtol=1e-1)
            except AssertionError:
                print('ACF-estimated phi and manually estimated phi do not align')
                print(f'ACF_phi = {phi:.3f}, Manual_phi = {phi_fitted:.3f}')

            # Innovation residuals are the residuals left after accounting for autocorrelation
            innovation_residuals = y - ar_model.predict(X)
            sigma_innovation = np.std(innovation_residuals)
            mean_innovation = np.mean(innovation_residuals)

            # calculate t-distribution
            if noise_type == "t-dist":
                if t_df is not None:
                    # use provided df value
                    df_fitted = t_df
                else:
                    df_fitted, loc, scale = stats.t.fit(
                        innovation_residuals
                    )  # these are the innovation residuals (with AC removed)
                    print(f"Fitted t-distribution to innovations: df = {df_fitted:.1f}")
            else:
                df_fitted = None

        else:
            phi_fitted = 0
            sigma_innovation = np.std(self.residuals)
            mean_innovation = np.mean(self.residuals)

        self.autocorr_params = {
            "has_autocorr": p_autocorr < AR1_PVALUE_THRESHOLD,
            "phi": phi_fitted,
            "residuals": innovation_residuals,
            "sigma_residuals": sigma_innovation,
            "mean_residuals": mean_innovation,
            "is_stationary": abs(phi_fitted) < 1, # Check that autocorrelation is not growing
            "likelihood_of_autocorr": 1 - p_autocorr,
            "noise_type": noise_type,
            "t_df_fitted": df_fitted,
        }

        print("Autocorrelation analysis:\n")
        print(f"  Lag-1 autocorr: {phi_fitted:.3f}")
        print(f"  Residual σ (post-autocorrelation): {sigma_innovation:.1f}")
        print(f"  Has significant autocorr: {self.autocorr_params['has_autocorr']}")
        print(
            f"  Likelihood of autocorr: {self.autocorr_params['likelihood_of_autocorr']}"
        )

        return self.autocorr_params

    # =================================
    # C) Create a noise generator that can randomly reproduce noise
    # =================================

    def create_noise_generator(self):
        """
        Create noise generator

        Args:
            ignore_years [list]: List of specific years to ignore from the residuals
            because they are too large a variation and could bias the results (e.g. the GFC or COVID-19)

            clip_distribution [tuple]: Clip the distribution of residuals to exclude
            outlying percentiles (e.g. <5 and >95) before calculating

            [noise_type [str]: 'normal', 't', or 'empirical' # NOT USED?
                - 'normal': Fit normal distribution
                - 't-dist': Fit t-distribution
                - 'empirical': randomly draw from innovations] delete


        Returns:
            params: Dictionary with standardized parameter names
            generator: Function to generate noise samples
        """

        if self.residuals is None:
            raise ValueError(
                "Must call characterize_noise() first to calculate residuals"
            )

        if self.autocorr_params is None:
            self._analyze_autocorrelation()

        # Decide if to use an autocorrelated noise generator or a white noise generator
        # This decision depends on if autocorrelation is still observed in the noise structure

        if (
            self.autocorr_params["has_autocorr"]
            and self.autocorr_params["is_stationary"]
        ):
                    
            phi = self.autocorr_params["phi"]
            residuals_post_ar = self.autocorr_params["residuals"]
            sigma = self.autocorr_params["sigma_residuals"] #Focus on the standard deviation of the innovations
            mean = self.autocorr_params["mean_residuals"]
            noise_type = self.autocorr_params["noise_type"]
                
            
            print(f"Using AR(1) noise generator with φ={phi:.2f}, mean={mean:.3f}, sigma={sigma:.3f}")

            def ar1_noise_generator(
                size: int, initial_value: float | None = 0
            ) -> np.ndarray:
                """Generate AR(1) autocorrelated noise."""
                if noise_type == "normal":
                    innovations = np.random.normal(0, sigma, size)
                elif noise_type == "t-dist":
                    df_fitted = self.autocorr_params["t_df_fitted"]
                    innovations = np.random.standard_t(df_fitted, size) * sigma
                elif noise_type == "empirical":
                    innovations = np.random.choice(residuals_post_ar, size)
                else:
                    raise ValueError(
                        "noise_type must be 'normal', 't-dist' or 'empirical'"
                    )

                series = np.zeros(size)
                if initial_value is None:
                    # Draw a random start value based on the historical residuals
                    init_sigma = self.residuals.abs().mean()
                    if noise_type == "t-dist":
                        series[0] = np.random.standard_t(df_fitted) * init_sigma
                    else:
                        series[0] = np.random.normal(0, init_sigma)
                else:
                    series[0] = initial_value

                for t in range(1, size):
                    series[t] = phi * series[t - 1] + innovations[t]

                return series

            self.noise_generator = ar1_noise_generator

        else:
            # If no autocorrelation is identified, then just use a whitenoise generator
            # Here the residuals are taken straight from the decomposition, i.e. 
            # no shift to focus on innovation residuals + autocorrelation
            sigma = np.std(self.residuals)
            mean = np.mean(self.residuals)
            scale = np.max(self.residuals) - np.min(self.residuals)
            try:
                assert np.isclose(mean/scale, 0, atol=1e-2)
                print(f"Using white noise generator with mean=0,  σ={sigma:.3f}")
            except AssertionError:
                print('White noise residuals are not distributed close to zero')
                print(f'Mean of residuals = {mean}')

            def white_noise_generator(
                size: int
            ) -> np.ndarray:
                """Generate white noise."""
                noise_type = self.autocorr_params["noise_type"]
                if noise_type == "t-dist":
                    df_fitted = self.autocorr_params["t_df_fitted"]
                    return np.random.standard_t(df_fitted, size) * sigma
                else:
                    return np.random.normal(0, sigma, size)

            self.noise_generator = white_noise_generator

        return self.noise_generator

    def set_test_data(
        self, test_data: Union[pd.Series, pd.DataFrame]) -> "EmissionsPeakTest":
        """
        Set the test data (recent emissions showing potential decline).

        Args:
            test_data: Either a pd.Series with year as index and emissions as values,
                       or a pd.DataFrame with year as index and an "emissions" column.

        Returns:
            Self for method chaining
        """

        if isinstance(test_data, pd.Series):
            self.test_data = test_data.rename("emissions").to_frame().sort_index()
            self.test_data.index.name = "year"
        elif isinstance(test_data, pd.DataFrame):
            self.test_data = test_data.sort_index()
        else:
            raise ValueError("test_data must be either a DataFrame or Series")

        self.emissions_data = pd.concat([self.historical_data, self.test_data])
        self.emissions_data = self.emissions_data[~self.emissions_data.index.duplicated(keep="last")]

        # Calculate test trend
        self.test_slope, self.test_r2 = self._calculate_test_slope()

        print(
            f"Test data set: {self.test_data.index.min()}-{self.test_data.index.max()}"
        )
        print(
            f"Test slope: {self.test_slope:.2f} {self.unit} (R² = {self.test_r2:.3f})"
        )

        return self

    def _calculate_test_slope(self) -> Tuple[float, float]:
        """Calculate slope and R² for test data."""
        X = self.test_data.index.values.reshape(-1, 1)
        y = self.test_data["emissions"].values

        model = LinearRegression()
        model.fit(X, y)

        return model.coef_[0], model.score(X, y)

    def run_complete_bootstrap_test(
        self,
        n_bootstrap: int = 10000,
        null_hypothesis: str | float | int = "zero_trend",
        bootstrap_method: str = "ar_bootstrap",
        n_years_for_trend: int = 5,
        test_sequential: bool = False, 
        store_sequential_sims: bool = False, # choose to store simulations for plotting
        
    ) -> Dict:
        """
        Run complete bootstrap test with all enhancements.

        Args:
            null_hypothesis: Three options:
                - "recent_trend" (testing if new data is consistent with recent trend, using n of years defined below)
                - "zero_trend" (testing if new data is consistent with a zero trend)
                - "2pc_decline" (testing if new data is consistent with a 2% per year decline)
                - "3pc_decline" (testing if new data is consistent with a 3% per year decline)
                - float: Give a specific trend to test if new data is consistent with
            bootstrap_method: "ar_bootstrap", "white_noise"
            n_years_for_trend: int (number of years used to calculate recent trend)
            test_sequential: False (default). Whether to calculate the sequential test 
        """
        if self.noise_generator is None:
            self.create_noise_generator()

        print(f"Running complete bootstrap test...")
        if isinstance(null_hypothesis, str):
            print(f"  Null hypothesis: {null_hypothesis}")
        else:
            print(f"  Null hypothesis: trend of {null_hypothesis} / yr")
        print(f"  Bootstrap method: {bootstrap_method}")
        print(f"  Bootstrap samples: {n_bootstrap}")

        # Recalculate recent trend (as by default this is set to 5 years)

        if n_years_for_trend != 5:
            recent_data = self.historical_data.tail(n_years_for_trend) # Exclude most recent year to avoid overlap with test data
            X_recent = recent_data.index.values.reshape(-1, 1) # extract years for regression
            y_recent = recent_data["emissions"].values
            model_recent = LinearRegression()
            model_recent.fit(X_recent, y_recent)
            self.recent_historical_trend = model_recent.coef_[0]

        # bootstrapping
        bootstrap_slopes = self._generate_bootstrap_slopes(
            n_bootstrap, null_hypothesis, bootstrap_method
        )
        
        self.bootstrap_results = {
            "test_slope": self.test_slope,
            "test_r2": self.test_r2,
            "recent_historical_trend": self.recent_historical_trend,
            "bootstrap_slopes": bootstrap_slopes,
            "null_hypothesis": null_hypothesis,
            "bootstrap_method": bootstrap_method,
            "n_bootstrap": n_bootstrap,
            "autocorr_phi": self.autocorr_params["phi"],
            # 'n_segments': len(self.optimal_segments['best']['segments'])
        }

        # Calculate p-values
        print("using slope-based test method")
        p_value_one_tail = np.sum(bootstrap_slopes <= self.test_slope) / len(
            bootstrap_slopes
        )

        # Effect size: how many standard deviations below the null distribution
        null_mean = np.mean(bootstrap_slopes)
        null_std = np.std(bootstrap_slopes)
        effect_size = (null_mean - self.test_slope) / null_std if null_std > 0 else 0
        
        self.bootstrap_results.update({
            "p_value_one_tail": p_value_one_tail,
            "significant_at_0.1": p_value_one_tail < 0.1,
            "significant_at_0.05": p_value_one_tail < 0.05,
            "significant_at_0.01": p_value_one_tail < 0.01,
            "effect_size": effect_size,
            "null_mean": null_mean,
            "null_std": null_std,
        })
        
        if test_sequential:
        #### NEW: Sequential test method
            p_flat, av_p_step, p_steps_list, sims_dict = self._compute_sequential_p_flat(
                n_bootstrap,
                store_sims=store_sequential_sims
                                        ) # calculate probability of being consistent with a flat trend

            self.bootstrap_results.update({
                "p_sequential_multiplied": p_flat,
                "p_sequential_averaged": av_p_step,
                "p_sequential_steps": p_steps_list,
                "simulated_values_dict": sims_dict,
            })


            print("Results:")
            print(f"  P-value sequential method: {p_flat:.4f}")

        else:
            # Just print the slope-based method otherwise
            print(f"  P-value slope method: {p_value_one_tail:.4f}")
            print(
                f"  Significant at α=0.1: {self.bootstrap_results['significant_at_0.1']}"
            )
            print(f"  Effect size: {effect_size:.2f} standard deviations")

        return

    def _generate_bootstrap_slopes(
        self, n_bootstrap: int, null_hypothesis: str | float, bootstrap_method: str
    ) -> np.ndarray:
        """Generate bootstrap slope distribution."""
        bootstrap_slopes = []
        n_test_points = len(self.test_data)

        years = self.test_data.index.values  # Arbitrary years for test

        # Baseline emissions level
        baseline_emissions = self.test_data["emissions"].iloc[0]
        baseline_year = self.test_data.index[0]

        # Null hypothesis trend
        if null_hypothesis == "recent_trend":
            null_trend = self.recent_historical_trend
        elif isinstance(null_hypothesis, (int,float)):
            null_trend = null_hypothesis
        elif null_hypothesis == "2pc_decline":
            null_trend = -0.02 * baseline_emissions
        elif null_hypothesis == "3pc_decline":
            null_trend = -0.03 * baseline_emissions
        else:  # zero_trend
            null_trend = 0.0

        for i in range(n_bootstrap):
            # Generate null hypothesis emissions trajectory
            trend_component = null_trend * (
                years - baseline_year
            )  # replace years[0] with baseline_year
            base_emissions = baseline_emissions + trend_component

            # Add autocorrelated noise
            # Mirror the condition in create_noise_generator() exactly:
            # use the AR(1) generator only if autocorrelation is both present and stationary.
            if (
                bootstrap_method == "ar_bootstrap"
                and self.autocorr_params["has_autocorr"]
                and self.autocorr_params["is_stationary"]
            ):
                # Use AR(1) noise generator. This creates the noise for all n_test_points in one go
                noise = self.noise_generator(n_test_points, initial_value=None)
            else:
                # White noise: delegate to self.noise_generator (white_noise_generator),
                # which was set up with the correct sigma and distribution in create_noise_generator().
                noise = self.noise_generator(n_test_points)

            null_emissions = base_emissions + noise

            # Calculate slope
            X = years.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, null_emissions)
            bootstrap_slopes.append(model.coef_[0])

        return np.array(bootstrap_slopes)
    
    # NEW function to calculate sequential p-flat
    def _compute_sequential_p_flat(self, n_bootstrap: int,
                                   null_hypothesis: str | float = "zero_trend",
                                   store_sims = False) -> float:
        """
        This method calculates the probability of observing the test data under the null hypothesis by 
        sequentially evaluating the probability of each step in the test data being as low as observed, 
        given the previous steps and the null hypothesis. It accounts for autocorrelation in the noise 
        and allows for different null hypotheses regarding the trend.

        Args:            n_bootstrap: Number of bootstrap simulations to run for each step
            null_hypothesis: Three options:
                - "recent_trend" (testing if new data is consistent with recent trend, using n of years defined below)
                - "zero_trend" (testing if new data is consistent with a zero trend)
                - "2pc_decline" (testing if new data is consistent with a 2% per year decline)
                - "3pc_decline" (testing if new data is consistent with a 3% per year decline)
                - float: Give a specific trend to test if new data is consistent with
         Returns:            p_flat: Overall probability of observing the test data under the null hypothesis, calculated as the product of the stepwise probabilities
            av_p_step: Average of the stepwise probabilities (for interpretation)
            p_steps_series: pd.Series with the stepwise probabilities for each year
            sims_dict: Dictionary with the simulated distributions for each step, keyed by year
        
    
        """

        obs = self.test_data["emissions"].values
        years = self.test_data.index.values
        n_steps = len(obs) - 1 
        start_value = obs[0]

        p_flat = 1.0
        log_p_flat = 0.0
        p_steps_dict = {}  # store p_step for each step, keyed by year
        sims_dict = {}  # store simulated distributions for each step, keyed by year

        # Null hypothesis trend
        if null_hypothesis == "recent_trend":
            null_trend = self.recent_historical_trend
        elif isinstance(null_hypothesis, (int,float)):
            null_trend = null_hypothesis
        elif null_hypothesis == "2pc_decline":
            null_trend = -0.02 * baseline_emissions
        elif null_hypothesis == "3pc_decline":
            null_trend = -0.03 * baseline_emissions
        else:  # zero_trend
            null_trend = 0.0

        for t in range(n_steps):

            y_prev = obs[t]
            y_next_obs = obs[t + 1]

            sims = []
            null_emissions_next = start_value + null_trend*(t+1)  # expected next value under null hypothesis, given starting value and null trend
            null_emissions_prev = start_value + null_trend*t  # expected previous value under null hypothesis, given starting value and null trend
            for _ in range(n_bootstrap):

                # generate ONE step of noise
                if (
                    self.autocorr_params["has_autocorr"]
                    and self.autocorr_params["is_stationary"]
                ):
                    noise = self.noise_generator(1, initial_value=None)[0]
                else:
                    noise = self.noise_generator(1)[0]

                # AR(1) step (adapt if your exact form differs)
                residual_prev = y_prev - null_emissions_prev  # innovation from previous step (will be 0 for the first step)
                y_next_sim = null_emissions_next + self.autocorr_params["phi"] * (residual_prev) + noise  # null trend + autocorrelation from previous step + noise)

                sims.append(y_next_sim)

            sims = np.array(sims)

            # probability of being at least as low as observed
            p_step = np.mean(sims <= y_next_obs)
            p_steps_dict[years[t + 1]] = p_step
            
            if store_sims:
                sims_dict[years[t + 1]] = sims  # Store the simulated values for each step
            print(f"Step {t+1}/{n_steps}: P(simulated ≤ observed) = {p_step:.4f}")
            log_p_flat += np.log(p_step + 1e-10)  # add small value to avoid log(0)

            # accumulate sequentially
            # p_flat *= p_step
            p_flat = np.exp(log_p_flat) # Use log probabilities to increase stability for long sequences

        p_steps_series = pd.Series(p_steps_dict, name="p_step")
        p_steps_series.index.name = "year"
        av_p_step = p_steps_series.mean()

        return p_flat, av_p_step, p_steps_series, sims_dict

    def interpret_results(self, verbose: bool = True) -> Dict[str, str]:
        """
        Provide interpretation of the test results.

        Args:
            verbose: Whether to print interpretation

        Returns:
            Dictionary with interpretation strings
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        results = self.bootstrap_results
        strength = None

        # Determine trend direction
        if results["test_slope"] < 0:
            direction = "decline"
            trend_desc = "declining"
        else:
            direction = "increase"
            trend_desc = "increasing"

        # Statistical significance
        if results["significant_at_0.01"]:
            strength = "very strong"
        elif results["significant_at_0.05"]:
            strength = "strong"
        elif results["significant_at_0.1"]:
            strength = "moderate"
        else:
            strength = "insufficient"

        significance = f"{strength} evidence"

        # Peak conclusion
        if direction == "decline" and strength != "insufficient":
            peak_conclusion = "Evidence that CO₂ emissions have peaked"
            confidence = strength
        elif direction == "decline":
            peak_conclusion = (
                "Declining trend present but not statistically significant"
            )
            confidence = "low"
        else:
            peak_conclusion = "No evidence of emissions peak"
            confidence = "none"

        interpretation = {
            "direction": direction,
            "trend_description": trend_desc,
            # "significance": significance,
            "peak_conclusion": peak_conclusion,
            "confidence_in_peak": confidence,
            "p_value": f"{results['p_value_one_tail']:.4f}",
            "slope": f"{results['test_slope']:.1f} {self.unit}",
        }

        if verbose:
            print("\n" + "=" * 50)
            print("INTERPRETATION OF RESULTS")
            print("=" * 50)
            print(f"Observed trend: {trend_desc} at {interpretation['slope']}")
            print(
                f"Statistical evidence: {significance} (p = {interpretation['p_value']})"
            )
            print(f"Conclusion: {peak_conclusion}")
            print("=" * 50)

        return interpretation

    def plot_analysis(
        self, figsize: Tuple[int, int] = (15, 10), save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization of the analysis.

        Args:
            figsize: Figure size tuple
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        # fig, axes = plt.subplots(2, 3, figsize=figsize)

        f = plt.figure(layout=None, figsize=(15, 15))
        gs = f.add_gridspec(
            nrows=16, ncols=7, left=0.05, right=0.75, hspace=0.1, wspace=0.05
        )

        a0 = f.add_subplot(gs[:5, :3])
        a1 = f.add_subplot(gs[:2, 4:])
        a2 = f.add_subplot(gs[3:5, 4:])
        a3 = f.add_subplot(gs[6:10, :3])
        a4 = f.add_subplot(gs[6:10, 4:])
        a5 = f.add_subplot(gs[11:, :])
        axes = [a0, a1, a2, a3, a4, a5]

        # 1. Historical data with test data overlay
        self._plot_historical_and_test_data(axes[0])

        # 2. Split of historical data into noise vs. signal
        self._plot_historical_trend(axes[1])
        self._plot_historical_noise(axes[2])

        # 3. Noise distribution
        self._plot_noise_distribution(axes[3])

        # 4. Bootstrap results
        self._plot_bootstrap_results(axes[4])

        # 5. Summary statistics
        self._plot_summary_statistics(axes[5])

        f.suptitle(
            f"Results:\n{self.variable}:{self.region}", fontweight="bold", x=0.4, y=0.95
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return

    def _plot_historical_and_test_data(self, ax: plt.Axes, historical_kwargs=None, test_kwargs=None) -> None:
        """Plot historical data with test data overlay

        Args:
            ax: matplotlib Axes object
            historical_kwargs: dict of plotting kwargs for historical data (default: {"color": "b", "linestyle": "-", "alpha": 0.7})
            test_kwargs: dict of plotting kwargs for test data (default: {"color": "k", "marker": "+", "linestyle": "", "linewidth": 1, "markersize": 5})
        """
        # Set default kwargs
        default_historical_kwargs = {"color": "b", "linestyle": "-", "alpha": 0.7, "label": "Historical emissions"}
        default_test_kwargs = {"color": "k", "marker": "+", "linestyle": "", "linewidth": 1, "markersize": 5, "label": "Recent test data"}

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        hist_plot_kwargs = {**default_historical_kwargs, **(historical_kwargs or {})}
        test_plot_kwargs = {**default_test_kwargs, **(test_kwargs or {})}

        ax.plot(
            self.historical_data.index,
            self.historical_data["emissions"],
            **hist_plot_kwargs,
        )
        ax.plot(
            self.test_data.index,
            self.test_data["emissions"],
            **test_plot_kwargs,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title("Historical data and Test data")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_historical_trend(self, ax: plt.Axes, **kwargs) -> None:
        """Plot historical trend data

        Args:
            ax: matplotlib Axes object
            **kwargs: Additional plotting arguments (default: {"color": "b", "linestyle": "-", "alpha": 0.7})
        """
        # Set default kwargs
        default_kwargs = {"color": "b", "linestyle": "-", "alpha": 0.7, "label": "Trend"}

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        plot_kwargs = {**default_kwargs, **kwargs}

        ax.plot(
            self.trend.index,
            self.trend.values,
            **plot_kwargs,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title(f"Historical Trend on {self.variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_historical_noise(self, ax: plt.Axes, **kwargs) -> None:
        """Plot historical noise data

        Args:
            ax: matplotlib Axes object
            **kwargs: Additional plotting arguments (default: {"color": "orange", "style": "-"})
        """
        # Set default kwargs
        default_kwargs = {"color": "orange", "style": "-", "label": "Residuals"}

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        plot_kwargs = {**default_kwargs, **kwargs}

        self.residuals.plot(
            ax=ax, **plot_kwargs
        )

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{self.variable}\n({self.unit})")
        ax.set_title(f"Historical Residuals on {self.variable}")
        ax.legend()
        ax.axhline(y=0, color="k", lw=1)
        ax.grid(True, alpha=0.3)

    def _plot_noise_distribution(self, ax: plt.Axes, residuals_kwargs=None, innovations_kwargs=None) -> None:
        """Plot the fitted noise distribution.

        Args:
            ax: matplotlib Axes object
            residuals_kwargs: dict of plotting kwargs for residuals histogram
                (default: {"bins": 30, "density": True, "alpha": 0.5, "color": "skyblue", "edgecolor": "blue", "linewidth": 0.5})
            innovations_kwargs: dict of plotting kwargs for innovations histogram
                (default: {"bins": 30, "density": True, "alpha": 0.5, "color": "green", "edgecolor": "darkgreen", "linewidth": 0.5})
        """
        # Set default kwargs
        default_residuals_kwargs = {
            "bins": 30,
            "density": True,
            "alpha": 0.5,
            "color": "skyblue",
            "label": "Original residuals",
            "edgecolor": "blue",
            "linewidth": 0.5
        }
        default_innovations_kwargs = {
            "bins": 30,
            "density": True,
            "alpha": 0.5,
            "color": "green",
            "label": "Innovations (after AR removal)",
            "edgecolor": "darkgreen",
            "linewidth": 0.5
        }

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        res_plot_kwargs = {**default_residuals_kwargs, **(residuals_kwargs or {})}
        innov_plot_kwargs = {**default_innovations_kwargs, **(innovations_kwargs or {})}

        # Plot original residuals
        ax.hist(
            self.residuals,
            **res_plot_kwargs
        )

        # Plot innovations (if autocorrelation was detected and removed)
        if self.autocorr_params["has_autocorr"] and self.autocorr_params["is_stationary"]:
            ax.hist(
                self.autocorr_params['residuals'],
                **innov_plot_kwargs
            )
        # Overlay fitted distribution
        # # TODO: Does this make sense to do with an autocorrelated system?
        # if self.noise_params["type"] == "normal":
        #     x_range = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        #     fitted_density = stats.norm.pdf(
        #         x_range, self.noise_params["mu"], self.noise_params["sigma"]
        #     )
        #     ax.plot(x_range, fitted_density, "r-", linewidth=2, label="Fitted normal")

        ax.set_xlabel(f"Residuals {self.unit}")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Noise Distribution"
        )  # (σ = {self.noise_params["sigma"]:.0f} Mt)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_bootstrap_results(self, ax: plt.Axes, hist_kwargs=None, vline_kwargs=None) -> None:
        """Plot bootstrap distribution with observed slope.

        Args:
            ax: matplotlib Axes object
            hist_kwargs: dict of plotting kwargs for histogram
                (default: {"bins": 50, "density": True, "alpha": 0.7, "color": "lightgreen"})
            vline_kwargs: dict of plotting kwargs for vertical line
                (default: {"color": "red", "linewidth": 2})
        """
        # Set default kwargs
        default_hist_kwargs = {
            "bins": 50,
            "density": True,
            "alpha": 0.7,
            "color": "lightgreen",
            "label": "Bootstrap slopes\n(null hypothesis)",
        }
        default_vline_kwargs = {
            "color": "red",
            "linewidth": 2,
            "label": f'Observed slope\n{self.bootstrap_results["test_slope"]:.1f} {self.unit}',
        }

        # Merge user-provided kwargs with defaults (user kwargs take precedence)
        hist_plot_kwargs = {**default_hist_kwargs, **(hist_kwargs or {})}
        vline_plot_kwargs = {**default_vline_kwargs, **(vline_kwargs or {})}

        ax.hist(
            self.bootstrap_results["bootstrap_slopes"],
            **hist_plot_kwargs,
        )

        ax.axvline(
            self.bootstrap_results["test_slope"],
            **vline_plot_kwargs,
        )

        ax.set_xlabel(f"Slope: {self.unit}")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap Distribution vs Observed Slope")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_summary_statistics(self, ax: plt.Axes, signif_level: float = 0.1) -> None:
        """Plot summary statistics and interpretation."""
        ax.axis("off")

        results = self.bootstrap_results

        # Create summary text
        summary_text = f"""
        STATISTICAL TEST RESULTS
        ________________________
        
        Test Data: {len(self.test_data)} years
        Observed Slope: {results['test_slope']:.2f} {self.unit}
        R²: {results['test_r2']:.3f}
        
        Bootstrap Analysis:
        • Samples: {results['n_bootstrap']:,}
        • P-value: {results['p_value_one_tail']:.4f}
        • Significant at p={signif_level}: {results[f'significant_at_{signif_level}']}
        
        Noise Characteristics:
        • Method: {self.trend_info['method']}
        • Autocorrelation present: {self.autocorr_params['has_autocorr']}
        • Autocorrelation: {self.autocorr_params['phi']}
        • Std Dev: {self.autocorr_params['sigma_residuals']:.3f} {self.unit}
        
        CONCLUSION:
        """

        interpretation = self.interpret_results(verbose=False)
        summary_text += interpretation["peak_conclusion"]

        ax.text(
            0.5,
            0.95,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
            fontfamily="monospace",
        )

    def export_results(self, filepath: str, signif_level: float = 0.1) -> None:
        """
        Export analysis results to CSV file.

        Args:
            filepath: Path for output CSV file
        """
        if self.bootstrap_results is None:
            raise ValueError("Must run bootstrap test first")

        # Prepare results data
        results_data = {"metric": [], "value": []}

        # Add test results
        results_data["metric"].extend(
            [
                "test_slope_mt_per_year",
                "test_r_squared",
                "p_value_one_tail",
                "significance",
                "n_bootstrap_samples",
                # "noise_std_mt",
                "n_test_years",
                "test_year_start",
                "test_year_end",
            ]
        )

        results_data["value"].extend(
            [
                self.bootstrap_results["test_slope"],
                self.bootstrap_results["test_r2"],
                self.bootstrap_results["p_value_one_tail"],
                self.bootstrap_results[f"significant_at_{signif_level}"],
                self.bootstrap_results["n_bootstrap"],
                # self.noise_params["sigma"],
                len(self.test_data),
                self.test_data.index.min(),
                self.test_data.index.max(),
            ]
        )

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")


# Example usage and demonstration
if __name__ == "__main__":
    print("Emissions Peak Detection Test")
    print("=" * 50)

    # Initialize test with random state for reproducibility
    peak_test = EmissionsPeakTest()

    # Method chaining example
    peak_test.load_historical_data(
        "gcb_hist_co2.csv",
        # emissions_col="fossil_co2_emissions",
        year_range=range(1970, 2024),
    )

    peak_test.set_test_data(
        [
            (2025, 37700),
            (2026, 37400), #37400
            (2027, 37100), # 37100
            # (2028, 37400)
        ]
    )
    peak_test.characterize_noise(method="spline", noise_type="t-dist", s=25, include_test_data=False)
    peak_test.create_noise_generator()
    peak_test.run_complete_bootstrap_test(bootstrap_method="ar_bootstrap")

    print(f"P-value (one-tail): {peak_test.bootstrap_results['p_value_one_tail']:.4f}")
    print(f"P-value (sequential): {peak_test.bootstrap_results['p_sequential_averaged']:.4f}")  
    print(f"P-value (sequential, list): {peak_test.bootstrap_results['p_sequential_steps']}") 
    # Get interpretation
    interpretation = peak_test.interpret_results(verbose=True)

    # Create visualizations
    peak_test.plot_analysis()

    # Export results
    # peak_test.export_results('_peak_test_results.csv')

    print("\nAnalysis complete!")
    print("\nKey methods:")
    print("• load_historical_data() or simulate_historical_data()")
    print("• characterize_noise()")
    print("• set_test_data()")
    print("• run_bootstrap_test()")
    print("• interpret_results()")
    print("• plot_analysis()")
    print("• export_results()")
