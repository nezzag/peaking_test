#!/usr/bin/env python3
"""
Script: queries/Q1_methods_comparison/Q1_methods_comparison.py
Query: For the World and G20 members, in EDGAR energy CO2 and GCB fossil CO2, does
compare_methods always find a decomposition method whose noise passes the diagnostic
tests (autocorrelation, ARCH, normality), or are there regions/datasets where every
method fails?

Method selection uses the priority list (peak_tester.METHOD_PRIORITY), no backtest, via
EmissionsPeakTest.compare_methods().

Outputs (all numbers computed here):
  Q1_methods_comparison.md     summary, selection table and per-method status grids
  Q1_1_method_results.csv      one row per dataset x region x method
  Q1_2_selection.csv           one row per dataset x region: selected method, exclusions

Inputs:
  data/processed/edgar_energy_co2.csv    Emissions|CO2|Energy (EDGAR)
  data/processed/gcb_hist_co2.csv        Emissions|CO2|Fossil (GCB)

Usage (from repo root):
    python queries/Q1_methods_comparison/Q1_methods_comparison.py
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from peak_tester import METHODS, METHOD_PRIORITY, EmissionsPeakTest  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
QUERY = "Q1"

DATASETS = {
    "EDGAR": ("edgar_energy_co2.csv", "Emissions|CO2|Energy"),
    "GCB": ("gcb_hist_co2.csv", "Emissions|CO2|Fossil"),
}
YEAR_RANGE = range(1970, 2025)  # common to both datasets (EDGAR 2025 left out)

# G20: 19 member countries + EU27 (the African Union is not in either dataset), plus World
REGIONS = [
    "WLD", "EU27",
    "ARG", "AUS", "BRA", "CAN", "CHN", "DEU", "FRA", "GBR", "IDN", "IND",
    "ITA", "JPN", "KOR", "MEX", "RUS", "SAU", "TUR", "USA", "ZAF",
]

# Short labels for the status grids
STATUS_LABELS = {
    "pass": "✓",
    "pass_with_exclusions": "✓",
    "fail_autocorrelation": "✗ AC",
    "fail_arch": "✗ ARCH",
    "fail_normality_heavy_tails": "✗ tails",
    "fail_normality_skewed": "✗ skew",
    "fail_structural_break": "✗ break",
    "error": "error",
}


def run_one(dataset: str, region: str) -> pd.DataFrame:
    """Run compare_methods for one dataset/region; return per-method results."""
    file, variable = DATASETS[dataset]
    tester = EmissionsPeakTest()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tester.load_historical_data(file, region=region, variable=variable, year_range=YEAR_RANGE)
        tester.compare_methods(selection="priority")

    results = tester.method_comparison.drop(columns=["noise"], errors="ignore").reset_index()
    results.insert(0, "region", region)
    results.insert(0, "dataset", dataset)
    results["selected"] = results["method"] == tester.selected_method
    return results


def cell(row: pd.Series) -> str:
    """Status grid cell: pass/fail label plus excluded or break years."""
    label = STATUS_LABELS.get(row["status"], row["status"])
    if row["status"] == "fail_structural_break":
        years = row["break_years"]
        label += f" {years[0]}–{years[-1]}"
    elif row["status"] == "pass_with_exclusions":
        label += " " + ", ".join(str(y) for y in row["years_excluded"])
    if row["selected"]:
        label = f"**{label}**"
    return label


def md_table(df: pd.DataFrame) -> str:
    """DataFrame -> markdown table (index included)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main():
    all_results = pd.concat(
        [run_one(dataset, region) for dataset in DATASETS for region in REGIONS],
        ignore_index=True,
    )

    # ---- Per dataset x region selection
    selection = []
    for (dataset, region), grp in all_results.groupby(["dataset", "region"], sort=False):
        sel = grp[grp["selected"]]
        warns = grp["warning"].dropna().unique().tolist()
        selection.append({
            "dataset": dataset,
            "region": region,
            "selected_method": sel["method"].iloc[0] if len(sel) else None,
            "years_excluded": sel["years_excluded"].iloc[0] if len(sel) else None,
            "n_passing": int(grp["passes"].sum()),
            "passing_methods": ", ".join(grp.loc[grp["passes"], "method"]),
            "warnings": "; ".join(f"{m}: {w}" for m, w in grp[["method", "warning"]].dropna().values),
        })
    selection = pd.DataFrame(selection)

    all_results.to_csv(OUT_DIR / f"{QUERY}_1_method_results.csv", index=False)
    selection.to_csv(OUT_DIR / f"{QUERY}_2_selection.csv", index=False)

    # ---- Markdown
    n_total = len(selection)
    no_method = selection[selection["selected_method"].isna()]
    sel_counts = selection["selected_method"].value_counts()

    def fmt_sel(r):
        if r["selected_method"] is None:
            return "**none**"
        excl = r["years_excluded"]
        return r["selected_method"] + (f" (excl. {', '.join(map(str, excl))})" if excl else "")

    sel_table = selection.assign(sel=selection.apply(fmt_sel, axis=1)).pivot(
        index="region", columns="dataset", values="sel").reindex(REGIONS)
    n_pass_table = selection.pivot(index="region", columns="dataset", values="n_passing").reindex(REGIONS)
    sel_table = sel_table.join(n_pass_table.add_prefix("n passing, "))
    sel_table.index.name = "region"

    # Methods passing, share of regions
    pass_share = (all_results.groupby(["dataset", "method"])["passes"].mean()
                  .unstack(0).reindex(METHODS).map(lambda v: f"{v:.0%}"))
    pass_share.index.name = "method"

    # Failure reasons
    fail_counts = (all_results[~all_results["passes"]]
                   .groupby(["dataset", "status"]).size().unstack(0).fillna(0).astype(int))
    fail_counts.index.name = "status"

    # Exclusions in selected methods
    excl = selection.dropna(subset=["selected_method"]).explode("years_excluded").dropna(subset=["years_excluded"])
    excl_counts = excl.groupby("years_excluded")["region"].apply(
        lambda s: ", ".join(f"{r}" for r in s)).to_frame("regions (dataset)")
    excl_counts["regions (dataset)"] = excl.groupby("years_excluded").apply(
        lambda g: ", ".join(f"{r} ({d})" for r, d in zip(g["region"], g["dataset"])))
    excl_counts.index.name = "year excluded"

    warnings_rows = selection[selection["warnings"] != ""][["dataset", "region", "warnings"]]

    # Regions where no method works in both datasets vs one
    no_method_regions = no_method.groupby("region")["dataset"].apply(list)
    both = [r for r, ds in no_method_regions.items() if len(ds) == len(DATASETS)]
    one = {r: ds[0] for r, ds in no_method_regions.items() if len(ds) == 1}

    answer = [
        f"**A method is selected for {n_total - len(no_method)} of {n_total} region-dataset "
        f"combinations; {len(no_method)} have no method passing all tests"
        + (f": {', '.join(f'{r} ({d})' for r, d in zip(no_method['region'], no_method['dataset']))}." if len(no_method) else ".")
        + "**",
        "",
        f"- No method in either dataset: {', '.join(both) if both else 'none'}.",
        f"- No method in one dataset only: "
        + (", ".join(f"{r} ({d})" for r, d in one.items()) if one else "none") + ".",
        "- Selected methods (priority list "
        + " > ".join(METHOD_PRIORITY) + "): "
        + ", ".join(f"{m} {n}" for m, n in sel_counts.items()) + ".",
        f"- Most common failure reason across all methods: "
        + ", ".join(f"{d}: {fail_counts[d].idxmax()} ({fail_counts[d].max()})" for d in fail_counts.columns)
        + ".",
    ]

    grids = []
    for dataset in DATASETS:
        sub = all_results[all_results["dataset"] == dataset].copy()
        sub["cell"] = sub.apply(cell, axis=1)
        grid = sub.pivot(index="region", columns="method", values="cell").reindex(index=REGIONS, columns=METHODS)
        grid.index.name = "region"
        grids.append(f"### {dataset}\n\n{md_table(grid)}")

    md = f"""# {QUERY}: Does compare_methods always select a method for the World and G20, in EDGAR and GCB CO2?

## Contents

- [Answer](#answer)
- [1. Selected method](#1-selected-method)
- [2. Status of every method](#2-status-of-every-method)
- [3. Pass rates and failure reasons](#3-pass-rates-and-failure-reasons)
- [4. Excluded years and warnings](#4-excluded-years-and-warnings)
- [Next steps](#next-steps)
- [Settings](#settings)

## Answer

{chr(10).join(answer)}

Scope: World, EU27 and 19 G20 countries; EDGAR `Emissions|CO2|Energy` and GCB `Emissions|CO2|Fossil`; 1970–2024.

## 1. Selected method

Priority selection among methods passing all tests. Excluded outlier years in brackets.

{md_table(sel_table)}

## 2. Status of every method

✓ = passes (excluded years listed); ✗ = fails, with the reason: AC = autocorrelation remaining after AR(1) (Ljung-Box), ARCH = heteroskedasticity, tails / skew = non-normality (Jarque-Bera) from heavy tails / skewness that up to 3 GESD outlier exclusions could not fix, break = consecutive same-sign outliers (structural break). **Bold** = selected.

{(chr(10) * 2).join(grids)}

## 3. Pass rates and failure reasons

Share of the {len(REGIONS)} regions where each method passes:

{md_table(pass_share)}

Failure reasons, count of region × method combinations:

{md_table(fail_counts)}

## 4. Excluded years and warnings

Outlier years excluded in the selected method:

{md_table(excl_counts) if len(excl_counts) else "None."}

Warnings (structural breaks, and outliers flagged in the last 3 historical years that were not excluded):

{md_table(warnings_rows.set_index("dataset")) if len(warnings_rows) else "None."}

## Next steps

1. **Investigate why India fails across all methods in both datasets.** The failure reasons differ by method (autocorrelation, ARCH, skew, and a possible 2019–2020 structural break), and 2023 is flagged as an outlier but protected. Candidates to test: forced breakpoints, a shorter historical period, and the t-distribution once it is added.
2. **Consider whether the exclusion of years needs to be stronger.** 2009 (the global financial crisis) is almost never removed. Outlier detection only runs when ARCH or normality fail, and often removing 2020 alone is enough to pass. Flexible trends (hp, spline, lowess) and broken_trend breakpoints can also bend into 2009 and absorb part of the shock. Keeping a genuine shock in the noise inflates σ, which widens the null distribution and could make the peak test too conservative (false negatives: real peaks not detected). Options include always testing known shock years (e.g. 2009, 2020) as exclusion candidates, or running GESD whether or not the tests fail.

## Settings

- `EmissionsPeakTest.compare_methods(selection="priority")`, alpha = 0.05, up to 3 GESD exclusions, last 3 historical years protected.
- Data: `data/processed/edgar_energy_co2.csv`, `data/processed/gcb_hist_co2.csv`; years {YEAR_RANGE.start}–{YEAR_RANGE.stop - 1}.
- Outputs: `{QUERY}_1_method_results.csv` (per method), `{QUERY}_2_selection.csv` (per region).
"""
    (OUT_DIR / f"{QUERY}_methods_comparison.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
