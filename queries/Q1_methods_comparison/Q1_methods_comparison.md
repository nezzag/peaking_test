# Q1: Does compare_methods always select a method for the World and G20, in EDGAR and GCB CO2?

## Contents

- [Answer](#answer)
- [1. Selected method](#1-selected-method)
- [2. Status of every method](#2-status-of-every-method)
- [3. Pass rates and failure reasons](#3-pass-rates-and-failure-reasons)
- [4. Excluded years and warnings](#4-excluded-years-and-warnings)
- [Next steps](#next-steps)
- [Settings](#settings)

## Answer

**A method is selected for 40 of 42 region-dataset combinations; 2 have no method passing all tests: IND (EDGAR), IND (GCB).**

- No method in either dataset: IND.
- No method in one dataset only: none.
- Selected methods (priority list broken_trend > linear_w_autocorrelation > linear > hp > spline > lowess > hamilton): broken_trend 28, linear_w_autocorrelation 7, hp 2, spline 2, linear 1.
- Most common failure reason across all methods: EDGAR: fail_autocorrelation (34), GCB: fail_autocorrelation (29).

Scope: World, EU27 and 19 G20 countries; EDGAR `Emissions|CO2|Energy` and GCB `Emissions|CO2|Fossil`; 1970–2024.

## 1. Selected method

Priority selection among methods passing all tests. Excluded outlier years in brackets.

| region | EDGAR | GCB | n passing, EDGAR | n passing, GCB |
|---|---|---|---|---|
| WLD | linear_w_autocorrelation (excl. 2010, 2020) | linear_w_autocorrelation (excl. 2010, 2020) | 2 | 4 |
| EU27 | broken_trend | broken_trend | 7 | 6 |
| ARG | linear_w_autocorrelation | broken_trend | 4 | 4 |
| AUS | broken_trend | broken_trend | 3 | 3 |
| BRA | broken_trend | linear_w_autocorrelation | 3 | 2 |
| CAN | broken_trend | broken_trend (excl. 2020) | 7 | 4 |
| CHN | broken_trend | linear_w_autocorrelation | 3 | 3 |
| DEU | broken_trend | broken_trend | 7 | 7 |
| FRA | broken_trend | hp | 7 | 4 |
| GBR | broken_trend (excl. 1979) | broken_trend (excl. 1979) | 7 | 6 |
| IDN | broken_trend (excl. 2020) | spline | 2 | 1 |
| IND | **none** | **none** | 0 | 0 |
| ITA | broken_trend | broken_trend | 7 | 7 |
| JPN | broken_trend | broken_trend | 6 | 6 |
| KOR | linear_w_autocorrelation (excl. 1998, 2010) | hp (excl. 1998, 2010, 2020) | 3 | 1 |
| MEX | broken_trend (excl. 2020) | broken_trend (excl. 2020) | 6 | 6 |
| RUS | broken_trend | broken_trend | 4 | 2 |
| SAU | linear_w_autocorrelation | broken_trend | 3 | 6 |
| TUR | linear (excl. 2017, 2021) | spline | 3 | 2 |
| USA | broken_trend | broken_trend (excl. 2020) | 6 | 6 |
| ZAF | broken_trend | broken_trend (excl. 2004) | 5 | 7 |

## 2. Status of every method

✓ = passes (excluded years listed); ✗ = fails, with the reason: AC = autocorrelation remaining after AR(1) (Ljung-Box), ARCH = heteroskedasticity, tails / skew = non-normality (Jarque-Bera) from heavy tails / skewness that up to 3 GESD outlier exclusions could not fix, break = consecutive same-sign outliers (structural break). **Bold** = selected.

### EDGAR

| region | linear | linear_w_autocorrelation | broken_trend | hp | hamilton | spline | lowess |
|---|---|---|---|---|---|---|---|
| WLD | ✗ AC | **✓ 2010, 2020** | ✗ AC | ✗ AC | ✓ 2010, 2020 | ✗ AC | ✗ AC |
| EU27 | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| ARG | ✓ | **✓** | ✗ AC | ✗ skew | ✓ | ✗ AC | ✓ 2002, 2020 |
| AUS | ✗ AC | ✓ | **✓** | ✗ AC | ✓ | ✗ AC | ✗ AC |
| BRA | ✓ | ✓ | **✓** | ✗ AC | ✗ ARCH | ✗ AC | ✗ ARCH |
| CAN | ✓ 2020 | ✓ 2020 | **✓** | ✓ | ✓ 2020 | ✓ | ✓ |
| CHN | ✗ AC | ✗ ARCH | **✓** | ✓ | ✓ | ✗ AC | ✗ AC |
| DEU | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| FRA | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| GBR | ✓ | ✓ | **✓ 1979** | ✓ | ✓ | ✓ 1979 | ✓ 1979 |
| IDN | ✗ AC | ✗ ARCH | **✓ 2020** | ✗ ARCH | ✓ | ✗ ARCH | ✗ ARCH |
| IND | ✗ AC | ✗ break 2019–2020 | ✗ AC | ✗ AC | ✗ break 2019–2020 | ✗ ARCH | ✗ AC |
| ITA | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| JPN | ✓ | ✓ | **✓** | ✓ | ✓ | ✗ AC | ✓ |
| KOR | ✗ AC | **✓ 1998, 2010** | ✗ break 2010–2011 | ✓ 1998 | ✓ 1998 | ✗ AC | ✗ ARCH |
| MEX | ✓ 2020 | ✓ 2020 | **✓ 2020** | ✓ 2020 | ✓ 2020 | ✗ AC | ✓ 2020 |
| RUS | ✗ AC | ✓ 1990 | **✓** | ✗ AC | ✓ | ✓ | ✗ AC |
| SAU | ✗ AC | **✓** | ✗ ARCH | ✓ | ✓ 1984 | ✗ AC | ✗ tails |
| TUR | **✓ 2017, 2021** | ✗ tails | ✗ AC | ✗ AC | ✓ | ✓ | ✗ AC |
| USA | ✓ | ✓ | **✓** | ✓ | ✓ | ✗ AC | ✓ 2020 |
| ZAF | ✗ skew | ✓ | **✓** | ✓ | ✗ skew | ✓ | ✓ |

### GCB

| region | linear | linear_w_autocorrelation | broken_trend | hp | hamilton | spline | lowess |
|---|---|---|---|---|---|---|---|
| WLD | ✗ AC | **✓ 2010, 2020** | ✗ AC | ✓ 2020 | ✓ 2010, 2020 | ✗ AC | ✓ 2020 |
| EU27 | ✓ | ✓ | **✓** | ✓ | ✓ | ✗ AC | ✓ |
| ARG | ✓ | ✓ | **✓** | ✗ break 2002–2003 | ✓ | ✗ AC | ✗ break 2002–2003 |
| AUS | ✗ AC | ✓ | **✓** | ✗ AC | ✓ | ✗ AC | ✗ AC |
| BRA | ✗ ARCH | **✓** | ✗ AC | ✗ ARCH | ✗ ARCH | ✓ | ✗ ARCH |
| CAN | ✓ 2020 | ✓ 2020 | **✓ 2020** | ✗ tails | ✓ 2020 | ✗ AC | ✗ tails |
| CHN | ✗ AC | **✓** | ✗ AC | ✗ AC | ✓ | ✓ | ✗ AC |
| DEU | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| FRA | ✗ ARCH | ✗ ARCH | ✗ ARCH | **✓** | ✓ | ✓ | ✓ |
| GBR | ✓ | ✓ | **✓ 1979** | ✓ 1979 | ✓ | ✗ AC | ✓ 1979 |
| IDN | ✗ ARCH | ✗ ARCH | ✗ ARCH | ✗ break 2020–2021 | ✗ ARCH | **✓** | ✗ break 2020–2021 |
| IND | ✗ AC | ✗ break 2019–2020 | ✗ skew | ✗ tails | ✗ break 2019–2020 | ✗ AC | ✗ ARCH |
| ITA | ✓ | ✓ | **✓** | ✓ | ✓ | ✓ | ✓ |
| JPN | ✓ | ✓ | **✓** | ✓ | ✓ | ✗ AC | ✓ |
| KOR | ✗ AC | ✗ tails | ✗ skew | **✓ 1998, 2010, 2020** | ✗ tails | ✗ AC | ✗ AC |
| MEX | ✓ 2020 | ✓ 2020 | **✓ 2020** | ✓ | ✓ | ✗ AC | ✓ 2020 |
| RUS | ✗ AC | ✗ break 1992–1994 | **✓** | ✗ AC | ✓ 1992 | ✗ AC | ✗ AC |
| SAU | ✓ | ✓ | **✓** | ✓ | ✓ | ✗ AC | ✓ |
| TUR | ✗ ARCH | ✗ ARCH | ✗ ARCH | ✗ ARCH | ✓ | **✓** | ✗ ARCH |
| USA | ✓ 2020 | ✓ 2020 | **✓ 2020** | ✓ 2020 | ✓ 2020 | ✗ AC | ✓ 2020 |
| ZAF | ✓ | ✓ | **✓ 2004** | ✓ | ✓ | ✓ | ✓ |

## 3. Pass rates and failure reasons

Share of the 21 regions where each method passes:

| method | EDGAR | GCB |
|---|---|---|
| linear | 57% | 52% |
| linear_w_autocorrelation | 81% | 71% |
| broken_trend | 71% | 62% |
| hp | 62% | 57% |
| hamilton | 86% | 81% |
| spline | 43% | 38% |
| lowess | 52% | 52% |

Failure reasons, count of region × method combinations:

| status | EDGAR | GCB |
|---|---|---|
| fail_arch | 10 | 17 |
| fail_autocorrelation | 34 | 29 |
| fail_normality_heavy_tails | 2 | 5 |
| fail_normality_skewed | 3 | 2 |
| fail_structural_break | 3 | 7 |

## 4. Excluded years and warnings

Outlier years excluded in the selected method:

| year excluded | regions (dataset) |
|---|---|
| 1979 | GBR (EDGAR), GBR (GCB) |
| 1998 | KOR (EDGAR), KOR (GCB) |
| 2004 | ZAF (GCB) |
| 2010 | WLD (EDGAR), KOR (EDGAR), WLD (GCB), KOR (GCB) |
| 2017 | TUR (EDGAR) |
| 2020 | WLD (EDGAR), IDN (EDGAR), MEX (EDGAR), WLD (GCB), CAN (GCB), KOR (GCB), MEX (GCB), USA (GCB) |
| 2021 | TUR (EDGAR) |

Warnings (structural breaks, and outliers flagged in the last 3 historical years that were not excluded):

| dataset | region | warnings |
|---|---|---|
| EDGAR | IND | linear_w_autocorrelation: possible structural break around 2019-2020 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints; hp: [2023] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; hamilton: possible structural break around 2019-2020 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints |
| EDGAR | KOR | broken_trend: possible structural break around 2010-2011 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints |
| EDGAR | ZAF | linear: [2022] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually |
| GCB | ARG | hp: possible structural break around 2002-2003 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints; lowess: possible structural break around 2002-2003 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints |
| GCB | IDN | linear: [2022] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; linear_w_autocorrelation: [2022] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; hp: possible structural break around 2020-2021 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints; lowess: possible structural break around 2020-2021 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints |
| GCB | IND | linear_w_autocorrelation: possible structural break around 2019-2020 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints; [2023] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; broken_trend: [2023] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; hp: [2023] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually; hamilton: possible structural break around 2019-2020 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints; lowess: [2023, 2024] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually |
| GCB | RUS | linear_w_autocorrelation: possible structural break around 1992-1994 (consecutive outliers of the same sign) - consider broken_trend or forced_breakpoints |
| GCB | TUR | lowess: [2022] flagged as outlier(s) but within the last 3 historical years - not excluded, review manually |

## Next steps

1. **Investigate why India fails across all methods in both datasets.** The failure reasons differ by method (autocorrelation, ARCH, skew, and a possible 2019–2020 structural break), and 2023 is flagged as an outlier but protected. Candidates to test: forced breakpoints, a shorter historical period, and the t-distribution once it is added.
2. **Consider whether the exclusion of years needs to be stronger.** 2009 (the global financial crisis) is almost never removed. Outlier detection only runs when ARCH or normality fail, and often removing 2020 alone is enough to pass. Flexible trends (hp, spline, lowess) and broken_trend breakpoints can also bend into 2009 and absorb part of the shock. Keeping a genuine shock in the noise inflates σ, which widens the null distribution and could make the peak test too conservative (false negatives: real peaks not detected). Options include always testing known shock years (e.g. 2009, 2020) as exclusion candidates, or running GESD whether or not the tests fail.

## Settings

- `EmissionsPeakTest.compare_methods(selection="priority")`, alpha = 0.05, up to 3 GESD exclusions, last 3 historical years protected.
- Data: `data/processed/edgar_energy_co2.csv`, `data/processed/gcb_hist_co2.csv`; years 1970–2024.
- Outputs: `Q1_1_method_results.csv` (per method), `Q1_2_selection.csv` (per region).
