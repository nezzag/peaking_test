# Understanding Mahalanobis Distance for Trajectory Testing

**A Complete Explanation from First Principles**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Building Up from Simple Cases](#building-up-from-simple-cases)
3. [The General Formula](#the-general-formula)
4. [AR(1) Autocorrelation in Your Case](#ar1-autocorrelation-in-your-case)
5. [Comparing Two Scenarios](#comparing-two-scenarios)
6. [The Inverse Covariance Matrix](#the-inverse-covariance-matrix)
7. [The Complete Test Procedure](#the-complete-test-procedure)
8. [Visualizing the Difference](#visualizing-the-difference)
9. [Key Takeaways](#key-takeaways)

---

## Introduction

### The Problem We're Solving

Let's imagine you have 3 years of emissions data: `[37500, 37500, 35700]`

This represents deviations from your null hypothesis (flat emissions) of: `[0, 0, -1800]`

**Question:** *"How unusual is this pattern if the null hypothesis is true?"*

But there's a complication: **The noise might not be independent year-to-year.** It might well be **AUTOCORRELATED** - if emissions are high one year, they tend to be high the next year too (φ ≈ 0.5).

So we can't just say "each year independently has 5% chance of being this low, so probability is 0.05³ = 0.000125". That would be **WRONG** because it ignores the correlation!

**Mahalanobis distance is the correct way to measure "how unusual" when you have correlated data.**

---

## Building Up from Simple Cases

### Case 1: One Variable, No Correlation (Standard Z-Score)

Imagine you have just **ONE** year of data:
- Observed deviation: -1800 Mt
- Historical noise std dev: σ = 200 Mt

**How unusual is -1800?**

Answer: Use a **z-score**!

```
z = (observed - mean) / std dev
z = (-1800 - 0) / 200
z = -9.0
```

**Interpretation:** This is 9 standard deviations below the mean. Very unusual! (p < 0.0001)

**The z-score measures "distance from the mean in units of standard deviation"**

---

### Case 2: Two Variables, No Correlation

Now imagine **TWO** years: `[year1_deviation, year2_deviation]`
- Observed: `[-900, -900]`
- Noise std dev: σ = 200 for both years
- **No correlation** between years

**How unusual is this?**

We can't just average the z-scores! We need to combine them properly.

Answer: Use **Euclidean distance**!

```
d² = (-900)²/200² + (-900)²/200²
d² = 20.25 + 20.25 = 40.5
d = 6.36
```

This is the **MAHALANOBIS DISTANCE** for uncorrelated data! It's just the Euclidean distance after standardizing by σ.

**Formula:** `d = sqrt(x₁²/σ² + x₂²/σ²)`

---

### Case 3: Two Variables, WITH Correlation (The Key Insight!)

Now imagine two years of data, but **with autocorrelation** present
- Observed: `[-900, -900]`

But now: **if year 1 is -900, year 2 is MORE LIKELY to also be negative** because of autocorrelation (φ = 0.5)!

So `[-900, -900]` is **LESS unusual** than it would be without correlation.

We need to account for this. **That's what Mahalanobis distance does!**

The key: Instead of assuming independence, we use a **COVARIANCE MATRIX:**

```
    Σ = [σ²      ρσ²  ]
        [ρσ²     σ²   ]
```

where ρ (rho) is the correlation coefficient.

For AR(1) with φ = 0.5: ρ = 0.5

So:
```
    Σ = [200²         0.5×200²]  = [40000  20000]
        [0.5×200²     200²    ]    [20000  40000]
```

Mahalanobis distance:
```
    d² = x' Σ⁻¹ x
```

where x = `[-900, -900]` and Σ⁻¹ is the **INVERSE** of the covariance matrix.

#### Example Calculation

```python
# Covariance matrix
Σ = [[40000, 20000],
     [20000, 40000]]

# Inverse covariance
Σ⁻¹ = [[3.33e-05, -1.67e-05],
       [-1.67e-05,  3.33e-05]]

# Observed deviations
x = [-900, -900]

# Mahalanobis distance
d² = x' Σ⁻¹ x
d = sqrt(d²) = 5.20
```

**Compare to uncorrelated case:**
- Euclidean distance (no correlation): 6.36
- Mahalanobis distance (with correlation): 5.20

**The Mahalanobis distance is SMALLER because the pattern `[-900, -900]` is more plausible when values are correlated!**

If year 1 is -900, then year 2 being -900 is not surprising given the autocorrelation. They "explain each other" to some extent.

---

## The General Formula

### Mahalanobis Distance - General Formula

For a vector **x** = [x₁, x₂, ..., xₙ] with covariance matrix **Σ**:

```
d² = x' Σ⁻¹ x
```

In matrix form:
```
d² = [x₁ x₂ ... xₙ] × Σ⁻¹ × [x₁]
                              [x₂]
                              [⋮]
                              [xₙ]
```

Breaking it down:
1. **Σ** captures how variables are related (correlation structure)
2. **Σ⁻¹** "undoes" the correlation
3. **x' Σ⁻¹ x** gives the squared distance accounting for correlation
4. **d = sqrt(d²)** is the final Mahalanobis distance

### Special Cases

**1. If Σ is diagonal (no correlation):**

```
Σ = diag(σ₁², σ₂², ..., σₙ²)
```

Then: `d² = x₁²/σ₁² + x₂²/σ₂² + ... + xₙ²/σₙ²`

This is just **standardized Euclidean distance**! This means we can still use the Mahalanobis test in cases where there is no autocorrelation at all.

**2. If Σ = σ²I (same variance, no correlation):**

Then: `d = Euclidean distance / σ`

This is the **multivariate z-score**!

### Why the Inverse?

The inverse **Σ⁻¹** "decorrelates" the data.

Think of it like this:
- If x₁ and x₂ are highly correlated, seeing both be negative doesn't give you TWICE as much information
- Σ⁻¹ adjusts the weight so you don't "double count" correlated info
- It's like asking: **"what's the truly independent information content?"**

---

## What about inclusion of AR(1) in your noise?

Let's imagine that you have an AR(1) noise model:

```
ε[t] = φ·ε[t-1] + η[t]
```

where:
- **φ = 0.5** (autocorrelation coefficient)
- **η[t] ~ N(0, σ²)** (white noise innovation)
- **σ = 100 Mt** (innovation std dev)

For AR(1), the covariance between years i and j is:

```
Cov(ε[i], ε[j]) = (σ²/(1-φ²)) × φ^|i-j|
```

For φ = 0.5, σ = 100:

```
Variance = σ²/(1-φ²) = 100²/(1-0.25) = 13,333

Cov(year 1, year 1) = 13,333 × φ⁰ = 13,333
Cov(year 1, year 2) = 13,333 × φ¹ = 6,667
Cov(year 1, year 3) = 13,333 × φ² = 3,333
Cov(year 2, year 3) = 13,333 × φ¹ = 6,667
```

So the full **covariance matrix** is:

```
    Σ = [13,333  6,667   3,333]
        [6,667   13,333  6,667]
        [3,333   6,667   13,333]
```

This captures:
- Years 1-2 are highly correlated (ρ = 0.5)
- Years 1-3 are less correlated (ρ = 0.25)
- Years 2-3 are highly correlated (ρ = 0.5)

---

## Comparing Two Scenarios

### Your Two Scenarios

**Scenario A: Sudden drop** `[0, 0, -1500]`  
**Scenario B: Gradual decline** `[0, -750, -1500]`

Both reach -1500 total, but **patterns differ**!

### Mahalanobis Distance Calculation

Using the AR(1) covariance matrix from above:

```
Scenario A: [0, 0, -1500]
  Mahalanobis distance: 15.00

Scenario B: [0, -750, -1500]
  Mahalanobis distance: 13.52

Ratio (A/B): 1.11
→ Scenario A is MORE unusual!
→ Sudden drop is 1.11x more extreme
```

### Why is the Sudden Drop More Unusual?

With AR(1) correlation, **the noise tends to PERSIST**.

**Scenario B** `[-750, -1500]` follows this pattern:
- Year 1: 0 (near baseline)
- Year 2: -750 (starting to deviate)  
- Year 3: -1500 (continuing in same direction, building on year 2)

This is **CONSISTENT** with AR(1) noise: a negative shock in year 2 naturally leads to continued negative values in year 3 (because φ > 0).

**Scenario A** `[0, 0, -1500]` VIOLATES this pattern:
- Year 1: 0
- Year 2: 0 (still at baseline)
- Year 3: -1500 (sudden massive jump!)

This **BREAKS** the autocorrelation structure. If year 2 is 0, then with AR(1) we expect year 3 to be near 0 too (maybe ±100 or so).

A jump from 0 to -1500 requires a **HUGE innovation** η[3] ≈ -1500, which is **15 standard deviations**! Very unlikely.

**The Mahalanobis distance captures this automatically via Σ⁻¹.**

---

## The Inverse Covariance Matrix

### Understanding Σ⁻¹ (The Precision Matrix)

The inverse covariance matrix **Σ⁻¹** is also called the **PRECISION MATRIX**.

It tells us: **"what combinations of variables are unusual?"**

For our 3-year case:

```python
# Covariance matrix Σ
[[13,333  6,666  3,333]
 [6,666  13,333  6,666]
 [3,333  6,666  13,333]]

# Precision matrix Σ⁻¹
[[1.00e-04  -5.00e-05   0.00e+00]
 [-5.00e-05  1.25e-04  -5.00e-05]
 [0.00e+00  -5.00e-05   1.00e-04]]
```

**Key insight:** The off-diagonal **negative values** in Σ⁻¹ encode the persistence! They penalize sudden changes between adjacent years.

### How d² = x' Σ⁻¹ x Works

When we calculate `d² = x' Σ⁻¹ x`, we're computing:

```
d² = Σᵢ Σⱼ Σ⁻¹[i,j] × x[i] × x[j]
```

**For scenario A:** `[0, 0, -1500]`

The term `Σ⁻¹[2,3] × x[2] × x[3] = Σ⁻¹[2,3] × 0 × (-1500) = 0`

But the term `Σ⁻¹[3,3] × x[3]²` is huge because `x[3] = -1500`!

**For scenario B:** `[0, -750, -1500]`

Now `x[2]` and `x[3]` are both negative, and `Σ⁻¹[2,3]` is negative, so that term:

```
Σ⁻¹[2,3] × x[2] × x[3] = (negative) × (neg) × (neg) = negative
```

This **REDUCES** the total d² because the pattern is consistent with correlation!

---

## The Complete Test Procedure

### How the Mahalanobis Peak Test Works

#### Step 1: FIT THE NOISE MODEL
- Use historical data (1970-2024)
- Decompose into trend + residuals
- Fit AR(1) model: `ε[t] = φ·ε[t-1] + η[t]`
- Estimate φ and σ

#### Step 2: BUILD THE COVARIANCE MATRIX
- Use AR(1) formula: `Σ[i,j] = (σ²/(1-φ²)) × φ^|i-j|`
- This is your "expected noise structure"

#### Step 3: SET UP NULL HYPOTHESIS
- Usually "zero trend" (flat emissions)
- Null baseline = `[37,500, 37,500, 37,500]`

#### Step 4: CALCULATE OBSERVED MAHALANOBIS DISTANCE
- Deviations: `x = observed - null baseline`
- Distance: `d_obs = sqrt(x' Σ⁻¹ x)`

#### Step 5: GENERATE BOOTSTRAP SAMPLES
For i = 1 to 10,000:
1. Generate AR(1) noise: `ε[t] = φ·ε[t-1] + η[t]`
2. Create trajectory: `null_baseline + noise`
3. Calculate Mahalanobis: `d[i] = sqrt(ε' Σ⁻¹ ε)`

#### Step 6: CALCULATE P-VALUE
- Count how many bootstrap samples have `d[i] ≤ d_obs`
- `p-value = (# with d[i] ≤ d_obs) / 10,000`

#### Step 7: INTERPRET
- **Small p-value (< 0.05):** Pattern is unusual → evidence of peak
- **Large p-value (> 0.10):** Pattern consistent with noise → insufficient evidence

---

## Visualizing the Difference

### With vs Without Autocorrelation

![Mahalanobis Visualization](mahalanobis_visualization.png)

**Left Panel: White Noise (φ=0)**
- No correlation between years
- Point cloud is circular
- Both scenarios equally distant from origin

**Right Panel: AR(1) Noise (φ=0.5)**
- Positive correlation between years
- Point cloud is **elongated** along diagonal
- Gradual decline `[-750, -1500]` sits **within** the cloud
- Sudden drop `[0, -1500]` is **outside** the expected pattern

The sudden drop is inconsistent with how AR(1) noise behaves!

---

## Key Takeaways

### 1. What is Mahalanobis Distance?
- A way to measure **"how unusual"** a pattern that accounts for when variables are correlated
- Generalizes z-score and Euclidean distance to correlated data
- Formula: `d = sqrt(x' Σ⁻¹ x)`

### 2. Why Do We Need It?
- If emissions noise is autocorrelated (φ ≈ 0.5)
- Can't treat each year independently
- Need to account for correlation structure

### 3. How Does It Work?
- Covariance matrix **Σ** encodes correlation structure
- Inverse **Σ⁻¹** "decorrelates" the data
- Distance accounts for which patterns are plausible given correlation

### 4. What Does It Tell Us?
- Sudden drops are **MORE unusual** (larger Mahalanobis) with AR(1)
- Gradual declines are **LESS unusual** (smaller Mahalanobis)
- A sudden drop is inconsistent with noise persistence → stronger evidence

### 5. For Peak Detection:
- Small p-value means: *"This pattern is so unusual given AR(1) noise that it's unlikely to be random"*
- Sudden drops get **SMALLER p-values** (more significant)
- Gradual declines get **LARGER p-values** (less significant from noise perspective)

---

## Comparison to Slope Test

### Slope Test
**Asks:** *"Is there a significant linear trend?"*

- ✅ **Pros:** Intuitive, easy to explain
- ❌ **Cons:** Treats all patterns with same slope equally

### Mahalanobis Test
**Asks:** *"Is this pattern consistent with autocorrelated noise?"*

- ✅ **Pros:** Accounts for temporal structure, distinguishes patterns
- ❌ **Cons:** More complex, harder to explain

### Potentially you can use both to give better understanding of the data

Report slope (with p-value) for intuitive interpretation  
**+**  
Report Mahalanobis (with p-value) for pattern analysis

**Example:**
> "Slope = -750 Mt/yr (p=0.03), but pattern analysis shows sudden drop inconsistent with noise structure (Mahalanobis p=0.001)"

---

## Summary

**Mahalanobis distance expands your analysis to be able to test trajectory patterns, rather than just the average slope across time. It also allows for autocorrelation to be accounted for in the test.**

It answers: *"How unusual is this ENTIRE pattern given what we know about how noise behaves?"*

### For Your Emissions Data:

1. Fit AR(1) model to historical data
2. Build covariance matrix from AR(1) parameters  
3. Calculate Mahalanobis distance for observed trajectory
4. Compare to bootstrap distribution
5. Get p-value

**With AR(1) autocorrelation, sudden drops are MORE extreme than gradual declines because they violate the expected persistence in the noise.**

### Implementation

Use the `MahalanobisTrajectoryTest` class:

```python
from mahalanobis_trajectory_test import MahalanobisTrajectoryTest

# Setup your peak test first
peak_test = EmissionsPeakTest()
peak_test.load_historical_data("gcb_hist_co2.csv")
peak_test.set_test_data([(2025, 37500), (2026, 37500), (2027, 35700)])
peak_test.characterize_noise(method="spline", noise_type="normal", s=25)
peak_test.create_noise_generator()

# Run Mahalanobis test
mahal_test = MahalanobisTrajectoryTest(peak_test)
results = mahal_test.run_test(
    n_bootstrap=10000,
    null_hypothesis="zero_trend"
)

# Visualize
mahal_test.plot_results()
```

---