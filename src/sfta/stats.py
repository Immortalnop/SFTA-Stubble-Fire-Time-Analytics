"""Shared statistics: moving-block bootstrap and trend tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap resample of range(n), length n."""
    if n == 0:
        return np.array([], dtype=int)
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def boot_ci(stat, arrays: list[np.ndarray], n_boot: int, block: int, seed: int, alpha: float = 0.05):
    """Point estimate and percentile CI of stat(*arrays) under a moving-block bootstrap over rows."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = stat(*arrays)
    if n < 2:
        return est, np.nan, np.nan
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_indices(n, block, rng)
        draws[i] = stat(*(a[idx] for a in arrays))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return est, np.nan, np.nan
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, lo, hi


def ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = np.sum(den)
    return np.sum(num) / d if d > 0 else np.nan


def trend(years, values) -> dict:
    """Theil-Sen slope (per year) with 95% CI, and Mann-Kendall p-value (Kendall tau vs time)."""
    x = np.asarray(years, float)
    y = np.asarray(values, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 4:
        return {"n_years": len(y), "slope": np.nan, "slope_lo": np.nan, "slope_hi": np.nan, "tau": np.nan, "mk_p": np.nan}
    slope, _, lo, hi = stats.theilslopes(y, x)
    tau, p = stats.kendalltau(x, y)
    return {"n_years": len(y), "slope": slope, "slope_lo": lo, "slope_hi": hi, "tau": tau, "mk_p": p}
