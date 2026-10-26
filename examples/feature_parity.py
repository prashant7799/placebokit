"""Train/serve parity: will the live features match the ones the model was trained on?

Synthetic hourly temperature readings. Six feature functions, each a common way to
build features (and a common way to get them wrong), are checked three ways:

- check_repeatable: two runs give the same table, and the input is left alone.
- check_causal: changing future rows must not move any past feature value.
- check_warmup: features from only the last few rows must match the full history.
  This finds how many rows of history a live server must keep.
"""

import numpy as np
import pandas as pd

from placebokit import check_causal, check_repeatable, check_warmup

rng = np.random.default_rng(0)
hours = pd.date_range("2026-01-01", periods=2000, freq="h")
raw = pd.DataFrame({"temp": 20 + np.cumsum(rng.normal(scale=0.1, size=2000))}, index=hours)


def rolling_features(df):
    """Fine: a 24-hour mean and the change since the last hour."""
    out = pd.DataFrame(index=df.index)
    out["mean_24h"] = df["temp"].rolling(24).mean()
    out["change_1h"] = df["temp"].diff()
    return out.dropna()


def centred_mean(df):
    """Leaks: a centred window also averages the next 12 hours."""
    return df["temp"].rolling(24, center=True).mean().dropna().rename("mean_24h")


def backfilled_mean(df):
    """Leaks, but only at the start: the empty warm-up rows are filled from later rows."""
    return df["temp"].rolling(24).mean().bfill().rename("mean_24h")


def whole_sample_zscore(df):
    """Leaks everywhere: the mean and spread come from the whole table, future included."""
    return ((df["temp"] - df["temp"].mean()) / df["temp"].std()).rename("temp_z")


def smoothed(df):
    """Causal, but every value depends a little on all of the history."""
    return df["temp"].ewm(span=24).mean().rename("smooth")


def adds_column(df):
    """Changes its input: the caller's table gains a column."""
    df["mean_24h"] = df["temp"].rolling(24).mean()
    return df[["mean_24h"]].dropna()


windows = list(range(1, 301))
print(f"{'feature function':<20} {'repeatable':<11} {'causal':<8} warm-up needed")
for fn in [rolling_features, centred_mean, backfilled_mean, whole_sample_zscore, smoothed,
           adds_column]:
    repeatable = check_repeatable(fn, raw)
    causal = check_causal(fn, raw)
    warmup = check_warmup(fn, raw, windows)
    needed = f"{warmup.min_window} rows" if warmup.ok else "never matches"
    print(
        f"{fn.__name__:<20} {'ok' if repeatable.ok else 'FAILED':<11} "
        f"{'ok' if causal.ok else 'FAILED':<8} {needed}"
    )

print()
print("Details for backfilled_mean:")
print(check_causal(backfilled_mean, raw))
