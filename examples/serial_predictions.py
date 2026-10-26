"""Rows close in time are linked. A plain shuffle forgets that.

A forecast and the reading it forecasts both drift slowly (each day is close to the day
before). Here they are generated independently, so the forecast has no skill at all.
Over 1,000 days two slow drifts still line up by chance, so their correlation is often
far from 0.

PermutedWeights() shuffles single days. The shuffled forecast no longer drifts, so its
chance correlations are much smaller than the real one, and the real forecast looks
skilled. PermutedWeights(block=100) shuffles 100-day runs, and LagShift(min_lag=100)
slides the whole forecast by 100 days or more. Both keep the drift, so they judge the
chance correlation fairly.
"""

import numpy as np

from placebokit import LagShift, PermutedWeights, run_prediction_controls


def slow_drift(n, rng, phi=0.95):
    """Each value is phi times the one before plus fresh noise (an AR(1) series)."""
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def correlation(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


controls = [PermutedWeights(), PermutedWeights(block=100), LagShift(min_lag=100)]

# One pair of unrelated series.
rng = np.random.default_rng(4)
reading = slow_drift(1000, rng)
forecast = slow_drift(1000, rng)
report = run_prediction_controls(
    reading, forecast, score=correlation, controls=controls, B=199, seed=0
)
print(report)
print()

# How often does each control call a skill-free forecast "skilled" (p <= 0.05)?
trials = 200
false_alarms = {c.label: 0 for c in controls}
rng = np.random.default_rng(5)
for trial in range(trials):
    reading = slow_drift(1000, rng)
    forecast = slow_drift(1000, rng)
    result = run_prediction_controls(
        reading, forecast, score=correlation, controls=controls, B=99, seed=trial
    )
    for r in result:
        false_alarms[r.name] += r.beaten(0.05)

print(f"Skill-free forecasts called skilled at p <= 0.05, out of {trials}:")
for label, count in false_alarms.items():
    print(f"  {label:<26} {count:>3}  ({count / trials:.0%})")
print("A fair test should be near 5%.")
