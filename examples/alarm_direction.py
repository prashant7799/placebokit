"""A lift that is not what it looks like: the model knows WHEN, not WHICH WAY.

A synthetic sensor has calm runs and noisy runs. In a noisy hour a big move is likely,
up or down. An alarm model raises +1 ("big rise coming") or -1 ("big fall coming") and
is scored by the share of alarms that called the right side.

Model A raises alarms in noisy hours and picks the side from a "hint" that is pure
noise. Its hit rate is still almost twice what random alarms get, so it looks skilled.
SignFlip gives the same alarms random sides. If that scores just as well, the side
calls carry no skill. RandomEntries keeps the alarms but moves them to random hours.
If that scores worse, the timing is real.

Model B has a hint with real (weak) direction information. It beats both placebos.
"""

import numpy as np

from placebokit import RandomEntries, SignFlip, run_prediction_controls

rng = np.random.default_rng(7)
n = 6000  # hours

# Calm (0.5) and noisy (2.0) runs of 48 hours. The level is known one hour ahead.
noise_level = np.repeat(rng.choice([0.5, 2.0], size=n // 48), 48)
change = noise_level * rng.normal(size=n)  # next-hour change of the reading
outcome = np.where(change > 1.5, 1, np.where(change < -1.5, -1, 0))  # big rise / fall / none

noisy_hour = noise_level > 1.0
useless_hint = rng.normal(size=n)  # has nothing to do with the direction
useful_hint = np.sign(change) * (rng.random(n) < 0.6) + rng.normal(scale=0.8, size=n)

model_a = np.where(noisy_hour, np.sign(useless_hint), 0).astype(int)
model_b = np.where(noisy_hour, np.sign(useful_hint), 0).astype(int)


def hit_rate(y_true, alarms):
    """Share of alarms that named the right side."""
    raised = alarms != 0
    return float(np.mean(y_true[raised] == alarms[raised]))


controls = [SignFlip(), RandomEntries()]
models = [("Model A (useless hint)", model_a), ("Model B (useful hint)", model_b)]
for i, (name, alarms) in enumerate(models):
    report = run_prediction_controls(
        outcome, alarms, score=hit_rate, controls=controls, B=199, seed=1
    )
    if i > 0:
        print()
    print(f"== {name}: {int((alarms != 0).sum())} alarms")
    print(report)
    print(f"every side flipped: {report['SignFlip'].details['full_flip_score']:.3f}")
