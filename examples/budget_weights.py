"""Is the gain from WHERE the budget goes, or just from HOW MUCH is spent?

Synthetic coupon test: 2,000 users, and a past randomised test measured each user's
extra spend per unit of coupon budget. Coupons help most users a little, so any budget
earns something. Two allocation models give each user a share of the budget:

- Model A targets: it gives more to users it expects to respond (mean weight 1.0).
- Model B spends twice as much, spread at random (mean weight 2.0).

Model B earns more in total. The placebos ask the fair question: does each model beat
random allocations of the SAME budget? RandomWeights(nonnegative=True) draws random
non-negative weights with the same mean and spread. PermutedWeights hands the model's
own weights to random users.
"""

import numpy as np

from placebokit import PermutedWeights, RandomWeights, run_prediction_controls

rng = np.random.default_rng(11)
n = 2000
visits = rng.normal(size=n)  # visits last month (standardised)
gain = 0.5 + 0.8 * visits + rng.normal(size=n)  # measured extra spend per budget unit

expected = 0.5 + 0.8 * (visits + rng.normal(scale=1.5, size=n))  # model A's noisy view
model_a = np.clip(expected, 0, None)
model_a = model_a / model_a.mean()  # mean weight 1.0
model_b = rng.uniform(1.0, 3.0, size=n)  # mean weight about 2.0, no targeting


def total_gain(y_true, weights):
    return float(np.sum(weights * y_true))


controls = [RandomWeights(nonnegative=True), PermutedWeights()]
models = [("Model A (targeted)", model_a), ("Model B (spends more)", model_b)]
for i, (name, weights) in enumerate(models):
    report = run_prediction_controls(
        gain, weights, score=total_gain, controls=controls, B=199, seed=2
    )
    if i > 0:
        print()
    print(f"== {name}: mean weight {weights.mean():.2f}")
    print(report.to_text(digits=0))
