import numpy as np

from placebokit import Ablate, PermutedWeights, ShuffledLabels, SignFlip, run_controls

# 2,000 hours of synthetic machine data: 3 vibration and 3 temperature sensors.
rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 6))
# The next-hour load change depends on one vibration and one temperature sensor.
y = 0.6 * X[:, 0] + 0.3 * X[:, 3] + rng.normal(size=2000)
train, test = np.arange(1500), np.arange(1500, 2000)  # older hours train, newer hours test


def fit(X_train, y_train):
    weights, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)  # least squares
    return lambda X_new: X_new @ weights  # fit returns a predict function


def score(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)  # R squared: higher is better


report = run_controls(
    X, y, fit=fit, score=score, train_idx=train, test_idx=test,
    controls=[
        ShuffledLabels(block=24),              # retrain on shuffled answers, in 24-hour runs
        Ablate([0, 1, 2], group="vibration"),  # retrain with the vibration sensors shuffled
        Ablate([4, 5], group="unused"),        # two sensors the target never used
        PermutedWeights(),                     # the real predictions, on the wrong hours
        SignFlip(),                            # the real predictions, random directions
    ],
    B=99, seed=0,
)
print(report)
