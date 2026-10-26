"""What does 90% accuracy mean when 90% of the answers are "no"?

Synthetic support tickets: about 1 in 10 is escalated. A simple model predicts
escalation from four ticket features, and its accuracy is 90.2%. That sounds high.
ShuffledLabels shows the real floor: a model trained on shuffled answers learns to say
"no" every time, and that alone is right 89.5% of the time on these test rows. The
model does beat the floor (p = 0.005), but only by 0.7 points.
"""

import numpy as np

from placebokit import ShuffledLabels, run_controls

rng = np.random.default_rng(3)
n = 5000
X = rng.normal(size=(n, 4))  # e.g. reply delay, customer tier, ticket length, reopen count
risk = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.normal(size=n)
y = (risk > np.quantile(risk, 0.9)).astype(int)  # the top 10% get escalated

rows = rng.permutation(n)  # tickets are independent, so a random split is fine here
train, test = rows[:4000], rows[4000:]


def fit(X_train, y_train):
    """Least squares on the 0/1 labels, with an intercept; predict 1 above 0.5."""
    design = np.column_stack([np.ones(len(X_train)), X_train])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)

    def predict(X_new):
        design_new = np.column_stack([np.ones(len(X_new)), X_new])
        return (design_new @ weights >= 0.5).astype(int)

    return predict


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


report = run_controls(
    X, y, fit=fit, score=accuracy, train_idx=train, test_idx=test,
    controls=[ShuffledLabels()], B=199, seed=0,
)
print(report)
print()
print(f"share of 'no' answers in the test rows: {1 - y[test].mean():.3f}")
placebo = report["ShuffledLabels"]
print(f"lift over the floor: {report.real_score - placebo.mean:+.3f}")
