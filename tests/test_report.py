import numpy as np
import pytest

from placebokit import PermutedWeights, SignFlip, run_prediction_controls
from placebokit.report import ControlReport, ControlResult


@pytest.fixture
def report():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    pred = 0.5 * y + rng.normal(size=300)
    return run_prediction_controls(
        y, pred, score=lambda a, b: float(np.corrcoef(a, b)[0, 1]),
        controls=[PermutedWeights(), SignFlip()], B=49, seed=3,
    )


def test_lookup_by_name_iteration_and_length(report):
    assert report.names == ["PermutedWeights", "SignFlip"]
    assert report["SignFlip"].runs == 49
    assert [r.name for r in report] == report.names
    assert len(report) == 2
    with pytest.raises(KeyError, match="no control named"):
        report["ShuffledLabels"]


def test_text_report_shows_every_control_and_the_formula(report):
    text = str(report)
    assert "placebokit report (B=49, seed=3, higher is better)" in text
    assert f"real score: {report.real_score:.3f}" in text
    for name in report.names:
        assert name in text
    assert "0.020  beaten" in text  # p = 1/50
    assert "(k + 1) / (runs + 1)" in text
    assert "real score: 0.5\n" in report.to_text(digits=1)


def test_frame_has_one_row_per_control(report):
    frame = report.to_frame()
    assert list(frame.columns) == ["control", "runs", "mean", "std", "best", "p_value", "beaten"]
    assert frame["beaten"].all()
    assert frame.loc[0, "best"] == pytest.approx(report["PermutedWeights"].scores.max())


def test_passed_depends_on_alpha(report):
    assert report.passed(0.05)
    assert not report.passed(0.01)


def test_result_summaries_skip_nan_runs():
    result = ControlResult(
        name="x", real_score=1.0, scores=np.array([0.0, np.nan, 2.0]), p_value=0.75
    )
    assert result.n_failed == 1
    assert result.mean == pytest.approx(1.0)
    assert result.best == 2.0
    assert result.min_p_value == pytest.approx(0.25)
    lower = ControlResult(
        name="x", real_score=1.0, scores=np.array([0.0, 2.0]), p_value=0.5,
        higher_is_better=False,
    )
    assert lower.best == 0.0
    empty = ControlResult(name="x", real_score=1.0, scores=np.array([np.nan]), p_value=1.0)
    assert np.isnan(empty.mean)


def test_nan_runs_are_flagged_in_the_text():
    result = ControlResult(name="x", real_score=1.0, scores=np.array([np.nan, 0.0]), p_value=2 / 3)
    text = str(ControlReport(real_score=1.0, results=[result], B=2, seed=0))
    assert "NOT beaten (1 NaN runs)" in text


@pytest.mark.parametrize(
    "B, digits, shown",
    [(2000, 3, "0.0005"), (9999, 3, "0.0001"), (9999, 1, "0.0001"), (2000, 6, "0.000500")],
)
def test_the_smallest_p_value_never_prints_as_zero(B, digits, shown):
    # 1 / (B + 1) at 3 decimals is 0.000 once B >= 2000, which reads as a p-value of 0.
    result = ControlResult(name="x", real_score=1.0, scores=np.zeros(B), p_value=1 / (B + 1))
    text = ControlReport(real_score=1.0, results=[result], B=B, seed=0).to_text(digits=digits)
    p_cell = text.splitlines()[4].split()[4]
    assert p_cell == shown
    assert float(p_cell) > 0


def test_the_p_value_column_uses_one_number_of_decimals():
    many = ControlResult(name="many", real_score=1.0, scores=np.zeros(9999), p_value=1e-4)
    few = ControlResult(name="few", real_score=1.0, scores=np.zeros(11), p_value=1 / 12)
    text = str(ControlReport(real_score=1.0, results=[many, few], B=9999, seed=0))
    assert "0.0001  beaten" in text
    assert "0.0833  NOT beaten" in text


def test_numbers_that_round_to_zero_print_without_a_minus_sign():
    result = ControlResult(
        name="x", real_score=-0.0004, scores=np.array([-0.0001, -0.0002]), p_value=1.0
    )
    text = str(ControlReport(real_score=-0.0004, results=[result], B=2, seed=0))
    assert "-0.000" not in text
    assert "real score: 0.000\n" in text
    assert text.splitlines()[4].split()[2:4] == ["0.000", "0.000"]
    negative = ControlResult(name="x", real_score=1.0, scores=np.array([-0.5]), p_value=1.0)
    assert "-0.500" in str(ControlReport(real_score=1.0, results=[negative], B=1, seed=0))
