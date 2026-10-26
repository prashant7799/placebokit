"""The empirical p-value: golden values and the exact guarantees from Phipson and Smyth (2010)."""

from math import comb

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from placebokit import empirical_p_value

# --- golden values -------------------------------------------------------------------


def test_real_score_above_every_placebo_gives_the_smallest_p():
    # B = 99, none at least as good: p = (0 + 1) / (99 + 1) = 0.01, never 0.
    assert empirical_p_value(1.0, np.zeros(99)) == pytest.approx(0.01)


def test_real_score_below_every_placebo_gives_one():
    assert empirical_p_value(0.0, np.ones(99)) == 1.0


def test_hand_counted_example():
    # Placebos 0.1..0.9, real 0.75: k = 2 (0.8 and 0.9), B = 9, p = 3 / 10.
    placebo = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert empirical_p_value(0.75, placebo) == pytest.approx(0.3)


def test_ties_count_against_the_model():
    assert empirical_p_value(0.5, [0.5, 0.1, 0.1, 0.1]) == pytest.approx(2 / 5)


def test_lower_is_better_flips_the_comparison():
    placebo = [1.0, 2.0, 3.0, 4.0]
    assert empirical_p_value(0.5, placebo, higher_is_better=False) == pytest.approx(1 / 5)
    assert empirical_p_value(3.0, placebo, higher_is_better=False) == pytest.approx(4 / 5)


def test_nan_placebos_count_against_the_model():
    assert empirical_p_value(1.0, [np.nan, 0.0, 0.0]) == pytest.approx(2 / 4)


def test_nan_real_score_raises():
    with pytest.raises(ValueError, match="NaN"):
        empirical_p_value(np.nan, [0.0])


def test_empty_placebo_raises():
    with pytest.raises(ValueError, match="at least one"):
        empirical_p_value(1.0, [])


# --- golden values from Phipson and Smyth (2010), Table 1 ----------------------------
#
# Table 1 of the paper: a one-sided test between two groups of five, m = 100 random
# permutations, so m_t + 1 = C(10, 5) = 252 possible values of the statistic. For each
# count b of permutations at least as extreme, it lists the upper bound p_u = (b+1)/(m+1),
# which is what placebokit reports, and the exact p-value p_e of its equation (1).

TABLE_1_UPPER_BOUND = [0.0099, 0.0198, 0.0297, 0.0396, 0.0495, 0.0594, 0.0693, 0.0792]
TABLE_1_EXACT = [
    0.008047755, 0.017818517, 0.027718516, 0.037619829,
    0.047520825, 0.057421814, 0.067322804, 0.077223794,
]


def binomial_cdf(b, m, p):
    return sum(comb(m, i) * p**i * (1 - p) ** (m - i) for i in range(b + 1))


def exact_p_value(b, m, m_t):
    """Equation (1) of the paper: the exact p-value when permutations are drawn with repeats."""
    total = 0.0
    for b_t in range(m_t + 1):
        total += binomial_cdf(b, m, (b_t + 1) / (m_t + 1))
    return total / (m_t + 1)


@pytest.mark.parametrize("b", range(8))
def test_table_1_upper_bound(b):
    placebo = [1.0] * b + [-1.0] * (100 - b)  # b of the 100 placebos beat a real score of 0
    assert round(empirical_p_value(0.0, placebo), 4) == TABLE_1_UPPER_BOUND[b]


@pytest.mark.parametrize("b", range(8))
def test_table_1_exact_value_is_below_our_upper_bound(b):
    # Reproducing the paper's exact column checks that we read its setting right. The
    # paper shows p_e < p_u, so the value placebokit reports is on the safe side.
    exact = exact_p_value(b, m=100, m_t=251)
    assert exact == pytest.approx(TABLE_1_EXACT[b], abs=5e-10)
    assert exact < empirical_p_value(0.0, [1.0] * b + [-1.0] * (100 - b))


def test_monte_carlo_count_is_uniform_so_p_u_is_exact():
    # Section 2 of the paper: if p is uniform and b ~ Binomial(m, p), then b is uniform
    # on 0..m, so P(b' <= b) = (b + 1) / (m + 1) exactly. Integrate numerically to check.
    m = 19
    grid = (np.arange(200000) + 0.5) / 200000  # midpoint rule on (0, 1)
    for b in range(m + 1):
        mass = np.mean(comb(m, b) * grid**b * (1 - grid) ** (m - b))
        assert mass == pytest.approx(1 / (m + 1), rel=1e-6)


# --- exact guarantees ----------------------------------------------------------------


@pytest.mark.parametrize("B", [9, 19, 99])
def test_every_rank_is_equally_likely_so_p_is_exactly_uniform(B):
    # If the real score is exchangeable with the B placebos, it is equally likely to hold
    # any of the B + 1 ranks. Going through every rank gives each p-value 1/(B+1), ..., 1
    # exactly once, so P(p <= j / (B + 1)) = j / (B + 1) exactly.
    values = np.arange(B + 1, dtype=float)
    p_values = []
    for rank in range(B + 1):
        real = values[rank]
        placebo = np.delete(values, rank)
        p_values.append(empirical_p_value(real, placebo))
    expected = np.arange(1, B + 2) / (B + 1)
    assert np.allclose(np.sort(p_values), expected)
    for j in range(1, B + 2):
        level = j / (B + 1)
        assert np.mean(np.array(p_values) <= level + 1e-12) == pytest.approx(level)


def exchangeable_rejection_rate(trials=20000):
    """Share of trials with p <= 0.05 when the real score is one of 20 exchangeable draws."""
    rng = np.random.default_rng(1)
    all_scores = rng.normal(size=(trials, 20))
    hits = 0
    for scores in all_scores:
        hits += empirical_p_value(scores[0], scores[1:]) <= 0.05
    return hits / trials


def test_simulated_level_matches_alpha_for_exchangeable_scores():
    # With B = 19, p <= 0.05 only when the real score beats all 19 placebos: chance 1/20.
    trials = 20000
    rate = exchangeable_rejection_rate(trials)
    tolerance = 4 * np.sqrt(0.05 * 0.95 / trials)
    assert abs(rate - 0.05) < tolerance


@given(
    real=st.floats(-10, 10),
    placebo=st.lists(st.floats(-10, 10), min_size=1, max_size=60),
)
def test_p_is_between_its_bounds(real, placebo):
    p = empirical_p_value(real, placebo)
    assert 1 / (len(placebo) + 1) <= p <= 1


@given(
    low=st.floats(-10, 10),
    step=st.floats(0, 5),
    placebo=st.lists(st.floats(-10, 10), min_size=1, max_size=60),
)
def test_a_better_real_score_never_gets_a_larger_p(low, step, placebo):
    assert empirical_p_value(low + step, placebo) <= empirical_p_value(low, placebo)


finite = st.floats(-1e6, 1e6, allow_nan=False)
scores_or_nan = st.one_of(finite, st.just(float("nan")))


@given(real=finite, placebo=st.lists(scores_or_nan, min_size=1, max_size=60))
def test_p_matches_a_plain_count(real, placebo):
    # The definition, written as plainly as possible: count the placebo runs that are at
    # least as good, or NaN.
    k = 0
    for score in placebo:
        if np.isnan(score) or score >= real:
            k += 1
    assert empirical_p_value(real, placebo) == (k + 1) / (len(placebo) + 1)


@given(real=finite, placebo=st.lists(finite, min_size=1, max_size=60))
def test_lower_is_better_is_the_mirror_image(real, placebo):
    mirrored = [-x for x in placebo]
    low = empirical_p_value(real, placebo, higher_is_better=False)
    assert low == empirical_p_value(-real, mirrored, higher_is_better=True)


@given(
    real=st.integers(-1000, 1000),
    placebo=st.lists(st.integers(-1000, 1000), min_size=1, max_size=60),
    seed=st.integers(0, 1000),
)
def test_p_depends_only_on_ranks(real, placebo, seed):
    # Only the order matters: shuffling the placebo runs or applying the same increasing
    # function to every score leaves the p-value unchanged. Whole-number scores keep the
    # transform from merging two nearly equal floats into a tie.
    p = empirical_p_value(real, placebo)
    shuffled = np.random.default_rng(seed).permutation(placebo)
    assert empirical_p_value(real, shuffled) == p
    assert empirical_p_value(np.exp(real / 100), np.exp(np.array(placebo) / 100)) == p
