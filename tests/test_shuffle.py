import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from placebokit import block_permutation, random_signs

from simdata import ar1


@given(
    n=st.integers(2, 300),
    block=st.one_of(st.none(), st.integers(1, 299)),
    seed=st.integers(0, 10**6),
)
def test_block_permutation_is_a_permutation(n, block, seed):
    if block is not None and block >= n:
        return
    order = block_permutation(n, block, np.random.default_rng(seed))
    assert sorted(order.tolist()) == list(range(n))


@given(n=st.integers(4, 300), block=st.integers(2, 150), seed=st.integers(0, 10**6))
def test_blocks_stay_whole_and_in_order(n, block, seed):
    if block >= n:
        return
    order = block_permutation(n, block, np.random.default_rng(seed))
    # Every original block shows up as one consecutive run, in its original order.
    for start in range(0, n, block):
        expected = list(range(start, min(start + block, n)))
        where = int(np.flatnonzero(order == start)[0])
        assert order[where : where + len(expected)].tolist() == expected


def test_last_block_may_be_shorter():
    order = block_permutation(10, 4, np.random.default_rng(0))  # blocks 0-3, 4-7, 8-9
    assert sorted(order.tolist()) == list(range(10))
    where = int(np.flatnonzero(order == 8)[0])
    assert order[where : where + 2].tolist() == [8, 9]


def test_block_must_leave_at_least_two_blocks():
    with pytest.raises(ValueError, match="only one block"):
        block_permutation(10, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least 1"):
        block_permutation(10, 0, np.random.default_rng(0))
    with pytest.raises(TypeError):
        block_permutation(10, 2.5, np.random.default_rng(0))


def test_same_seed_same_order():
    a = block_permutation(100, 7, np.random.default_rng(3))
    b = block_permutation(100, 7, np.random.default_rng(3))
    c = block_permutation(100, 7, np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_block_shuffle_keeps_short_range_pattern_and_plain_shuffle_does_not():
    rng = np.random.default_rng(0)
    y = ar1(2000, 0.9, rng)

    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    plain = y[block_permutation(len(y), None, rng)]
    blocked = y[block_permutation(len(y), 100, rng)]
    assert lag1(y) > 0.85
    assert abs(lag1(plain)) < 0.1
    assert lag1(blocked) > 0.8


def test_random_signs_are_plus_or_minus_one():
    signs = random_signs(1000, None, np.random.default_rng(0))
    assert set(np.unique(signs).tolist()) == {-1, 1}
    assert abs(signs.mean()) < 0.1


def test_random_signs_are_constant_within_blocks():
    signs = random_signs(23, 5, np.random.default_rng(1))
    for start in range(0, 23, 5):
        assert len(set(signs[start : start + 5].tolist())) == 1


@given(n=st.integers(2, 300), block=st.integers(1, 150), seed=st.integers(0, 10**6))
def test_random_signs_share_one_sign_per_block(n, block, seed):
    if block >= n:
        return
    signs = random_signs(n, block, np.random.default_rng(seed))
    assert signs.shape == (n,)
    assert set(np.unique(signs).tolist()) <= {-1, 1}
    for start in range(0, n, block):
        assert len(set(signs[start : start + block].tolist())) == 1


def test_every_block_order_is_possible():
    # 3 blocks have 3! = 6 orders. A fair block shuffle should produce all of them,
    # each about 1/6 of the time.
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(6000):
        order = tuple(block_permutation(9, 3, rng)[::3].tolist())  # first row of each block
        counts[order] = counts.get(order, 0) + 1
    assert len(counts) == 6
    for count in counts.values():
        assert abs(count / 6000 - 1 / 6) < 0.02
