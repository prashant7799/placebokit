"""Random reorderings used by the controls: plain shuffles, block shuffles and sign flips.

Rows that are close in time are often linked. A plain shuffle breaks every link. A block
shuffle cuts the rows into runs of `block` rows, shuffles the order of the runs, and keeps
the order inside each run. Short-range patterns survive inside the runs.
"""

from __future__ import annotations

import numpy as np


def _check_block(n: int, block: int | None) -> None:
    if block is None:
        return
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)):
        raise TypeError(f"block must be an integer or None, got {block!r}")
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}")
    if block >= n:
        raise ValueError(
            f"block={block} leaves only one block for {n} rows, so nothing can move; "
            "use a smaller block"
        )


def block_starts(n: int, block: int) -> np.ndarray:
    """First row of each block: 0, block, 2*block, ... The last block may be shorter."""
    return np.arange(0, n, block)


def block_permutation(n: int, block: int | None, rng: np.random.Generator) -> np.ndarray:
    """A random reordering of the positions 0..n-1.

    With block=None (or 1) every position moves on its own. With a block size, the
    positions are cut into consecutive blocks and only the order of the blocks changes.
    """
    _check_block(n, block)
    if block is None or block == 1:
        return rng.permutation(n)
    starts = block_starts(n, block)
    pieces = []
    for b in rng.permutation(len(starts)):
        start = starts[b]
        stop = min(start + block, n)
        pieces.append(np.arange(start, stop))
    return np.concatenate(pieces)


def random_signs(n: int, block: int | None, rng: np.random.Generator) -> np.ndarray:
    """n random signs (+1 or -1). With a block size, each block shares one sign."""
    _check_block(n, block)
    if block is None or block == 1:
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    starts = block_starts(n, block)
    block_signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(starts))
    signs = np.empty(n, dtype=np.int8)
    for b, start in enumerate(starts):
        signs[start : start + block] = block_signs[b]
    return signs
