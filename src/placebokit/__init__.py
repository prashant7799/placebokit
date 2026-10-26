"""placebokit: negative controls and train/serve parity checks for model results.

If a placebo that cannot know the answer scores like your model, the score shows no real skill.
"""

from .controls import (
    Ablate,
    Context,
    Control,
    LagShift,
    PermutedWeights,
    RandomEntries,
    RandomWeights,
    ShuffledLabels,
    SignFlip,
)
from .parity import (
    ParityReport,
    check_causal,
    check_parity,
    check_repeatable,
    check_warmup,
    perturb_numbers,
)
from .report import ControlReport, ControlResult
from .runner import run_controls, run_prediction_controls
from .shuffle import block_permutation, random_signs
from .stats import empirical_p_value
from .testing import assert_beats_placebos, assert_parity

__version__ = "0.1.0"

__all__ = [
    "Ablate",
    "Context",
    "Control",
    "ControlReport",
    "ControlResult",
    "LagShift",
    "ParityReport",
    "PermutedWeights",
    "RandomEntries",
    "RandomWeights",
    "ShuffledLabels",
    "SignFlip",
    "__version__",
    "assert_beats_placebos",
    "assert_parity",
    "block_permutation",
    "check_causal",
    "check_parity",
    "check_repeatable",
    "check_warmup",
    "empirical_p_value",
    "perturb_numbers",
    "random_signs",
    "run_controls",
    "run_prediction_controls",
]
