"""Cotter — compliance testing for AI-controlled robot policies.

Load a trained policy as a black box, run it through standardized test
categories (performance, safety, regression, adversarial) in a MuJoCo
simulation, and get structured pass/fail results with statistical
guarantees.
"""

from cotter.envs.wrapper import (
    ACTUATOR_FORCES,
    CONTACT_COUNT,
    INSTRUMENTED_KEYS,
    JOINT_VELOCITIES,
    CotterWrapper,
)
from cotter.policy import (
    Policy,
    SB3Policy,
    SpaceMismatchError,
    TorchPolicy,
    load_policy,
    validate_spaces,
)
from cotter.report import TestReport
from cotter.runner import (
    EpisodeRecord,
    RolloutSet,
    make_seed_sequence,
    rollout_one,
    run_rollouts,
)
from cotter.tests.adversarial import (
    AdversarialResult,
    NullAdversary,
    ObservationPerturbationEnv,
    RandomAdversary,
    get_adversary,
    run_adversarial_test,
    train_adversary,
)
from cotter.tests.regression import (
    RegressionDecision,
    RegressionResult,
    mcnemar_exact,
    wilcoxon_regression,
)
from cotter.tests.safety import (
    SafetyDecision,
    SafetyLimit,
    SafetyResult,
    SafetyViolation,
    check_step,
    evaluate_safety,
)
from cotter.tests.sprt import SPRT, SPRTDecision, SPRTResult, run_sprt

__version__ = "0.1.0"

__all__ = [
    # envs
    "CotterWrapper",
    "JOINT_VELOCITIES",
    "ACTUATOR_FORCES",
    "CONTACT_COUNT",
    "INSTRUMENTED_KEYS",
    # policy
    "Policy",
    "SB3Policy",
    "TorchPolicy",
    "load_policy",
    "validate_spaces",
    "SpaceMismatchError",
    # runner
    "EpisodeRecord",
    "RolloutSet",
    "rollout_one",
    "run_rollouts",
    "make_seed_sequence",
    # performance
    "SPRT",
    "SPRTDecision",
    "SPRTResult",
    "run_sprt",
    # safety
    "SafetyLimit",
    "SafetyViolation",
    "SafetyResult",
    "SafetyDecision",
    "check_step",
    "evaluate_safety",
    # regression
    "RegressionDecision",
    "RegressionResult",
    "mcnemar_exact",
    "wilcoxon_regression",
    # adversarial
    "AdversarialResult",
    "RandomAdversary",
    "NullAdversary",
    "ObservationPerturbationEnv",
    "run_adversarial_test",
    "train_adversary",
    "get_adversary",
    # report
    "TestReport",
    "__version__",
]
