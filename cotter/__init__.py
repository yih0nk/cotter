"""Cotter — compliance testing for AI-controlled robot policies.

Load a trained policy as a black box, run it through standardized test
categories (performance, safety, regression, adversarial) in a MuJoCo
simulation, and get structured pass/fail results with statistical
guarantees.
"""

from cotter.envs.wrapper import (
    ACTUATOR_FORCES,
    CONTACT_COUNT,
    CONTACT_FORCES,
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
    run_rollouts_parallel,
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
from cotter.tests.action_adversarial import (
    ActionPerturbationEnv,
    ActionPerturbedPolicy,
    NullActionAdversary,
    PPOActionAdversary,
    RandomActionAdversary,
    get_action_adversary,
    run_action_adversarial_test,
    train_action_adversary,
)
from cotter.tests.transfer import (
    IncompatibleAdversaryError,
    check_adversary_compatible,
    transfer_attack,
)
from cotter.tests.iso15066 import (
    ISO_TS_15066_LIMITS,
    BodyRegion,
    PFLResult,
    body_region,
    collision_energy,
    evaluate_pfl,
    max_relative_speed,
    peak_force,
    reduced_mass,
    region_limits,
)
from cotter.zoo import AdversaryZoo, ZooEntry, victim_hash
from cotter.zoo.pretrained import PretrainedEntry, PretrainedZoo
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
from cotter.backends import (
    BackendFactory,
    BackendNotAvailableError,
    GymnasiumBackend,
    IsaacSimBackend,
)
from cotter.stats import clopper_pearson
from cotter.tests.sprt import SPRT, SPRTDecision, SPRTResult, run_sprt

__version__ = "0.2.0"

__all__ = [
    # envs
    "CotterWrapper",
    "JOINT_VELOCITIES",
    "ACTUATOR_FORCES",
    "CONTACT_COUNT",
    "CONTACT_FORCES",
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
    "run_rollouts_parallel",
    "make_seed_sequence",
    # performance
    "SPRT",
    "SPRTDecision",
    "SPRTResult",
    "run_sprt",
    # stats
    "clopper_pearson",
    # backends
    "BackendFactory",
    "BackendNotAvailableError",
    "GymnasiumBackend",
    "IsaacSimBackend",
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
    # action-space adversarial
    "ActionPerturbationEnv",
    "ActionPerturbedPolicy",
    "RandomActionAdversary",
    "NullActionAdversary",
    "PPOActionAdversary",
    "run_action_adversarial_test",
    "train_action_adversary",
    "get_action_adversary",
    # transfer attacks + zoo
    "transfer_attack",
    "check_adversary_compatible",
    "IncompatibleAdversaryError",
    "AdversaryZoo",
    "ZooEntry",
    "victim_hash",
    "PretrainedZoo",
    "PretrainedEntry",
    # ISO/TS 15066 power-and-force-limiting
    "ISO_TS_15066_LIMITS",
    "BodyRegion",
    "body_region",
    "reduced_mass",
    "collision_energy",
    "peak_force",
    "max_relative_speed",
    "region_limits",
    "evaluate_pfl",
    "PFLResult",
    # report
    "TestReport",
    "__version__",
]
