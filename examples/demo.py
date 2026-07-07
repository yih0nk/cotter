"""End-to-end Cotter demo: all four test categories on a real policy.

Runs against a PPO policy trained on MuJoCo InvertedPendulum-v5 (see
scripts/train_victim.py; the artifact is checked in and retrained
automatically if missing). Prints the console report and writes
artifacts/demo_report.json.

What each category demonstrates:

* performance — Wald SPRT on "episode survives the full 1000-step
  horizon", H0: success rate <= 0.80 vs H1: >= 0.95, alpha = beta = 0.05.
* safety — hard limits on joint velocities, actuator forces, and contact
  count, checked at every timestep of 20 episodes.
* regression — the victim (baseline) vs a deliberately undertrained
  5k-step candidate on the same 30 seeds, exact McNemar + Wilcoxon.
* adversarial — L-inf observation perturbations at eps = 0.07: the random
  baseline adversary, then a PPO adversary trained against the frozen
  victim. This is the differentiator: random noise at this budget is
  harmless, so only the learned attack reveals the true worst case.

Runtime: roughly 2-4 minutes on Apple Silicon CPU (dominated by
adversary training).

Usage: poetry run python examples/demo.py [--skip-adversary-training]
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from cotter import (
    CotterWrapper,
    SafetyLimit,
    TestReport,
    evaluate_safety,
    get_adversary,
    load_policy,
    make_seed_sequence,
    mcnemar_exact,
    rollout_one,
    run_adversarial_test,
    run_rollouts,
    run_sprt,
    wilcoxon_regression,
)
from cotter.envs.wrapper import (
    ACTUATOR_FORCES,
    CONTACT_COUNT,
    CONTACT_FORCES,
    JOINT_VELOCITIES,
)

SEED = 0
ENV_ID = "InvertedPendulum-v5"
ROOT = Path(__file__).resolve().parent.parent
VICTIM_PATH = ROOT / "artifacts" / "victim_ppo_inverted_pendulum.zip"
REPORT_PATH = ROOT / "artifacts" / "demo_report.json"
ADVERSARY_PATH = ROOT / "artifacts" / "adversary_ppo_eps007.zip"

EPSILON = 0.07  # L-inf observation perturbation budget
SAFETY_LIMITS = [
    SafetyLimit(JOINT_VELOCITIES, 5.0),  # rad/s hinge, m/s slider
    SafetyLimit(ACTUATOR_FORCES, 2.5),  # N on the cart actuator
    SafetyLimit(CONTACT_COUNT, 0.5),  # any contact at all is a violation
    SafetyLimit(CONTACT_FORCES, 1.0),  # N per-body external contact wrench
]


def survived_full_horizon(total_reward, length, terminated, truncated, final_info):
    """Task success: the pole stays balanced for the entire 1000-step episode."""
    return length >= 1000


def stage(title: str):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-adversary-training",
        action="store_true",
        help="use only the random-baseline adversary (much faster)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    start = time.time()

    env = CotterWrapper(gym.make(ENV_ID))

    if not VICTIM_PATH.exists():
        print(f"victim artifact missing; training it now via scripts/train_victim.py")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_victim.py")], check=True
        )
    victim = load_policy(VICTIM_PATH, env, algo=PPO, name="victim_ppo")
    print(f"loaded victim policy from {VICTIM_PATH.relative_to(ROOT)}")

    report = TestReport(
        policy_name="victim_ppo",
        env_id=ENV_ID,
        metadata={
            "seed": SEED,
            "epsilon": EPSILON,
            "safety_limits": {l.quantity: l.max_abs for l in SAFETY_LIMITS},
            "success_criterion": "episode length >= 1000 (full horizon survived)",
        },
    )

    # 1. performance ------------------------------------------------------
    stage("performance: sequential probability ratio test")
    perf_seeds = make_seed_sequence(50, base_seed=SEED + 100)

    def trial(i: int) -> bool:
        rec = rollout_one(
            victim, env, perf_seeds[i], survived_full_horizon, record_infos=False
        )
        print(f"  trial {i + 1}: length={rec.length} success={rec.success}")
        return rec.success

    sprt_result = run_sprt(trial, p0=0.80, p1=0.95, alpha=0.05, beta=0.05, n_max=50)
    report.add_sprt(sprt_result)
    print(f"  => {sprt_result.decision.value} after {sprt_result.n_trials} trials")

    # 2. safety ------------------------------------------------------------
    stage("safety: per-timestep hard limits over 20 episodes")
    safety_rollouts = run_rollouts(
        victim, env, 20, survived_full_horizon, base_seed=SEED + 200
    )
    safety_result = evaluate_safety(safety_rollouts.episode_infos, SAFETY_LIMITS)
    report.add_safety(safety_result)
    print(f"  checked {safety_result.n_timesteps_checked} timesteps")
    for q, worst in safety_result.worst_observed.items():
        limit = next(l.max_abs for l in SAFETY_LIMITS if l.quantity == q)
        print(f"  worst |{q}| = {worst:.4f} (limit {limit})")
    print(f"  => {safety_result.decision.value}")

    # 3. regression --------------------------------------------------------
    stage("regression: victim vs deliberately undertrained candidate")
    print("  training candidate (PPO, 5k timesteps — deliberately undertrained)")
    cand_model = PPO(
        "MlpPolicy",
        gym.make(ENV_ID),
        seed=SEED + 1,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=2048,
        verbose=0,
        device="cpu",
    )
    cand_model.learn(total_timesteps=5_000)
    candidate = load_policy(cand_model, env, name="candidate_undertrained")

    shared_seeds = make_seed_sequence(30, base_seed=SEED + 300)
    base_rs = run_rollouts(
        victim, env, 30, survived_full_horizon, seeds=shared_seeds, record_infos=False
    )
    cand_rs = run_rollouts(
        candidate, env, 30, survived_full_horizon, seeds=shared_seeds, record_infos=False
    )
    mcnemar = mcnemar_exact(base_rs.successes, cand_rs.successes)
    report.add_regression(mcnemar, name="success_mcnemar")
    wilcoxon = wilcoxon_regression(base_rs.returns, cand_rs.returns)
    report.add_regression(wilcoxon, name="return_wilcoxon")
    print(
        f"  baseline {base_rs.success_rate:.0%} vs candidate {cand_rs.success_rate:.0%} "
        f"on 30 shared seeds"
    )
    print(f"  => McNemar {mcnemar.decision.value} (p={mcnemar.p_value:.2e}), "
          f"Wilcoxon {wilcoxon.decision.value} (p={wilcoxon.p_value:.2e})")

    # 4. adversarial -------------------------------------------------------
    stage(f"adversarial: L-inf observation perturbation, eps={EPSILON}")
    random_result = run_adversarial_test(
        victim, env, survived_full_horizon, epsilon=EPSILON,
        n_episodes=20, min_success_rate=0.5, base_seed=SEED + 400,
        notes="uniform random baseline",
    )
    report.add_adversarial(random_result, name="random_baseline")
    print(f"  random baseline: {random_result.clean_success_rate:.0%} clean -> "
          f"{random_result.adversarial_success_rate:.0%} perturbed")

    adversary, notes = get_adversary(
        victim, env, EPSILON,
        timesteps=150_000, seed=SEED, train=not args.skip_adversary_training,
    )
    if adversary.name == "ppo":
        print("  trained PPO adversary against the frozen victim (150k timesteps)")
        adversary.model.save(ADVERSARY_PATH)
    learned_result = run_adversarial_test(
        victim, env, survived_full_horizon, epsilon=EPSILON,
        n_episodes=20, adversary=adversary, min_success_rate=0.5,
        base_seed=SEED + 400, notes=notes or "trained PPO adversary",
    )
    report.add_adversarial(learned_result, name=f"learned_{adversary.name}")
    print(f"  {adversary.name} adversary: {learned_result.clean_success_rate:.0%} clean -> "
          f"{learned_result.adversarial_success_rate:.0%} perturbed")

    # report ---------------------------------------------------------------
    print(f"\ntotal wall time: {time.time() - start:.0f}s\n")
    print(report.summary())
    path = report.to_json(REPORT_PATH)
    print(f"\nJSON report written to {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
