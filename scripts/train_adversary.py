"""Train a PPO adversary against a frozen victim policy from the CLI.

Trains on ObservationPerturbationEnv (adversary action = bounded L-inf
perturbation of what the victim observes), evaluates clean vs attacked
success rate on shared seeds, and saves the adversary artifact.

Usage:
    poetry run python scripts/train_adversary.py \
        [--victim artifacts/victim_ppo_inverted_pendulum.zip] \
        [--epsilon 0.07] [--timesteps 150000] [--episodes 20]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from cotter import CotterWrapper, load_policy, run_adversarial_test, train_adversary

ROOT = Path(__file__).resolve().parent.parent
ENV_ID = "InvertedPendulum-v5"


def survived_full_horizon(total_reward, length, terminated, truncated, final_info):
    return length >= 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--victim", type=Path,
        default=ROOT / "artifacts" / "victim_ppo_inverted_pendulum.zip",
    )
    parser.add_argument("--epsilon", type=float, default=0.07)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CotterWrapper(gym.make(ENV_ID))
    victim = load_policy(args.victim, env, algo=PPO, name=args.victim.stem)

    start = time.time()
    adversary = train_adversary(
        victim, env, epsilon=args.epsilon, timesteps=args.timesteps, seed=args.seed
    )
    print(f"trained PPO adversary in {time.time() - start:.0f}s "
          f"(eps={args.epsilon}, {args.timesteps} timesteps)")

    result = run_adversarial_test(
        victim, env, survived_full_horizon, epsilon=args.epsilon,
        n_episodes=args.episodes, adversary=adversary, base_seed=args.seed,
    )
    print(result.summary())

    eps_tag = f"{args.epsilon:g}".replace(".", "")
    out = args.out or ROOT / "artifacts" / f"adversary_ppo_eps{eps_tag}.zip"
    out.parent.mkdir(exist_ok=True)
    adversary.model.save(out)
    print(f"saved adversary to {out}")


if __name__ == "__main__":
    main()
