"""Train the demo victim policy: PPO on InvertedPendulum-v5.

Produces artifacts/victim_ppo_inverted_pendulum.zip, the policy under
test in examples/demo.py. Sized to train in a few minutes on CPU; the
task is considered solved when episodes reliably reach the 1000-step
time limit.

Usage: poetry run python scripts/train_victim.py [--timesteps N]
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
from stable_baselines3.common.evaluation import evaluate_policy

SEED = 0
ENV_ID = "InvertedPendulum-v5"
ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_inverted_pendulum.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
        device="cpu",
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - start

    eval_env = gym.make(ENV_ID)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )

    ARTIFACT.parent.mkdir(exist_ok=True)
    model.save(ARTIFACT)
    print(
        f"\ntrained {args.timesteps} timesteps in {elapsed:.0f}s; "
        f"eval mean_reward={mean_reward:.1f} +/- {std_reward:.1f} (max 1000)\n"
        f"saved {ARTIFACT}"
    )


if __name__ == "__main__":
    main()
