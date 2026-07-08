"""Train a locomotion victim: PPO on Ant-v5.

Standard MuJoCo quadruped locomotion. PPO is used for fast wall-clock
training on CPU; 500k timesteps comfortably clears a positive average
return (Ant's per-step survive bonus means an upright, non-falling gait
scores well above zero). Saves artifacts/victim_ant.zip.

Usage:
    poetry run python scripts/train_ant_victim.py [--timesteps 500000]
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
ENV_ID = "Ant-v5"
ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ant.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        policy_kwargs={"net_arch": [256, 256]},
        n_steps=2048,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
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

    args.out.parent.mkdir(exist_ok=True)
    model.save(args.out)
    print(
        f"\ntrained {ENV_ID} for {elapsed / 60:.1f} min ({args.timesteps} steps); "
        f"eval return {mean_reward:.0f} +/- {std_reward:.0f} over 20 episodes; "
        f"saved {args.out}"
    )


if __name__ == "__main__":
    main()
