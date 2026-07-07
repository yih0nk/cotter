"""Train the Dict-obs demo victim: PPO on FetchReachDense-v4.

Produces artifacts/victim_ppo_fetch_reach.zip. Success metric reported
is the fraction of evaluation episodes ending with is_success (end
effector within 5 cm of the goal at the 50-step horizon).

Usage: poetry run python scripts/train_fetch_victim.py [--timesteps N]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from cotter.envs.registry import make_env_by_id

SEED = 0
ENV_ID = "FetchReachDense-v4"
ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "victim_ppo_fetch_reach.zip"


def final_success_rate(model, env, n_episodes: int = 20, seed: int = 1000) -> float:
    wins = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        wins += bool(info["is_success"])
    return wins / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env_by_id(ENV_ID)
    model = PPO(
        "MultiInputPolicy",
        env,
        seed=SEED,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.95,  # short 50-step horizon
        verbose=1,
        device="cpu",
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - start

    eval_env = make_env_by_id(ENV_ID)
    rate = final_success_rate(model, eval_env)

    ARTIFACT.parent.mkdir(exist_ok=True)
    model.save(ARTIFACT)
    print(
        f"\ntrained {args.timesteps} timesteps in {elapsed:.0f}s; "
        f"final-step success rate {rate:.0%} over 20 eval episodes\n"
        f"saved {ARTIFACT}"
    )


if __name__ == "__main__":
    main()
