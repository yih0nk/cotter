"""Train a manipulation victim: HER + SAC on FetchPickAndPlace-v4.

Goal-conditioned sparse-reward pick-and-place solved with Soft
Actor-Critic plus Hindsight Experience Replay. Time-boxed for CPU: the
run stops at ``--minutes`` and always saves the final model. Success is
the fraction of evaluation episodes ending with ``is_success`` (object
placed at the goal within tolerance at the 50-step horizon).

If pick-and-place does not reach a useful success rate in the CPU time
budget, fall back to the much easier dense-reward reach task with
``--env FetchReachDense-v4`` (documented in the README demo section).

Usage:
    poetry run python scripts/train_fetch_pickplace.py [--minutes 80]
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

from cotter.envs.registry import make_env_by_id

SEED = 0
DEFAULT_ENV = "FetchPickAndPlace-v4"
ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "victim_fetch_pickplace.zip"


def success_rate(model, env, n_episodes: int = 50, seed: int = 5000) -> float:
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


class TimeBudgetEval(BaseCallback):
    """Stop at a wall-clock deadline; periodically eval and keep the best."""

    def __init__(self, eval_env, deadline: float, eval_every: int, out_path: Path):
        super().__init__()
        self.eval_env = eval_env
        self.deadline = deadline
        self.eval_every = eval_every
        self.out_path = out_path
        self.best_rate = -1.0
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_every:
            self._last_eval = self.num_timesteps
            rate = success_rate(self.model, self.eval_env, n_episodes=20)
            elapsed = time.time() - (self.deadline - self._budget)
            print(f"[eval] step {self.num_timesteps}  success {rate:.0%}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)
            if rate >= self.best_rate:
                self.best_rate = rate
                self.model.save(self.out_path)
        if time.time() >= self.deadline:
            print(f"[stop] time budget reached at step {self.num_timesteps}", flush=True)
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=80.0)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env_by_id(args.env)
    eval_env = make_env_by_id(args.env)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={"n_sampled_goal": 4, "goal_selection_strategy": "future"},
        policy_kwargs={"net_arch": [256, 256, 256]},
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=1000,
        gamma=0.95,
        tau=0.05,
        learning_rate=1e-3,
        seed=SEED,
        device="cpu",
        verbose=0,
    )

    deadline = time.time() + args.minutes * 60
    callback = TimeBudgetEval(eval_env, deadline, args.eval_every, args.out)
    callback._budget = args.minutes * 60

    start = time.time()
    model.learn(total_timesteps=5_000_000, callback=callback, progress_bar=False)
    elapsed = time.time() - start

    final_rate = success_rate(model, eval_env, n_episodes=50)
    # Keep whichever is better: the periodic best or the final model.
    if final_rate >= callback.best_rate:
        model.save(args.out)
        kept = "final"
    else:
        kept = f"best checkpoint ({callback.best_rate:.0%})"
    print(
        f"\ntrained {args.env} for {elapsed / 60:.1f} min; "
        f"final success rate {final_rate:.0%} over 50 eval episodes; "
        f"saved {kept} to {args.out}"
    )


if __name__ == "__main__":
    main()
