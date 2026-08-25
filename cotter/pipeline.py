"""Execute a RunConfig against a policy: the engine behind `cotter run`.

Runs whichever test categories the config declares and returns the
aggregated :class:`~cotter.report.TestReport`. Categories run in a fixed
order (performance, safety, regression, adversarial) with seeds derived
from ``base_seed`` per category, so adding or removing one category does
not change another's rollouts.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import gymnasium as gym

from cotter import __version__
from cotter.backends import BackendFactory
from cotter.config import RunConfig
from cotter.manifest import build_manifest
from cotter.envs.factory import WrappedEnvFactory
from cotter.envs.wrapper import CotterWrapper
from cotter.policy import Policy, load_policy
from cotter.report import TestReport
from cotter.runner import (
    make_seed_sequence,
    rollout_one,
    run_rollouts,
    run_rollouts_parallel,
)
from cotter.tests.adversarial import get_adversary, run_adversarial_test
from cotter.tests.regression import mcnemar_exact, wilcoxon_regression
from cotter.tests.safety import evaluate_safety
from cotter.tests.sprt import run_sprt

# per-category seed offsets, kept stable so results are comparable across
# configs that enable different category subsets
_PERF_SEED, _SAFETY_SEED, _REGRESSION_SEED, _ADV_SEED = 100, 200, 300, 400


def resolve_algo(name: str):
    """Look up an SB3 algorithm class by name (PPO, SAC, TD3, ...)."""
    sb3 = importlib.import_module("stable_baselines3")
    algo = getattr(sb3, name, None)
    if algo is None:
        available = [a for a in ("A2C", "DDPG", "DQN", "PPO", "SAC", "TD3") if hasattr(sb3, a)]
        raise ValueError(f"unknown SB3 algorithm '{name}' (available: {available})")
    return algo


def make_env(env_id: str, backend: str = "gymnasium", log: Callable[[str], None] = print) -> gym.Env:
    """Create the test environment via the named backend.

    The env is instrumented with :class:`CotterWrapper` when the backend
    produces a MuJoCo-backed env; otherwise safety quantities are
    unavailable and a note is logged.
    """
    env = BackendFactory.from_name(backend).make_env(env_id)
    if not isinstance(env, CotterWrapper):
        log(f"[cotter] {env_id} is not MuJoCo-backed; safety quantities unavailable")
    return env


def _obtain_adversary(policy_path, policy, env, cfg, adv_cfg, log):
    """Load a cached adversary from the zoo, or train one and cache it.

    Falls back to :func:`get_adversary` (train-with-random-fallback) when
    the zoo is disabled or the victim cannot be hashed.
    """
    if not adv_cfg.use_zoo:
        return get_adversary(
            policy, env, adv_cfg.epsilon, timesteps=adv_cfg.timesteps,
            seed=cfg.base_seed, log=log, max_seconds=adv_cfg.max_seconds,
        )

    from cotter.tests.adversarial import train_adversary
    from cotter.zoo import AdversaryZoo

    zoo = AdversaryZoo(adv_cfg.zoo_root) if adv_cfg.zoo_root else AdversaryZoo()
    # the artifact path hashes identically however it is later referenced
    victim_ref = Path(policy_path)

    cached = zoo.load(cfg.env, victim_ref, adv_cfg.epsilon)
    if cached is not None:
        log(f"[cotter]   reusing cached adversary from zoo ({zoo.root})")
        return cached, "reused cached zoo adversary"

    log(f"[cotter]   no cached adversary; training and storing in zoo ({zoo.root})")
    adversary = train_adversary(
        policy, env, adv_cfg.epsilon, timesteps=adv_cfg.timesteps,
        seed=cfg.base_seed, max_seconds=adv_cfg.max_seconds,
    )
    entry = zoo.save(adversary, cfg.env, victim_ref, notes=f"trained via cotter run, {cfg.env}")
    log(f"[cotter]   stored adversary at {entry.path}")
    return adversary, "trained PPO adversary (cached in zoo)"


def run_from_config(
    policy_path: str | Path,
    cfg: RunConfig,
    log: Callable[[str], None] = print,
) -> TestReport:
    env = make_env(cfg.env, backend=cfg.backend, log=log)
    if cfg.safety is not None and not isinstance(env, CotterWrapper):
        raise ValueError(
            f"config declares safety limits but {cfg.env} is not MuJoCo-backed, "
            "so the instrumented quantities do not exist"
        )

    algo = resolve_algo(cfg.algo)
    policy = load_policy(Path(policy_path), env, algo=algo)
    success_fn = cfg.success_fn()
    log(f"[cotter] loaded policy '{policy.name}' onto {cfg.env}")

    report = TestReport(
        policy_name=policy.name,
        env_id=cfg.env,
        metadata={"base_seed": cfg.base_seed, "success": cfg.success, "algo": cfg.algo},
        manifest=build_manifest(
            cotter_version=__version__,
            env_id=cfg.env,
            base_seed=cfg.base_seed,
            policy_path=policy_path,
        ),
    )

    if cfg.performance is not None:
        p = cfg.performance
        log(f"[cotter] performance: SPRT p0={p.p0} p1={p.p1} n_max={p.n_max}")
        seeds = make_seed_sequence(p.n_max, cfg.base_seed + _PERF_SEED)

        def trial(i: int) -> bool:
            rec = rollout_one(policy, env, seeds[i], success_fn, record_infos=False)
            log(f"[cotter]   trial {i + 1}: length={rec.length} "
                f"return={rec.total_reward:.1f} success={rec.success}")
            return rec.success

        result = run_sprt(trial, p0=p.p0, p1=p.p1, alpha=p.alpha, beta=p.beta, n_max=p.n_max)
        report.add_sprt(result)
        log(f"[cotter]   => {result.decision.value} after {result.n_trials} trials")

    factory = WrappedEnvFactory(cfg.env)

    def dispatch(pol, n, *, seeds=None, base=0, record_infos, n_workers):
        if n_workers > 1:
            return run_rollouts_parallel(
                pol, factory, n, success_fn, seeds=seeds, base_seed=base,
                record_infos=record_infos, n_workers=n_workers,
            )
        return run_rollouts(
            pol, env, n, success_fn, seeds=seeds, base_seed=base,
            record_infos=record_infos,
        )

    if cfg.safety is not None:
        s = cfg.safety
        log(f"[cotter] safety: {len(s.limits)} limit(s) over {s.n_episodes} episodes"
            + (f" ({s.n_workers} workers)" if s.n_workers > 1 else ""))
        rollouts = dispatch(
            policy, s.n_episodes, base=cfg.base_seed + _SAFETY_SEED,
            record_infos=True, n_workers=s.n_workers,
        )
        result = evaluate_safety(rollouts.episode_infos, s.limits)
        report.add_safety(result)
        for quantity, worst in result.worst_observed.items():
            limit = next(l.max_abs for l in s.limits if l.quantity == quantity)
            log(f"[cotter]   worst |{quantity}| = {worst:.4f} (limit {limit})")
        log(f"[cotter]   => {result.decision.value}")

    if cfg.regression is not None:
        r = cfg.regression
        log(f"[cotter] regression: vs baseline {r.baseline} on {r.n_pairs} paired seeds")
        baseline = load_policy(r.baseline, env, algo=algo, name=f"baseline:{r.baseline.stem}")
        seeds = make_seed_sequence(r.n_pairs, cfg.base_seed + _REGRESSION_SEED)
        # info-based metrics need per-episode final info recorded
        record = r.metric != "return"
        base_rs = dispatch(baseline, r.n_pairs, seeds=seeds, record_infos=record, n_workers=r.n_workers)
        cand_rs = dispatch(policy, r.n_pairs, seeds=seeds, record_infos=record, n_workers=r.n_workers)
        # baseline vs candidate: the CLI policy is the candidate
        mcnemar = mcnemar_exact(base_rs.successes, cand_rs.successes, alpha=r.alpha)
        report.add_regression(mcnemar, name="success_mcnemar")
        wilcoxon = wilcoxon_regression(
            base_rs.metric_values(r.metric), cand_rs.metric_values(r.metric), alpha=r.alpha
        )
        report.add_regression(wilcoxon, name=f"{r.metric}_wilcoxon")
        log(f"[cotter]   baseline {base_rs.success_rate:.0%} vs candidate "
            f"{cand_rs.success_rate:.0%} (metric: {r.metric})")
        log(f"[cotter]   => McNemar {mcnemar.decision.value} (p={mcnemar.p_value:.3g}), "
            f"Wilcoxon {wilcoxon.decision.value} (p={wilcoxon.p_value:.3g})")

    if cfg.adversarial is not None:
        a = cfg.adversarial
        log(f"[cotter] adversarial: eps={a.epsilon} over {a.n_episodes} episodes")
        random_result = run_adversarial_test(
            policy, env, success_fn, epsilon=a.epsilon, n_episodes=a.n_episodes,
            min_success_rate=a.min_success_rate, base_seed=cfg.base_seed + _ADV_SEED,
            notes="uniform random baseline",
            n_workers=a.n_workers, env_factory=factory,
        )
        report.add_adversarial(random_result, name="random_baseline")
        log(f"[cotter]   random baseline: {random_result.clean_success_rate:.0%} clean -> "
            f"{random_result.adversarial_success_rate:.0%} perturbed")
        if a.train:
            adversary, notes = _obtain_adversary(policy_path, policy, env, cfg, a, log)
            learned_result = run_adversarial_test(
                policy, env, success_fn, epsilon=a.epsilon, n_episodes=a.n_episodes,
                adversary=adversary, min_success_rate=a.min_success_rate,
                base_seed=cfg.base_seed + _ADV_SEED,
                notes=notes or "trained PPO adversary",
                n_workers=a.n_workers, env_factory=factory,
            )
            report.add_adversarial(learned_result, name=f"learned_{adversary.name}")
            log(f"[cotter]   {adversary.name} adversary: "
                f"{learned_result.clean_success_rate:.0%} clean -> "
                f"{learned_result.adversarial_success_rate:.0%} perturbed")

    if cfg.report is not None:
        path = report.to_json(cfg.report)
        log(f"[cotter] JSON report written to {path}")

    if cfg.report_html is not None:
        path = report.to_html(cfg.report_html)
        log(f"[cotter] HTML report written to {path}")

    if cfg.report_junit is not None:
        path = report.to_junit(cfg.report_junit)
        log(f"[cotter] JUnit XML report written to {path}")

    if cfg.report_md is not None:
        path = report.to_markdown(cfg.report_md)
        log(f"[cotter] Markdown report written to {path}")

    return report
