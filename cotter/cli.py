"""`cotter` command-line interface.

    cotter run --policy artifacts/victim.zip --config run.yaml
    cotter compare --baseline a.zip --candidate b.zip --config run.yaml

Exit codes: 0 = all executed categories passed (compare: no regression),
1 = at least one category failed (compare: regression detected),
2 = configuration/usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cotter import __version__
from cotter.config import ConfigError, RegressionConfig, RunConfig, load_config
from cotter.pipeline import run_from_config
from cotter.policy import SpaceMismatchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cotter",
        description="Compliance testing for AI-controlled robot policies.",
    )
    parser.add_argument("--version", action="version", version=f"cotter {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="run the test categories declared in a YAML config"
    )
    run.add_argument(
        "--policy", type=Path, required=True,
        help="policy under test (SB3 .zip or torch .pt)",
    )
    run.add_argument(
        "--config", type=Path, required=True,
        help="YAML config declaring test categories and parameters",
    )
    run.add_argument(
        "--env", default=None,
        help="Gymnasium env id (overrides the config's 'env')",
    )
    run.add_argument(
        "--report", type=Path, default=None,
        help="JSON report path (overrides the config's 'report')",
    )
    run.add_argument(
        "--report-html", type=Path, default=None,
        help="HTML report path (overrides the config's 'report_html')",
    )
    run.add_argument(
        "--report-junit", type=Path, default=None,
        help="JUnit XML report path (overrides the config's 'report_junit')",
    )
    run.add_argument(
        "--quiet", action="store_true", help="suppress progress output"
    )

    compare = subparsers.add_parser(
        "compare",
        help="run only the regression test between a baseline and a candidate",
    )
    compare.add_argument("--baseline", type=Path, required=True, help="baseline policy")
    compare.add_argument("--candidate", type=Path, required=True, help="candidate policy")
    compare.add_argument(
        "--config", type=Path, required=True,
        help="YAML config supplying env, success, base_seed, and regression params",
    )
    compare.add_argument("--env", default=None, help="Gymnasium env id (overrides config)")
    compare.add_argument(
        "--report", type=Path, default=None,
        help="write a JSON report of the comparison to this path",
    )
    compare.add_argument(
        "--report-html", type=Path, default=None,
        help="write an HTML report of the comparison to this path",
    )
    compare.add_argument(
        "--report-junit", type=Path, default=None,
        help="write a JUnit XML report of the comparison to this path",
    )
    compare.add_argument("--quiet", action="store_true", help="suppress progress output")

    list_envs = subparsers.add_parser(
        "list-envs", help="list available Gymnasium env ids grouped by package"
    )
    list_envs.add_argument(
        "--filter", default=None, help="only show env ids containing this substring"
    )

    verify = subparsers.add_parser(
        "verify", help="check a report's content hash and (optionally) its policy hash"
    )
    verify.add_argument("report", type=Path, help="path to a JSON report")
    verify.add_argument(
        "--policy", type=Path, default=None,
        help="policy file to re-hash against the report's manifest",
    )

    zoo = subparsers.add_parser("zoo", help="inspect the cached adversary zoo")
    zoo.add_argument("--root", type=Path, default=None, help="zoo root (default ~/.cotter/zoo)")
    zoo_sub = zoo.add_subparsers(dest="zoo_command", required=True)
    zoo_list = zoo_sub.add_parser("list", help="list cached adversaries")
    zoo_list.add_argument("--env", default=None, help="filter by env id")
    zoo_sub.add_parser("prune", help="remove entries whose artifact files are missing")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    log = (lambda msg: None) if args.quiet else print
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.env is not None:
        cfg.env = args.env
    if args.report is not None:
        cfg.report = args.report
    if args.report_html is not None:
        cfg.report_html = args.report_html
    if args.report_junit is not None:
        cfg.report_junit = args.report_junit

    try:
        report = run_from_config(args.policy, cfg, log=log)
    except (FileNotFoundError, SpaceMismatchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(report.summary())
    return 0 if report.overall_passed else 1


def cmd_compare(args: argparse.Namespace) -> int:
    log = (lambda msg: None) if args.quiet else print
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Regression-only run: keep the loaded regression params (n_pairs,
    # alpha, n_workers) but override the baseline with the CLI argument and
    # drop every other category so only the comparison executes.
    reg = cfg.regression or RegressionConfig()
    reg.baseline = args.baseline
    reg_only = RunConfig(
        env=args.env or cfg.env,
        success=cfg.success,
        algo=cfg.algo,
        base_seed=cfg.base_seed,
        regression=reg,
        report=args.report,
        report_html=args.report_html,
        report_junit=args.report_junit,
    )

    try:
        report = run_from_config(args.candidate, reg_only, log=log)
    except (FileNotFoundError, SpaceMismatchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(report.summary())
    regressed = any(r.passed is False for r in report.results)
    return 1 if regressed else 0


def cmd_list_envs(args: argparse.Namespace) -> int:
    from collections import defaultdict

    import gymnasium as gym

    from cotter.envs.registry import register_extension_envs

    register_extension_envs()  # make gymnasium-robotics ids visible

    groups: dict[str, set[str]] = defaultdict(set)
    for env_id, spec in gym.registry.items():
        if args.filter and args.filter.lower() not in env_id.lower():
            continue
        entry = spec.entry_point
        if isinstance(entry, str):
            package = entry.split(":", 1)[0].split(".", 1)[0]
        elif entry is not None:
            package = getattr(entry, "__module__", "unknown").split(".", 1)[0]
        else:
            package = "unknown"
        groups[package].add(env_id)

    total = sum(len(ids) for ids in groups.values())
    scope = f" matching '{args.filter}'" if args.filter else ""
    if total == 0:
        print(f"no envs{scope}")
        return 0
    print(f"{total} env id(s){scope}:")
    for package in sorted(groups):
        ids = sorted(groups[package])
        print(f"\n{package} ({len(ids)}):")
        for env_id in ids:
            print(f"  {env_id}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from cotter.verify import load_report, verify_report

    try:
        payload = load_report(args.report)
    except FileNotFoundError:
        print(f"error: report not found: {args.report}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: could not read report {args.report}: {exc}", file=sys.stderr)
        return 2

    result = verify_report(payload, policy_path=args.policy)
    mark = {True: "ok  ", False: "FAIL", None: "n/a "}
    print(f"verifying {args.report}")
    for check in result.checks:
        print(f"  [{mark[check.ok]}] {check.name}: {check.detail}")
    print(f"=> {'VERIFIED' if result.ok else 'FAILED'}")
    return 0 if result.ok else 1


def cmd_zoo(args: argparse.Namespace) -> int:
    from cotter.zoo import AdversaryZoo

    zoo = AdversaryZoo(args.root) if args.root else AdversaryZoo()

    if args.zoo_command == "list":
        entries = zoo.entries(args.env)
        scope = f" for env '{args.env}'" if args.env else ""
        if not entries:
            print(f"no cached adversaries{scope} (zoo root: {zoo.root})")
            return 0
        print(f"{len(entries)} cached adversar{'y' if len(entries) == 1 else 'ies'}"
              f"{scope} (zoo root: {zoo.root}):")
        for e in entries:
            missing = "" if (zoo.root / e.path).exists() else "  [MISSING ARTIFACT]"
            print(f"  {e.env_id}  eps={e.epsilon:g}  victim={e.victim_hash}  "
                  f"{e.algo}  {e.created_at}{missing}")
            print(f"      {e.path}")
        return 0

    if args.zoo_command == "prune":
        removed = zoo.prune()
        if not removed:
            print(f"nothing to prune (zoo root: {zoo.root})")
            return 0
        print(f"pruned {len(removed)} entr{'y' if len(removed) == 1 else 'ies'} "
              f"with missing artifacts:")
        for e in removed:
            print(f"  {e.env_id}  eps={e.epsilon:g}  victim={e.victim_hash}")
        return 0

    raise AssertionError(f"unhandled zoo command {args.zoo_command}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "zoo":
        return cmd_zoo(args)
    if args.command == "list-envs":
        return cmd_list_envs(args)
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
