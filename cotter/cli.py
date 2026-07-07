"""`cotter` command-line interface.

    cotter run --policy artifacts/victim.zip --config run.yaml
    cotter run --policy victim.zip --env InvertedPendulum-v5 --config run.yaml

Exit codes: 0 = all executed categories passed, 1 = at least one
category failed, 2 = configuration/usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cotter import __version__
from cotter.config import ConfigError, load_config
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
        "--quiet", action="store_true", help="suppress progress output"
    )
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

    try:
        report = run_from_config(args.policy, cfg, log=log)
    except (FileNotFoundError, SpaceMismatchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(report.summary())
    return 0 if report.overall_passed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
