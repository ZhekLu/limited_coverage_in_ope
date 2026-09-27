"""Thin command-line entry points for documented Makefile workflows."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Dispatch to experiment, analysis or plotting objects."""
    parser = argparse.ArgumentParser(description="Controlled OPE reliability study")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Collect independent datasets and evaluate estimators")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    for name in ("plot", "summarize"):
        command = commands.add_parser(name, help="Regenerate saved-data outputs independently")
        command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            from limited_ope.experiment import CoverageExperiment

            output = CoverageExperiment.from_file(args.config).run(resume=args.resume)
        elif args.command == "plot":
            from limited_ope.plotting import StudyPlotter

            output = StudyPlotter(args.run).render()
        else:
            from limited_ope.analysis import ResultsAnalyzer

            output = ResultsAnalyzer(args.run).summarize()
    except (ValueError, FileExistsError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
