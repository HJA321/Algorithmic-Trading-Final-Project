from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from lobster_mm.config import ExperimentConfig
from lobster_mm.data import download_public_samples, prepare_all_tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare public 10-level LOBSTER samples")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download public 10-level LOBSTER samples")
    download.add_argument("--tickers", nargs="*", default=None)
    download.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare-data", help="Prepare downloaded LOBSTER files into processed datasets")
    prepare.add_argument("--tickers", nargs="*", default=None)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = ExperimentConfig(root_dir=Path(__file__).resolve().parent)

    if args.command == "download-data":
        download_public_samples(config, tickers=args.tickers, force=args.force)
        return 0

    if args.command == "prepare-data":
        prepare_all_tickers(config, tickers=args.tickers)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
