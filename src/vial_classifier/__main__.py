"""Command-line entry point: ``python -m vial_classifier <command>``.

Commands map one-to-one to the stages of the pipeline so that the Dagger
workflow and GitHub Actions can call each stage on its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vial_classifier", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pull-data", help="dvc pull and download the raw images")

    train = sub.add_parser("train", help="train the CNN and register it in Staging")
    train.add_argument("--no-pull", action="store_true", help="skip dvc pull and image download")

    sub.add_parser("promote", help="promote the best run if it beats Production and export the model")

    serve = sub.add_parser("serve", help="serve the exported model over HTTP")
    serve.add_argument("--port", type=int, default=None, help="API port (default from config)")
    serve.add_argument("--model-dir", default=None, help="exported model directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Imported lazily so that `--help` works without the heavy ML dependencies
    from . import config

    if args.command == "pull-data":
        from .data import pull_data
        from .pipelines import dvc_pull

        dvc_pull()
        pull_data()
        return 0

    if args.command == "train":
        from .pipelines import run_training

        summary = run_training(pull=not args.no_pull)
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "promote":
        from .pipelines import run_promotion

        summary = run_promotion()
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "serve":
        from pathlib import Path

        from .serving import run_server

        run_server(
            model_dir=Path(args.model_dir) if args.model_dir else config.EXPORT_MODEL_DIR,
            api_port=args.port or config.API_PORT,
        )
        return 0

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
