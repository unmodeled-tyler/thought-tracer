from __future__ import annotations

import argparse

from rich.console import Console

from .app import LogitLensApp
from .modeling import ModelArtifactError, load_ministral_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Terminal logit lens for Ministral 3 3B.")
    parser.add_argument(
        "--model-path",
        default="./Ministral-3-3B-Instruct-2512",
        help="Path to the local model directory.",
    )
    parser.add_argument(
        "--dequantize-bf16",
        action="store_true",
        help="Request BF16 dequantization via FineGrainedFP8Config when supported.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    try:
        loaded = load_ministral_model(
            args.model_path,
            dequantize_to_bf16=args.dequantize_bf16,
        )
    except ModelArtifactError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    app = LogitLensApp(loaded=loaded, console=console)
    app.run()


if __name__ == "__main__":
    main()
