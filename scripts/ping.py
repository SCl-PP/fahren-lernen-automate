"""Minimal text-only Gemini call. No image, no schema, no streaming.

Isolates whether the API itself responds for your key + model. If this hangs
or 504s, the problem is upstream (model overloaded or key-level issue), not
in our vision/structured-output path.

Usage:
    uv run python scripts/ping.py
    uv run python scripts/ping.py --model gemini-4-hydrogen
"""
import argparse
import time

from google import genai
from google.genai import types

from harness.config import load_config
from harness.decide import enable_debug_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Override GEMINI_MODEL from env")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--quiet", action="store_true", help="Skip wire-level logs")
    args = p.parse_args()

    if not args.quiet:
        enable_debug_logging()

    cfg = load_config()
    model = args.model or cfg.model
    print(f"Pinging {model} (timeout {args.timeout}s)…")

    client = genai.Client(
        api_key=cfg.gemini_api_key,
        http_options=types.HttpOptions(timeout=int(args.timeout * 1000)),
    )

    t0 = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents="Reply with the single word: pong",
    )
    print(f"\n--- response in {time.monotonic() - t0:.2f}s ---")
    print(response.text)


if __name__ == "__main__":
    main()
