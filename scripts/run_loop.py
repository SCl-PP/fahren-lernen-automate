"""Unified navigation loop entry point.

Capture -> triage -> dispatch (foto/text/film/input) -> submit -> advance.
Simulates a user by answering correctly CORRECT_RATIO fraction of the time.

Each iteration writes artifacts under runs/session-<ts>/iter-NNNN/.
A session.jsonl in the session root logs one line per iteration.

Usage:
    uv run python scripts/run_loop.py                   # unbounded
    uv run python scripts/run_loop.py --max 3           # bounded test run
    CORRECT_RATIO=0.9 uv run python scripts/run_loop.py # 90%% correct
"""
import argparse
import logging
from dataclasses import replace

from harness.act import accessibility_granted
from harness.config import load_config
from harness.loop import load_loop_config, run_loop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="Fahren Lernen")
    p.add_argument("--title")
    p.add_argument("--max", type=int, default=None,
                   help="Override MAX_ITERATIONS env var (bounded test runs)")
    p.add_argument("--correct", type=float, default=None,
                   help="Override CORRECT_RATIO env var")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence the chatty HTTP libraries — only surface warnings and errors.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not accessibility_granted():
        print("WARNING: Accessibility permission not granted — taps will be dropped.")

    cfg = load_config()
    loop_cfg = load_loop_config()
    if args.max is not None:
        loop_cfg = replace(loop_cfg, max_iterations=args.max)
    if args.correct is not None:
        loop_cfg = replace(loop_cfg, correct_ratio=args.correct)

    print(
        f"model={cfg.model}  correct_ratio={loop_cfg.correct_ratio}  "
        f"max_iter={loop_cfg.max_iterations}  film_s={loop_cfg.film_record_seconds}"
    )
    session = run_loop(cfg, loop_cfg, app_name=args.app, window_title=args.title)
    print(f"\nSession artifacts: {session}")


if __name__ == "__main__":
    main()
