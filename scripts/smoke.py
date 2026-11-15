"""End-to-end wiring check: capture a window → send to Gemini → print JSON.

Usage:
    uv run python scripts/smoke.py --list
    uv run python scripts/smoke.py --app Safari
    uv run python scripts/smoke.py --title "iPhone"
"""
import argparse
import json

from pydantic import BaseModel, Field

from harness.capture import capture_window, find_window, list_windows
from harness.config import load_config
from harness.decide import GeminiClient, enable_debug_logging


class SceneCheck(BaseModel):
    summary: str = Field(..., description="One sentence describing what is on screen.")
    visible_interactive_elements: list[str] = Field(
        default_factory=list,
        description="Labels of buttons, links, or inputs that look interactive.",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", help="Substring match against window owner (app name)")
    p.add_argument("--title", help="Substring match against window title")
    p.add_argument("--list", action="store_true", help="List visible windows and exit")
    p.add_argument("--debug", action="store_true", help="Enable wire-level HTTP logging")
    args = p.parse_args()

    if args.debug:
        enable_debug_logging()

    if args.list or (not args.app and not args.title):
        for w in list_windows():
            if not w.owner:
                continue
            print(
                f"[{w.window_id:>7}] {w.owner!r} :: {w.title!r}  "
                f"{int(w.width)}x{int(w.height)} @ ({int(w.x)},{int(w.y)})"
            )
        return

    window = find_window(app_name=args.app, window_title=args.title)
    print(f"Matched: {window}")

    capture = capture_window(window)
    print(f"Captured: {capture.image.size} px  (scale_x={capture.scale_x:.2f})")

    client = GeminiClient(load_config())
    result = client.ask(
        prompt=(
            "This is a screenshot of an iPhone app window running on macOS. "
            "Return JSON matching the schema that summarizes the scene."
        ),
        image=capture.image,
        schema=SceneCheck,
    )
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
