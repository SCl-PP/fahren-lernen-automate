"""Step 1 validation: capture a question screen, solve it, draw predicted
answer-box centers and correct-answer rings on the image. Does NOT tap.

Usage:
    uv run python scripts/validate_coords.py
    uv run python scripts/validate_coords.py --app "Fahren Lernen"
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import ImageDraw, ImageFont

from harness.capture import capture_window, find_window
from harness.config import load_config
from harness.decide import GeminiClient
from harness.solver import solve_question

RED = (220, 40, 40, 255)
GREEN = (40, 200, 60, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(image, result):
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    correct = set(result.correct_solutions)
    W, H = image.width, image.height

    for box in result.answer_boxes:
        # draw the full predicted bounding box as a thin rectangle
        x0 = int(box.xmin / 1000 * W)
        y0 = int(box.ymin / 1000 * H)
        x1 = int(box.xmax / 1000 * W)
        y1 = int(box.ymax / 1000 * H)
        cx, cy = box.center_px(W, H)

        rect_color = GREEN if box.index in correct else RED
        draw.rectangle((x0, y0, x1, y1), outline=rect_color, width=2)
        # crosshair at the predicted tap point
        draw.line((cx - 6, cy, cx + 6, cy), fill=RED, width=1)
        draw.line((cx, cy - 6, cx, cy + 6), fill=RED, width=1)

        label = str(box.index)
        tx, ty = x1 + 6, y0
        bg = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((bg[0] - 3, bg[1] - 1, bg[2] + 3, bg[3] + 1), fill=BLACK)
        draw.text((tx, ty), label, fill=WHITE, font=font)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="Fahren Lernen")
    p.add_argument("--title")
    p.add_argument("--no-open", action="store_true", help="Skip auto-open of annotated image")
    args = p.parse_args()

    window = find_window(app_name=args.app, window_title=args.title)
    capture = capture_window(window)
    print(f"Captured: {capture.image.size} px from window {window.window_id}")

    client = GeminiClient(load_config())
    result = solve_question(client, capture)

    print("\n--- EXPLANATION ---")
    print(result.explanation)
    print(f"\n--- CORRECT INDICES: {result.correct_solutions} ---")
    W, H = capture.image.width, capture.image.height
    for box in result.answer_boxes:
        mark = "X" if box.index in result.correct_solutions else " "
        cx, cy = box.center_px(W, H)
        print(
            f" [{mark}] {box.index}  bbox[{box.ymin:>4},{box.xmin:>4},{box.ymax:>4},{box.xmax:>4}]"
            f"  center=({cx:>4},{cy:>4})  {box.label}"
        )

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.png"
    annotated_path = run_dir / "annotated.png"

    capture.image.save(raw_path)
    _annotate(capture.image, result).save(annotated_path)
    print(f"\nSaved:\n  {raw_path}\n  {annotated_path}")

    if not args.no_open:
        subprocess.run(["open", str(annotated_path)], check=False)


if __name__ == "__main__":
    main()
