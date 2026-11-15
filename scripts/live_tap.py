"""Step 2: capture -> solve -> tap one answer -> verify screen changed.

Fires exactly one synthetic tap through CGEvent, then uses perceptual hashing
to confirm the window content changed afterwards. Saves pre/post screenshots
and the full solver response to a run directory for audit.

Usage:
    uv run python scripts/live_tap.py                     # taps answer 1
    uv run python scripts/live_tap.py --index 2           # taps answer 2
    uv run python scripts/live_tap.py --countdown 5       # longer abort window
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import ImageDraw, ImageFont

from harness.act import accessibility_granted, image_to_screen, tap
from harness.capture import capture_window, find_window
from harness.config import load_config
from harness.decide import GeminiClient
from harness.solver import solve_question

RED = (220, 40, 40, 255)
GREEN = (40, 200, 60, 255)
BLUE = (40, 90, 220, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

PHASH_SAME_THRESHOLD = 4  # Hamming distance <= 4 counts as "unchanged"


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate(image, result, tap_index: int):
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    correct = set(result.correct_solutions)
    W, H = image.width, image.height

    for box in result.answer_boxes:
        x0 = int(box.xmin / 1000 * W)
        y0 = int(box.ymin / 1000 * H)
        x1 = int(box.xmax / 1000 * W)
        y1 = int(box.ymax / 1000 * H)
        cx, cy = box.center_px(W, H)

        if box.index == tap_index:
            rect_color = BLUE  # explicit tap target
        elif box.index in correct:
            rect_color = GREEN
        else:
            rect_color = RED
        draw.rectangle((x0, y0, x1, y1), outline=rect_color, width=2)
        draw.line((cx - 6, cy, cx + 6, cy), fill=RED, width=1)
        draw.line((cx, cy - 6, cx, cy + 6), fill=RED, width=1)

        label = f"{box.index}{'*' if box.index == tap_index else ''}"
        tx, ty = x1 + 6, y0
        bg = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((bg[0] - 3, bg[1] - 1, bg[2] + 3, bg[3] + 1), fill=BLACK)
        draw.text((tx, ty), label, fill=WHITE, font=font)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="Fahren Lernen")
    p.add_argument("--title")
    p.add_argument("--index", type=int, default=1, help="1-based answer index to tap")
    p.add_argument("--countdown", type=int, default=3, help="Seconds to abort with Ctrl+C before tap")
    p.add_argument("--settle", type=float, default=1.0, help="Seconds to wait after tap before re-capture")
    args = p.parse_args()

    ax_ok = accessibility_granted()
    print(f"Accessibility trusted: {ax_ok}")
    if not ax_ok:
        print(
            "  CGEvent taps will be silently dropped without this permission.\n"
            "  System Settings -> Privacy & Security -> Accessibility -> add Terminal,\n"
            "  then quit and reopen Terminal before running this script."
        )

    window = find_window(app_name=args.app, window_title=args.title)
    print(f"Window: {window.owner!r} id={window.window_id} "
          f"{int(window.width)}x{int(window.height)} @ ({int(window.x)},{int(window.y)})")

    pre = capture_window(window)
    print(f"Captured pre: {pre.image.size} px")

    client = GeminiClient(load_config())
    result = solve_question(client, pre)

    print(f"\n--- CORRECT INDICES: {result.correct_solutions} ---")
    W, H = pre.image.width, pre.image.height
    for box in result.answer_boxes:
        mark = "X" if box.index in result.correct_solutions else " "
        cx, cy = box.center_px(W, H)
        pick = " <<< TAP" if box.index == args.index else ""
        print(f" [{mark}] {box.index}  center=({cx:>4},{cy:>4})  {box.label}{pick}")

    boxes_by_index = {b.index: b for b in result.answer_boxes}
    if args.index not in boxes_by_index:
        raise SystemExit(f"--index {args.index} not in solver output {sorted(boxes_by_index)}")

    target = boxes_by_index[args.index]
    cx_img, cy_img = target.center_px(W, H)
    screen_pt = image_to_screen(cx_img, cy_img, pre)
    print(
        f"\nTap target: img=({cx_img},{cy_img})  screen=({screen_pt.x:.1f},{screen_pt.y:.1f})"
    )

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    pre.image.save(run_dir / "pre.png")
    _annotate(pre.image, result, args.index).save(run_dir / "annotated.png")
    (run_dir / "solver_response.json").write_text(result.model_dump_json(indent=2))

    print(f"\nFiring tap in {args.countdown}s — Ctrl+C to abort")
    for n in range(args.countdown, 0, -1):
        print(f"  {n}…", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r")

    tap(screen_pt.x, screen_pt.y)
    print(f"Tapped at ({screen_pt.x:.1f}, {screen_pt.y:.1f}); waiting {args.settle}s to settle…")
    time.sleep(args.settle)

    post = capture_window(window)
    post.image.save(run_dir / "post.png")

    pre_hash = imagehash.phash(pre.image)
    post_hash = imagehash.phash(post.image)
    distance = int(pre_hash - post_hash)

    verdict = "CHANGED" if distance > PHASH_SAME_THRESHOLD else "UNCHANGED"
    print(f"\nScreen phash distance: {distance}  ({verdict})")

    if verdict == "UNCHANGED":
        print(
            "\nHint: tap did not change the screen.\n"
            "  Most common cause: Accessibility permission not granted to this terminal.\n"
            "  System Settings -> Privacy & Security -> Accessibility -> add Terminal,\n"
            "  then quit and reopen Terminal before retrying.\n"
            "  Verify by manually clicking the window — if your click works but ours does not,\n"
            "  CGEvents are being silently dropped (classic missing-Accessibility symptom)."
        )

    (run_dir / "verify.json").write_text(json.dumps({
        "tap_index": args.index,
        "tap_screen_x": screen_pt.x,
        "tap_screen_y": screen_pt.y,
        "pre_phash": str(pre_hash),
        "post_phash": str(post_hash),
        "phash_distance": distance,
        "phash_threshold": PHASH_SAME_THRESHOLD,
        "verdict": verdict,
    }, indent=2))

    subprocess.run(["open", str(run_dir)], check=False)
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
