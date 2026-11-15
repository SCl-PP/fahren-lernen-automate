"""Input-field question: triage -> solve -> tap field -> type value -> submit.

Works for standalone input questions (no video). For film+input, see the
film flow which dispatches on question_type after the 'Frage ansehen' tap.

Usage:
    uv run python scripts/live_tap_input.py
    uv run python scripts/live_tap_input.py --no-answer    # dry-run
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import imagehash
from PIL import ImageDraw, ImageFont

from harness.act import accessibility_granted, image_to_screen, tap, type_text
from harness.capture import capture_window, find_window
from harness.config import load_config
from harness.decide import GeminiClient
from harness.solver import locate_button, solve_input_question

BLUE = (40, 90, 220, 255)
RED = (220, 40, 40, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

WHOLE_PHASH_THRESHOLD = 4
REGION_PHASH_THRESHOLD = 6
REGION_SIZE_PX = 160


def _font(size: int):
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
    W, H = image.width, image.height
    box = result.input_field
    x0 = int(box.xmin / 1000 * W)
    y0 = int(box.ymin / 1000 * H)
    x1 = int(box.xmax / 1000 * W)
    y1 = int(box.ymax / 1000 * H)
    cx, cy = box.center_px(W, H)
    draw.rectangle((x0, y0, x1, y1), outline=BLUE, width=3)
    draw.line((cx - 6, cy, cx + 6, cy), fill=RED, width=1)
    draw.line((cx, cy - 6, cx, cy + 6), fill=RED, width=1)
    label = f"TYPE: {result.answer_value}"
    tx, ty = x1 + 6, y0
    bg = draw.textbbox((tx, ty), label, font=font)
    draw.rectangle((bg[0] - 3, bg[1] - 1, bg[2] + 3, bg[3] + 1), fill=BLACK)
    draw.text((tx, ty), label, fill=WHITE, font=font)
    return canvas


def _countdown(label: str, seconds: int) -> None:
    print(f"\n{label} in {seconds}s — Ctrl+C to abort")
    for n in range(seconds, 0, -1):
        print(f"  {n}…", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="Fahren Lernen")
    p.add_argument("--title")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--settle", type=float, default=1.0)
    p.add_argument("--no-answer", action="store_true",
                   help="Stop after solving; do not type or submit")
    args = p.parse_args()

    print(f"Accessibility trusted: {accessibility_granted()}")
    window = find_window(app_name=args.app, window_title=args.title)
    print(f"Window: {window.owner!r} id={window.window_id} "
          f"{int(window.width)}x{int(window.height)} @ ({int(window.x)},{int(window.y)})")

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-input")
    run_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(load_config())

    pre = capture_window(window)
    pre.image.save(run_dir / "pre.png")

    result = solve_input_question(client, pre)
    (run_dir / "solver_response.json").write_text(result.model_dump_json(indent=2))

    print(f"\n--- EXPLANATION ---\n{result.explanation}")
    print(f"\n--- ANSWER TO TYPE: {result.answer_value!r} ---")
    W, H = pre.image.width, pre.image.height
    fx_img, fy_img = result.input_field.center_px(W, H)
    print(f"Input field: img=({fx_img},{fy_img})  "
          f"bbox[ymin,xmin,ymax,xmax]=[{result.input_field.ymin},"
          f"{result.input_field.xmin},{result.input_field.ymax},{result.input_field.xmax}]")

    _annotate(pre.image, result).save(run_dir / "annotated.png")

    if args.no_answer:
        subprocess.run(["open", str(run_dir)], check=False)
        print(f"\n--no-answer set; stopping. Artifacts: {run_dir}")
        return

    field_pt = image_to_screen(fx_img, fy_img, pre)
    print(f"\nTap field: screen=({field_pt.x:.1f},{field_pt.y:.1f})")
    _countdown("Tap field + type + submit", args.countdown)
    tap(field_pt.x, field_pt.y)
    time.sleep(0.3)  # let focus register
    type_text(result.answer_value)
    time.sleep(0.5)

    after_type = capture_window(window)
    after_type.image.save(run_dir / "after_type.png")

    print("\nLocating 'Frage überprüfen' submit button…")
    submit = locate_button(client, after_type, "Frage überprüfen")
    (run_dir / "locate_submit.json").write_text(submit.model_dump_json(indent=2))
    print(f"  found={submit.found}  reasoning: {submit.reasoning}")

    if submit.found and submit.box is not None:
        sx_img, sy_img = submit.box.center_px(
            after_type.image.width, after_type.image.height
        )
        submit_pt = image_to_screen(sx_img, sy_img, after_type)
        _countdown("Tap Frage überprüfen", args.countdown)
        tap(submit_pt.x, submit_pt.y)
        time.sleep(args.settle)
    else:
        print("  No submit button found; app may auto-advance or require Enter.")

    post = capture_window(window)
    post.image.save(run_dir / "post.png")

    whole_distance = int(imagehash.phash(pre.image) - imagehash.phash(post.image))
    half = REGION_SIZE_PX // 2
    box = (
        max(0, fx_img - half),
        max(0, fy_img - half),
        min(pre.image.width, fx_img + half),
        min(pre.image.height, fy_img + half),
    )
    pre_region = pre.image.crop(box)
    post_region = post.image.crop(box)
    pre_region.save(run_dir / "pre_region.png")
    post_region.save(run_dir / "post_region.png")
    region_distance = int(imagehash.phash(pre_region) - imagehash.phash(post_region))

    whole_changed = whole_distance > WHOLE_PHASH_THRESHOLD
    region_changed = region_distance > REGION_PHASH_THRESHOLD
    verdict = "CHANGED" if (whole_changed or region_changed) else "UNCHANGED"
    print(
        f"\nPhash whole={whole_distance} (>{WHOLE_PHASH_THRESHOLD}? {whole_changed})  "
        f"region={region_distance} (>{REGION_PHASH_THRESHOLD}? {region_changed})  "
        f"=> {verdict}"
    )

    (run_dir / "verify.json").write_text(json.dumps({
        "typed_value": result.answer_value,
        "field_screen_x": field_pt.x,
        "field_screen_y": field_pt.y,
        "field_img_x": fx_img,
        "field_img_y": fy_img,
        "phash_whole": whole_distance,
        "phash_region": region_distance,
        "verdict": verdict,
    }, indent=2))

    subprocess.run(["open", str(run_dir)], check=False)
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
