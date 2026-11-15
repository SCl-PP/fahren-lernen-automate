"""Step 3: film-question handling end-to-end.

Flow:
  1. Triage current screen -> expect question_type='film', video_playing=False
  2. Tap the predicted 'Film starten' / play button
  3. Record the window for `--duration` seconds while the video plays
  4. Capture a final frame after recording stops
  5. Upload the recording, solve with video + final frame
  6. Tap the first correct answer
  7. phash-verify the screen changed after the tap

All artifacts (pre, final_frame, post, video.mov, triage.json, solver.json,
verify.json, annotated.png) are saved to runs/<timestamp>/.

Usage:
    uv run python scripts/live_tap_film.py
    uv run python scripts/live_tap_film.py --duration 25
    uv run python scripts/live_tap_film.py --no-answer   # stop before tapping answer
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
from harness.recorder import record_window
from harness.solver import (
    locate_button,
    solve_film_question,
    solve_input_film_question,
    triage_screen,
)

RED = (220, 40, 40, 255)
GREEN = (40, 200, 60, 255)
BLUE = (40, 90, 220, 255)
ORANGE = (240, 150, 40, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

WHOLE_PHASH_THRESHOLD = 4
REGION_PHASH_THRESHOLD = 6  # a filled vs empty checkbox is huge in a small crop
REGION_SIZE_PX = 120  # crop side length centered on tap target


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_answers(image, result, tap_index: int):
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
            color = BLUE
        elif box.index in correct:
            color = GREEN
        else:
            color = RED
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.line((cx - 6, cy, cx + 6, cy), fill=RED, width=1)
        draw.line((cx, cy - 6, cx, cy + 6), fill=RED, width=1)

        label = f"{box.index}{'*' if box.index == tap_index else ''}"
        tx, ty = x1 + 6, y0
        bg = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle((bg[0] - 3, bg[1] - 1, bg[2] + 3, bg[3] + 1), fill=BLACK)
        draw.text((tx, ty), label, fill=WHITE, font=font)
    return canvas


def _annotate_input(image, result):
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


def _annotate_start(image, triage):
    canvas = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas)
    W, H = image.width, image.height
    box = triage.video_start_button
    if box is None:
        return canvas
    x0 = int(box.xmin / 1000 * W)
    y0 = int(box.ymin / 1000 * H)
    x1 = int(box.xmax / 1000 * W)
    y1 = int(box.ymax / 1000 * H)
    draw.rectangle((x0, y0, x1, y1), outline=ORANGE, width=3)
    return canvas


def _countdown(label: str, seconds: int) -> None:
    print(f"\n{label} in {seconds}s — Ctrl+C to abort")
    for n in range(seconds, 0, -1):
        print(f"  {n}…", end="\r", flush=True)
        time.sleep(1)
    print(" " * 20, end="\r")


def _run_film_input(client, window, final, video_path, run_dir, args) -> None:
    print("\nSolving film question (input, uploading video)…")
    result = solve_input_film_question(client, final, video_path)
    (run_dir / "solver_response.json").write_text(result.model_dump_json(indent=2))

    W, H = final.image.width, final.image.height
    fx_img, fy_img = result.input_field.center_px(W, H)
    print(f"\n--- EXPLANATION ---\n{result.explanation}")
    print(f"\n--- ANSWER TO TYPE: {result.answer_value!r} ---")
    print(f"Input field: img=({fx_img},{fy_img})")

    _annotate_input(final.image, result).save(run_dir / "annotated.png")

    if args.no_answer:
        print(f"\n--no-answer set; stopping. Artifacts: {run_dir}")
        subprocess.run(["open", str(run_dir)], check=False)
        return

    field_pt = image_to_screen(fx_img, fy_img, final)
    _countdown("Tap field + type + submit", args.countdown)
    tap(field_pt.x, field_pt.y)
    time.sleep(0.3)
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
        print("  No submit button found; app may auto-advance.")

    post = capture_window(window)
    post.image.save(run_dir / "post.png")

    whole_distance = int(imagehash.phash(final.image) - imagehash.phash(post.image))
    half = REGION_SIZE_PX // 2
    box = (
        max(0, fx_img - half),
        max(0, fy_img - half),
        min(final.image.width, fx_img + half),
        min(final.image.height, fy_img + half),
    )
    region_distance = int(
        imagehash.phash(final.image.crop(box)) - imagehash.phash(post.image.crop(box))
    )
    whole_changed = whole_distance > WHOLE_PHASH_THRESHOLD
    region_changed = region_distance > REGION_PHASH_THRESHOLD
    verdict = "CHANGED" if (whole_changed or region_changed) else "UNCHANGED"
    print(
        f"\nPhash whole={whole_distance} (>{WHOLE_PHASH_THRESHOLD}? {whole_changed})  "
        f"region={region_distance} (>{REGION_PHASH_THRESHOLD}? {region_changed})  "
        f"=> {verdict}"
    )
    (run_dir / "verify.json").write_text(json.dumps({
        "flow": "film_input",
        "typed_value": result.answer_value,
        "field_img_x": fx_img,
        "field_img_y": fy_img,
        "field_screen_x": field_pt.x,
        "field_screen_y": field_pt.y,
        "phash_whole": whole_distance,
        "phash_region": region_distance,
        "verdict": verdict,
    }, indent=2))
    subprocess.run(["open", str(run_dir)], check=False)
    print(f"\nArtifacts: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="Fahren Lernen")
    p.add_argument("--title")
    p.add_argument("--duration", type=int, default=30, help="Video recording seconds")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--settle", type=float, default=1.0, help="Wait after answer tap")
    p.add_argument("--no-answer", action="store_true",
                   help="Stop after solving; do not tap an answer")
    args = p.parse_args()

    print(f"Accessibility trusted: {accessibility_granted()}")
    window = find_window(app_name=args.app, window_title=args.title)
    print(f"Window: {window.owner!r} id={window.window_id} "
          f"{int(window.width)}x{int(window.height)} @ ({int(window.x)},{int(window.y)})")

    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-film")
    run_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiClient(load_config())

    # --- phase 1: triage -----------------------------------------------------
    pre = capture_window(window)
    pre.image.save(run_dir / "pre.png")
    triage = triage_screen(client, pre)
    (run_dir / "triage.json").write_text(triage.model_dump_json(indent=2))
    print(f"\nTriage: type={triage.question_type}  video_playing={triage.video_playing}")
    print(f"  reasoning: {triage.reasoning}")

    if triage.question_type != "film":
        print(f"Not a film question (type={triage.question_type}). Aborting.")
        raise SystemExit(1)
    if triage.video_playing:
        print("Video is already playing. Aborting -- please reload the question.")
        raise SystemExit(1)
    if triage.video_start_button is None:
        print("No start button detected but screen classified as film. Aborting.")
        raise SystemExit(1)

    _annotate_start(pre.image, triage).save(run_dir / "pre_annotated.png")

    # --- phase 2: tap start --------------------------------------------------
    W, H = pre.image.width, pre.image.height
    sx_img, sy_img = triage.video_start_button.center_px(W, H)
    start_pt = image_to_screen(sx_img, sy_img, pre)
    print(f"Start button: img=({sx_img},{sy_img})  screen=({start_pt.x:.1f},{start_pt.y:.1f})")
    _countdown("Tapping Film starten", args.countdown)
    tap(start_pt.x, start_pt.y)
    # Give the UI a moment to begin playback before the recorder starts.
    time.sleep(0.5)

    # --- phase 3: record -----------------------------------------------------
    video_path = run_dir / "video.mov"
    print(f"Recording {args.duration}s to {video_path} …")
    record_window(window, args.duration, video_path)

    # --- phase 4: advance to the question screen ---------------------------
    # After the video, the app shows a "Frage ansehen" button. Tapping it
    # reveals the checkbox question. Capture the post-video screen, locate
    # the button, tap it, then capture the actual question screen.
    post_video = capture_window(window)
    post_video.image.save(run_dir / "post_video.png")

    print("\nLocating 'Frage ansehen' button on post-video screen…")
    fa = locate_button(client, post_video, "Frage ansehen")
    (run_dir / "locate_frage_ansehen.json").write_text(fa.model_dump_json(indent=2))
    print(f"  found={fa.found}  reasoning: {fa.reasoning}")
    if not fa.found or fa.box is None:
        print(
            "Did not find 'Frage ansehen'. Likely causes:\n"
            "  - Video did not actually play (start tap missed)\n"
            "  - Recording duration too short; app still mid-video\n"
            "  - Recording duration too long; app auto-advanced or reset\n"
            f"Artifacts: {run_dir}"
        )
        subprocess.run(["open", str(run_dir)], check=False)
        raise SystemExit(2)

    fx_img, fy_img = fa.box.center_px(post_video.image.width, post_video.image.height)
    fa_pt = image_to_screen(fx_img, fy_img, post_video)
    print(f"Frage-ansehen: img=({fx_img},{fy_img})  screen=({fa_pt.x:.1f},{fa_pt.y:.1f})")
    _countdown("Tapping Frage ansehen", args.countdown)
    tap(fa_pt.x, fa_pt.y)
    time.sleep(1.0)  # let the question screen render

    final = capture_window(window)
    final.image.save(run_dir / "final_frame.png")

    # --- phase 4b: triage question screen to pick the right solver ----------
    q_triage = triage_screen(client, final)
    (run_dir / "triage_final.json").write_text(q_triage.model_dump_json(indent=2))
    print(f"\nFinal-frame triage: type={q_triage.question_type}  "
          f"reasoning: {q_triage.reasoning}")

    if q_triage.question_type == "input":
        _run_film_input(
            client, window, final, video_path, run_dir, args,
        )
        return
    if q_triage.question_type not in ("foto", "text"):
        print(
            f"Unexpected post-video question type: {q_triage.question_type}. Aborting."
        )
        subprocess.run(["open", str(run_dir)], check=False)
        raise SystemExit(2)

    # --- phase 5a: multiple-choice flow ------------------------------------
    print("\nSolving film question (multiple-choice, uploading video)…")
    result = solve_film_question(client, final, video_path)
    (run_dir / "solver_response.json").write_text(result.model_dump_json(indent=2))

    print(f"\n--- EXPLANATION ---\n{result.explanation}")
    print(f"\n--- CORRECT INDICES: {result.correct_solutions} ---")
    for box in result.answer_boxes:
        mark = "X" if box.index in result.correct_solutions else " "
        cx, cy = box.center_px(final.image.width, final.image.height)
        print(f" [{mark}] {box.index}  center=({cx:>4},{cy:>4})  {box.label}")

    if not result.answer_boxes:
        print("\nNo answer boxes returned. Final frame may not yet show the question UI.")
        print("Try a longer --duration.")
        raise SystemExit(1)

    tap_index = result.correct_solutions[0] if result.correct_solutions else 1
    _annotate_answers(final.image, result, tap_index).save(run_dir / "annotated.png")

    if args.no_answer:
        print(f"\n--no-answer set; stopping. Artifacts: {run_dir}")
        subprocess.run(["open", str(run_dir)], check=False)
        return

    target = next(b for b in result.answer_boxes if b.index == tap_index)
    tx_img, ty_img = target.center_px(final.image.width, final.image.height)
    answer_pt = image_to_screen(tx_img, ty_img, final)
    print(f"\nTap answer #{tap_index}: img=({tx_img},{ty_img})  "
          f"screen=({answer_pt.x:.1f},{answer_pt.y:.1f})")
    _countdown(f"Tapping answer {tap_index}", args.countdown)
    tap(answer_pt.x, answer_pt.y)
    time.sleep(args.settle)

    post = capture_window(window)
    post.image.save(run_dir / "post.png")

    whole_distance = int(imagehash.phash(final.image) - imagehash.phash(post.image))

    # Crop a small region around the predicted tap target. The static video
    # thumbnail dominates whole-window phash for film questions, so a local
    # region check is a much stronger signal for checkbox state changes.
    half = REGION_SIZE_PX // 2
    box = (
        max(0, tx_img - half),
        max(0, ty_img - half),
        min(final.image.width, tx_img + half),
        min(final.image.height, ty_img + half),
    )
    pre_region = final.image.crop(box)
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
        "tap_index": tap_index,
        "tap_screen_x": answer_pt.x,
        "tap_screen_y": answer_pt.y,
        "tap_img_x": tx_img,
        "tap_img_y": ty_img,
        "phash_whole": whole_distance,
        "phash_whole_threshold": WHOLE_PHASH_THRESHOLD,
        "phash_region": region_distance,
        "phash_region_threshold": REGION_PHASH_THRESHOLD,
        "verdict": verdict,
    }, indent=2))

    subprocess.run(["open", str(run_dir)], check=False)
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
