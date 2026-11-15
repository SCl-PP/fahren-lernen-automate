"""Unified navigation loop: capture -> triage -> dispatch -> answer -> advance.

Simulates a real user by answering correctly CORRECT_RATIO fraction of the
time (default 0.7). Logs each iteration's artifacts for crash reproduction.

Entry point: run_loop(). Env knobs (also exposed via LoopConfig):
    CORRECT_RATIO           fraction of questions answered correctly (0..1)
    MAX_ITERATIONS          -1 = unlimited (default)
    FILM_RECORD_SECONDS     window-recording seconds for film questions
    INTER_ITER_SLEEP_S      pause between iterations (user realism)
    RANDOM_SEED             optional int, for deterministic correct/wrong picks
"""
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .act import image_to_screen, tap, type_text
from .capture import Capture, WindowInfo, capture_window, find_window
from .config import Config
from .decide import GeminiClient
from .recorder import record_window
from .answer_cache import AnswerCache, CacheEntry
from .solver import (
    AnswerBox,
    BoundingBox,
    FahrschulAntwort,
    FahrschulInputAntwort,
    detect_bundle_end,
    detect_home_screen,
    extract_musterloesung,
    locate_button,
    solve_film_question,
    solve_input_film_question,
    solve_input_question,
    solve_question,
    triage_screen,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoopConfig:
    correct_ratio: float = 1.0  # always answer correctly; raise CORRECT_RATIO<1 later
    max_iterations: int = -1
    film_record_seconds: int = 30
    inter_iter_sleep_s: float = 1.5
    max_consecutive_other: int = 12  # tolerate loading/transition screens
    max_consecutive_errors: int = 3
    other_stuck_sleep_s: float = 4.0  # extra wait when 'other' can't advance
    random_seed: int | None = None


def load_loop_config() -> LoopConfig:
    seed_env = os.getenv("RANDOM_SEED")
    return LoopConfig(
        correct_ratio=float(os.getenv("CORRECT_RATIO", "1.0")),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "-1")),
        film_record_seconds=int(os.getenv("FILM_RECORD_SECONDS", "30")),
        inter_iter_sleep_s=float(os.getenv("INTER_ITER_SLEEP_S", "1.5")),
        random_seed=int(seed_env) if seed_env else None,
    )


# ---------------------------------------------------------------- strategy

def pick_mc_indices(result: FahrschulAntwort, answer_correctly: bool) -> list[int]:
    """Return 1-based indices to tap, in tap order."""
    correct_set = set(result.correct_solutions)
    all_indices = {b.index for b in result.answer_boxes}
    if answer_correctly:
        return sorted(correct_set)
    wrong_pool = sorted(all_indices - correct_set)
    if wrong_pool:
        return [random.choice(wrong_pool)]
    # All options are correct — pick a strict subset so the submit is "wrong".
    correct_sorted = sorted(correct_set)
    return correct_sorted[:-1] if len(correct_sorted) > 1 else []


def mutate_input_value(value: str) -> str:
    """Plausible-but-wrong version of a typed value."""
    if value.isdigit():
        n = int(value)
        offset = random.choice([-30, -20, -10, 10, 20, 30])
        return str(max(1, n + offset))
    if len(value) <= 1:
        return value + "x"
    chars = list(value)
    i = random.randrange(len(value) - 1)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


# ------------------------------------------------------- per-iteration logic

@dataclass
class IterationResult:
    index: int
    timestamp: str
    question_type: str = "unknown"
    chose_correct: bool | None = None
    answered: bool = False
    submitted: bool = False
    advanced: bool = False
    cache_hit: bool = False
    phash_whole: int | None = None
    notes: str = ""
    error: str | None = None


def _tap_mc_answers(
    capture: Capture,
    boxes_by_index: dict[int, AnswerBox],
    indices: list[int],
) -> None:
    W, H = capture.image.width, capture.image.height
    for i in indices:
        box = boxes_by_index[i]
        cx, cy = box.center_px(W, H)
        pt = image_to_screen(cx, cy, capture)
        tap(pt.x, pt.y)
        time.sleep(0.4)


def _type_input_answer(
    capture: Capture,
    result: FahrschulInputAntwort,
    answer_correctly: bool,
) -> str:
    value = result.answer_value if answer_correctly else mutate_input_value(result.answer_value)
    fx, fy = result.input_field.center_px(capture.image.width, capture.image.height)
    pt = image_to_screen(fx, fy, capture)
    tap(pt.x, pt.y)
    time.sleep(0.3)
    type_text(value)
    time.sleep(0.5)
    return value


def _mc_from_cache(entry: CacheEntry) -> FahrschulAntwort:
    boxes = [AnswerBox(**b) for b in entry.answer_boxes]
    return FahrschulAntwort(
        explanation=f"[cache hit, seen {entry.hit_count}x, first {entry.first_seen}]",
        correct_solutions=entry.correct_solutions,
        answer_boxes=boxes,
    )


def _input_from_cache(entry: CacheEntry) -> FahrschulInputAntwort:
    return FahrschulInputAntwort(
        explanation=f"[cache hit, seen {entry.hit_count}x, first {entry.first_seen}]",
        answer_value=entry.answer_value or "",
        input_field=BoundingBox(**(entry.input_field or {})),
    )


def _try_cache_and_extract(
    client: GeminiClient,
    cache: AnswerCache | None,
    pre_capture: Capture,
    feedback_path: Path,
    iter_dir: Path,
    question_type: str,
    original_solver_boxes: list[AnswerBox] | None = None,
    original_input_field: BoundingBox | None = None,
) -> None:
    """After a fresh (non-cache) answer was submitted, extract the Musterloesung
    from the feedback screen and persist it to the cache so later encounters
    of this question resolve zero-call."""
    if cache is None or not feedback_path.exists():
        return
    try:
        feedback_capture = Capture(
            image=Image.open(feedback_path),
            window=pre_capture.window,
        )
        ml = extract_musterloesung(client, feedback_capture)
        (iter_dir / "musterloesung.json").write_text(ml.model_dump_json(indent=2))

        if ml.question_type in ("foto", "text") and ml.correct_indices and original_solver_boxes:
            entry = CacheEntry(
                question_type=ml.question_type,
                correct_solutions=ml.correct_indices,
                answer_boxes=[b.model_dump() for b in original_solver_boxes],
                label=next((b.label for b in original_solver_boxes if b.index == 1), "")[:80],
            )
            cache.remember(pre_capture, entry)
        elif ml.question_type == "input" and ml.correct_value and original_input_field is not None:
            entry = CacheEntry(
                question_type="input",
                answer_value=ml.correct_value,
                input_field=original_input_field.model_dump(),
                label=f"{ml.correct_value}"[:80],
            )
            cache.remember(pre_capture, entry)
    except Exception as e:
        log.warning("musterloesung extraction/cache store failed: %s", e)


def _tap_button(window: WindowInfo, capture: Capture, box) -> None:
    cx, cy = box.center_px(capture.image.width, capture.image.height)
    pt = image_to_screen(cx, cy, capture)
    tap(pt.x, pt.y)


def _try_skip_recovery(
    client: GeminiClient,
    window: WindowInfo,
    iter_dir: Path,
    reason: str,
) -> tuple[bool, str]:
    """Fallback for unrecoverable iteration states: tap 'Überspringen' to skip
    the current question so the loop stays alive. Returns (tapped, notes).
    """
    try:
        current = capture_window(window)
        current.image.save(iter_dir / "skip_recovery_pre.png")
        skip = locate_button(client, current, "Überspringen")
        (iter_dir / "locate_ueberspringen.json").write_text(
            skip.model_dump_json(indent=2)
        )
        if skip.found and skip.box is not None:
            _tap_button(window, current, skip.box)
            time.sleep(1.5)
            return True, f"recovered via Überspringen ({reason})"
        return False, f"could not recover: no Überspringen visible ({reason})"
    except Exception as e:
        log.warning("skip recovery failed: %s", e)
        return False, f"skip recovery failed: {e}"


def _submit_then_advance(
    client: GeminiClient,
    window: WindowInfo,
    iter_dir: Path,
    settle_s: float = 1.0,
) -> tuple[bool, bool, str]:
    """After an answer is in, tap submit then navigate the follow-up state.

    Returns (submitted, advanced, notes). Follow-up paths:
      - mid-bundle: feedback -> 'Weiter' -> next question
      - last question in bundle: feedback -> 'Fertig' -> summary screen ->
        'Falsche Fragen üben' -> new bundle starts
      - all-correct variant: handled separately by user; we stop at summary
    """
    time.sleep(0.6)
    after = capture_window(window)
    after.image.save(iter_dir / "after_answer.png")
    submit = locate_button(client, after, "Frage überprüfen")
    (iter_dir / "locate_submit.json").write_text(submit.model_dump_json(indent=2))
    if not submit.found or submit.box is None:
        return False, False, "no 'Frage überprüfen' after answer"
    _tap_button(window, after, submit.box)
    time.sleep(settle_s)

    feedback = capture_window(window)
    feedback.image.save(iter_dir / "feedback.png")

    weiter = locate_button(client, feedback, "Weiter")
    (iter_dir / "locate_weiter.json").write_text(weiter.model_dump_json(indent=2))
    if weiter.found and weiter.box is not None:
        _tap_button(window, feedback, weiter.box)
        time.sleep(settle_s)
        return True, True, ""

    # Last question of a bundle shows 'Fertig' instead of 'Weiter'.
    fertig = locate_button(client, feedback, "Fertig")
    (iter_dir / "locate_fertig.json").write_text(fertig.model_dump_json(indent=2))
    if not fertig.found or fertig.box is None:
        return True, False, "feedback has neither 'Weiter' nor 'Fertig'"

    _tap_button(window, feedback, fertig.box)
    time.sleep(settle_s + 1.0)

    summary = capture_window(window)
    summary.image.save(iter_dir / "bundle_summary.png")

    # Detect the bundle-end summary structurally. The continue-button's label
    # varies by quiz mode ('Falsche Fragen üben', 'Lernweg fortsetzen', etc.),
    # so we classify the screen as a whole and tap whatever large bottom
    # call-to-action is present.
    bundle = detect_bundle_end(client, summary)
    (iter_dir / "bundle_end.json").write_text(bundle.model_dump_json(indent=2))
    if bundle.is_bundle_end and bundle.continue_button is not None:
        _tap_button(window, summary, bundle.continue_button)
        time.sleep(settle_s + 1.0)
        score = f" (score: {bundle.score_text})" if bundle.score_text else ""
        return True, True, f"bundle complete; restarted via continue-button{score}"

    return True, False, "bundle complete; no continue-button detected (user to handle)"


def _handle_film(
    client: GeminiClient,
    window: WindowInfo,
    pre: Capture,
    triage,
    iter_dir: Path,
    cfg: LoopConfig,
    r: IterationResult,
    cache: AnswerCache | None = None,
) -> None:
    if triage.video_playing or triage.video_start_button is None:
        r.notes = "film mid-play or missing start button"
        return
    sx, sy = triage.video_start_button.center_px(pre.image.width, pre.image.height)
    start_pt = image_to_screen(sx, sy, pre)
    tap(start_pt.x, start_pt.y)
    time.sleep(0.5)

    video_path = iter_dir / "video.mov"
    record_window(window, cfg.film_record_seconds, video_path)

    post_video = capture_window(window)
    post_video.image.save(iter_dir / "post_video.png")
    fa = locate_button(client, post_video, "Frage ansehen")
    (iter_dir / "locate_frage_ansehen.json").write_text(fa.model_dump_json(indent=2))
    if not fa.found or fa.box is None:
        r.notes = "no 'Frage ansehen' after recording"
        return
    fx, fy = fa.box.center_px(post_video.image.width, post_video.image.height)
    fa_pt = image_to_screen(fx, fy, post_video)
    tap(fa_pt.x, fa_pt.y)
    time.sleep(1.0)

    question = capture_window(window)
    question.image.save(iter_dir / "question.png")

    q_triage = triage_screen(client, question)
    (iter_dir / "question_triage.json").write_text(q_triage.model_dump_json(indent=2))

    # Cache lookup on the POST-"Frage ansehen" question screen -- saves the
    # expensive 2-pass film solve (video upload + 2 Gemini calls) on repeats.
    cached = cache.lookup(question) if cache else None

    if q_triage.question_type == "input":
        if cached and cached.question_type == "input" and cached.answer_value:
            ans = _input_from_cache(cached)
            r.cache_hit = True
            (iter_dir / "cache_hit.json").write_text(json.dumps(asdict(cached), indent=2))
        else:
            ans = solve_input_film_question(client, question, video_path)
            (iter_dir / "solver.json").write_text(ans.model_dump_json(indent=2))
        _type_input_answer(question, ans, r.chose_correct)
        r.answered = True
    elif q_triage.question_type in ("foto", "text"):
        if (
            cached
            and cached.question_type in ("foto", "text")
            and cached.answer_boxes
        ):
            ans_mc = _mc_from_cache(cached)
            r.cache_hit = True
            (iter_dir / "cache_hit.json").write_text(json.dumps(asdict(cached), indent=2))
        else:
            ans_mc = solve_film_question(client, question, video_path)
            (iter_dir / "solver.json").write_text(ans_mc.model_dump_json(indent=2))
        boxes = {b.index: b for b in ans_mc.answer_boxes}
        if not boxes:
            r.notes = "film MC solver returned no answer boxes"
            return
        indices = pick_mc_indices(ans_mc, r.chose_correct)
        _tap_mc_answers(question, boxes, indices)
        r.answered = True
    else:
        r.notes = f"unexpected post-video type {q_triage.question_type}"
        return

    submitted, advanced, notes = _submit_then_advance(client, window, iter_dir)
    r.submitted, r.advanced = submitted, advanced
    if notes:
        r.notes = notes

    if not r.cache_hit and r.submitted:
        if q_triage.question_type == "input":
            _try_cache_and_extract(
                client, cache, question, iter_dir / "feedback.png", iter_dir,
                "input", original_input_field=ans.input_field,
            )
        elif q_triage.question_type in ("foto", "text"):
            _try_cache_and_extract(
                client, cache, question, iter_dir / "feedback.png", iter_dir,
                q_triage.question_type, original_solver_boxes=ans_mc.answer_boxes,
            )

    if not r.advanced:
        recovered, note = _try_skip_recovery(client, window, iter_dir, notes or "no advance (film)")
        if recovered:
            r.advanced = True
        r.notes = f"{r.notes + '; ' if r.notes else ''}{note}"


def _run_iteration(
    client: GeminiClient,
    window: WindowInfo,
    iter_dir: Path,
    cfg: LoopConfig,
    idx: int,
    cache: AnswerCache | None = None,
) -> IterationResult:
    iter_dir.mkdir(parents=True, exist_ok=True)
    r = IterationResult(index=idx, timestamp=datetime.now().isoformat())
    try:
        pre = capture_window(window)
        pre.image.save(iter_dir / "pre.png")

        triage = triage_screen(client, pre)
        (iter_dir / "triage.json").write_text(triage.model_dump_json(indent=2))
        r.question_type = triage.question_type

        if triage.question_type == "other":
            # 1. Bundle-end summary (structural detection works for any CTA label).
            bundle = detect_bundle_end(client, pre)
            (iter_dir / "bundle_end.json").write_text(bundle.model_dump_json(indent=2))
            if bundle.is_bundle_end and bundle.continue_button is not None:
                _tap_button(window, pre, bundle.continue_button)
                time.sleep(2.0)
                r.advanced = True
                score = f" (score: {bundle.score_text})" if bundle.score_text else ""
                r.notes = f"'other' advanced via bundle-end continue-button{score}"
                return r

            # 2. Vorprüfung modal dialog -- ALWAYS dismiss via 'Nicht jetzt'.
            #    Never let the harness start a Vorprüfung (user-enforced policy).
            nicht_jetzt = locate_button(client, pre, "Nicht jetzt")
            (iter_dir / "locate_nicht_jetzt.json").write_text(
                nicht_jetzt.model_dump_json(indent=2)
            )
            if nicht_jetzt.found and nicht_jetzt.box is not None:
                _tap_button(window, pre, nicht_jetzt.box)
                time.sleep(1.5)
                r.advanced = True
                r.notes = "dismissed Vorprüfung dialog via 'Nicht jetzt'"
                return r

            # 3. Generic 'Weiter' advance.
            weiter = locate_button(client, pre, "Weiter")
            (iter_dir / "locate_weiter.json").write_text(weiter.model_dump_json(indent=2))
            if weiter.found and weiter.box is not None:
                _tap_button(window, pre, weiter.box)
                time.sleep(1.0)
                r.advanced = True
                r.notes = "'other' advanced via Weiter"
                return r

            # 4. Home screen -- re-enter learning by tapping the topmost active
            #    Lerneinheit card (structural: works for whichever unit is current).
            home = detect_home_screen(client, pre)
            (iter_dir / "home_screen.json").write_text(home.model_dump_json(indent=2))
            if home.is_home_screen and home.top_active_card is not None:
                _tap_button(window, pre, home.top_active_card)
                time.sleep(2.0)
                r.advanced = True
                label = home.card_label or "top Lerneinheit"
                r.notes = f"entered {label} from home screen"
                return r

            r.notes = "'other' with no handler match (likely transient loading)"
            return r

        r.chose_correct = random.random() < cfg.correct_ratio

        if triage.question_type == "film":
            _handle_film(client, window, pre, triage, iter_dir, cfg, r, cache)
            if not r.advanced:
                recovered, note = _try_skip_recovery(
                    client, window, iter_dir, r.notes or "film path did not advance"
                )
                if recovered:
                    r.advanced = True
                r.notes = f"{r.notes + '; ' if r.notes else ''}{note}"
            return r

        if triage.question_type == "input":
            cached = cache.lookup(pre) if cache else None
            if cached and cached.question_type == "input" and cached.answer_value:
                ans = _input_from_cache(cached)
                r.cache_hit = True
                (iter_dir / "cache_hit.json").write_text(json.dumps(asdict(cached), indent=2))
            else:
                ans = solve_input_question(client, pre)
                (iter_dir / "solver.json").write_text(ans.model_dump_json(indent=2))
            _type_input_answer(pre, ans, r.chose_correct)
            r.answered = True
            r.submitted, r.advanced, notes = _submit_then_advance(client, window, iter_dir)
            if notes:
                r.notes = notes
            if not r.cache_hit and r.submitted:
                _try_cache_and_extract(
                    client, cache, pre, iter_dir / "feedback.png", iter_dir,
                    "input", original_input_field=ans.input_field,
                )
            if not r.advanced:
                recovered, note = _try_skip_recovery(client, window, iter_dir, notes or "no advance")
                if recovered:
                    r.advanced = True
                r.notes = f"{r.notes + '; ' if r.notes else ''}{note}"
            return r

        if triage.question_type in ("foto", "text"):
            cached = cache.lookup(pre) if cache else None
            if (
                cached
                and cached.question_type in ("foto", "text")
                and cached.answer_boxes
            ):
                ans_mc = _mc_from_cache(cached)
                r.cache_hit = True
                (iter_dir / "cache_hit.json").write_text(json.dumps(asdict(cached), indent=2))
            else:
                ans_mc = solve_question(client, pre)
                (iter_dir / "solver.json").write_text(ans_mc.model_dump_json(indent=2))
            boxes = {b.index: b for b in ans_mc.answer_boxes}
            if not boxes:
                r.notes = "MC solver returned no answer boxes"
                return r
            indices = pick_mc_indices(ans_mc, r.chose_correct)
            _tap_mc_answers(pre, boxes, indices)
            r.answered = True
            r.submitted, r.advanced, notes = _submit_then_advance(client, window, iter_dir)
            if notes:
                r.notes = notes
            if not r.cache_hit and r.submitted:
                _try_cache_and_extract(
                    client, cache, pre, iter_dir / "feedback.png", iter_dir,
                    triage.question_type, original_solver_boxes=ans_mc.answer_boxes,
                )
            if not r.advanced:
                recovered, note = _try_skip_recovery(client, window, iter_dir, notes or "no advance")
                if recovered:
                    r.advanced = True
                r.notes = f"{r.notes + '; ' if r.notes else ''}{note}"
            return r

    except Exception as e:
        log.exception("iteration %d failed", idx)
        r.error = str(e)
        recovered, note = _try_skip_recovery(client, window, iter_dir, f"error: {e}")
        if recovered:
            r.advanced = True
        r.notes = f"{r.notes + '; ' if r.notes else ''}{note}"
    return r


# -------------------------------------------------------------------- loop

def run_loop(
    config: Config,
    loop_cfg: LoopConfig,
    *,
    app_name: str | None = None,
    window_title: str | None = None,
    session_dir: Path | None = None,
) -> Path:
    if loop_cfg.random_seed is not None:
        random.seed(loop_cfg.random_seed)

    if session_dir is None:
        session_dir = Path("runs") / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = session_dir / "session.jsonl"
    log.info("session dir: %s", session_dir)
    log.info("config: %s", loop_cfg)

    client = GeminiClient(config)
    cache = AnswerCache()
    log.info("answer cache: %d entries at start", len(cache))
    idx = 0
    consecutive_other = 0
    consecutive_errors = 0
    stats = {"submitted": 0, "cache_hits": 0, "errors": 0, "by_type": {}}

    try:
        while loop_cfg.max_iterations < 0 or idx < loop_cfg.max_iterations:
            idx += 1
            try:
                window = find_window(app_name=app_name, window_title=window_title)
            except RuntimeError as e:
                log.error("window gone at iter %d: %s", idx, e)
                break

            iter_dir = session_dir / f"iter-{idx:04d}"
            result = _run_iteration(client, window, iter_dir, loop_cfg, idx, cache)

            with jsonl_path.open("a") as f:
                f.write(json.dumps(asdict(result)) + "\n")

            if result.submitted:
                stats["submitted"] += 1
            if result.cache_hit:
                stats["cache_hits"] += 1
            if result.error:
                stats["errors"] += 1
            stats["by_type"][result.question_type] = stats["by_type"].get(result.question_type, 0) + 1

            log.info(
                "iter %d: type=%s correct=%s cache=%s answered=%s submitted=%s advanced=%s%s%s",
                idx,
                result.question_type,
                result.chose_correct,
                result.cache_hit,
                result.answered,
                result.submitted,
                result.advanced,
                f" notes={result.notes!r}" if result.notes else "",
                f" error={result.error}" if result.error else "",
            )

            if idx % 10 == 0:
                by_type = " ".join(f"{k}={v}" for k, v in sorted(stats["by_type"].items()))
                log.info(
                    "---- progress @ iter %d: submitted=%d cache_hits=%d errors=%d cache_size=%d  %s",
                    idx, stats["submitted"], stats["cache_hits"], stats["errors"],
                    len(cache), by_type,
                )

            # Only count errors that we could NOT recover from. If the loop
            # skipped the question via Überspringen and advanced, we're still
            # making progress and shouldn't trip the abort cap.
            if result.error and not result.advanced:
                consecutive_errors += 1
                if consecutive_errors >= loop_cfg.max_consecutive_errors:
                    log.error("%d consecutive unrecovered errors; aborting", consecutive_errors)
                    break
            else:
                consecutive_errors = 0

            if result.question_type == "other" and not result.advanced:
                consecutive_other += 1
                if consecutive_other >= loop_cfg.max_consecutive_other:
                    log.error("stuck on 'other' screens; aborting")
                    break
            else:
                consecutive_other = 0

            time.sleep(loop_cfg.inter_iter_sleep_s)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — exiting loop at iter %d", idx)

    log.info(
        "session ended after %d iterations; cache=%d entries; artifacts under %s",
        idx, len(cache), session_dir,
    )
    return session_dir
