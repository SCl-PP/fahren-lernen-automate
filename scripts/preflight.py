"""Pre-run checks before the harness starts the loop.

Verifies:
  - GEMINI_API_KEY is populated in .env
  - Accessibility permission granted to this process (CGEvent taps)
  - Screen Recording permission granted (live window capture)
  - Target app window is findable (lists visible windows for debugging)

Exits 0 if everything is green; 1 with a clear error message otherwise.
"""
import os
import sys

TARGET_APP = os.environ.get("TARGET_APP", "Fahren Lernen")


def _result(label: str, ok: bool, detail: str = "") -> bool:
    tag = " OK " if ok else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok


def _check_api_key() -> bool:
    try:
        from harness.config import load_config
        cfg = load_config()
        return _result("GEMINI_API_KEY", True, f"model={cfg.model}")
    except Exception as e:
        return _result("GEMINI_API_KEY", False, str(e))


def _check_accessibility() -> bool:
    from harness.act import accessibility_granted
    if accessibility_granted():
        return _result("Accessibility permission", True, "trusted")
    return _result(
        "Accessibility permission",
        False,
        "System Settings -> Privacy & Security -> Accessibility -> add Terminal",
    )


def _list_windows() -> None:
    from harness.capture import list_windows
    print("\n  Visible windows:")
    for w in list_windows():
        if not w.owner or not w.title:
            continue
        print(
            f"    [{w.window_id:>7}] {w.owner!r} :: {w.title!r}  "
            f"{int(w.width)}x{int(w.height)} @ ({int(w.x)},{int(w.y)})"
        )
    print()


def _check_window() -> bool:
    from harness.capture import find_window
    try:
        w = find_window(app_name=TARGET_APP)
        return _result(
            f"Target window matching {TARGET_APP!r}",
            True,
            f"id={w.window_id} {int(w.width)}x{int(w.height)} @ ({int(w.x)},{int(w.y)})",
        )
    except RuntimeError:
        return _result(
            f"Target window matching {TARGET_APP!r}",
            False,
            "not found -- is the app running and visible?",
        )


def _check_screen_recording() -> bool:
    from harness.capture import capture_window, find_window
    try:
        w = find_window(app_name=TARGET_APP)
        cap = capture_window(w)
        return _result(
            "Screen Recording permission",
            True,
            f"captured {cap.image.size[0]}x{cap.image.size[1]} px",
        )
    except RuntimeError as e:
        hint = ""
        if "Screen Recording" in str(e):
            hint = " -- grant via System Settings -> Privacy & Security -> Screen Recording"
        return _result("Screen Recording permission", False, f"{e}{hint}")


def main() -> None:
    print("Preflight checks:\n")
    api_ok = _check_api_key()
    ax_ok = _check_accessibility()
    _list_windows()
    win_ok = _check_window()
    sr_ok = _check_screen_recording() if win_ok else False

    if not all([api_ok, ax_ok, win_ok, sr_ok]):
        print(
            "\nOne or more preflight checks failed. Fix the issues above "
            "and run again.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
