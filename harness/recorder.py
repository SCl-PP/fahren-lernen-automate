"""Window-targeted screen recording via the macOS `screencapture` CLI.

No ffmpeg dependency; uses the stock system tool. Blocks for `duration_s`
seconds and writes a .mov to the given path.
"""
import logging
import subprocess
from pathlib import Path

from .capture import WindowInfo

log = logging.getLogger(__name__)


def record_window(window: WindowInfo, duration_s: int, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "screencapture",
        "-v",  # video
        "-V", str(duration_s),  # max duration in seconds
        "-x",  # no shutter sound
        "-l", str(window.window_id),  # target window
        str(out_path),
    ]
    log.info("recording %ds -> %s", duration_s, out_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"screencapture failed (exit {result.returncode}): "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"screencapture produced no output at {out_path}")
    log.info("recorded %.1f MB", out_path.stat().st_size / 1e6)
    return out_path
