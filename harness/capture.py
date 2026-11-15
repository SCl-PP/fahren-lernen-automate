"""Window discovery and screenshot capture for iOS-on-Mac app windows.

Requires Screen Recording permission for the process running this code
(System Settings → Privacy & Security → Screen Recording).
"""
from dataclasses import dataclass

from PIL import Image
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectNull,
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListExcludeDesktopElements,
    kCGWindowListOptionIncludingWindow,
    kCGWindowListOptionOnScreenOnly,
)


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    owner: str
    title: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Capture:
    image: Image.Image
    window: WindowInfo

    @property
    def scale_x(self) -> float:
        return self.image.width / self.window.width

    @property
    def scale_y(self) -> float:
        return self.image.height / self.window.height


def list_windows() -> list[WindowInfo]:
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    raw = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    out: list[WindowInfo] = []
    for w in raw:
        bounds = w.get("kCGWindowBounds", {})
        out.append(
            WindowInfo(
                window_id=int(w.get("kCGWindowNumber", 0)),
                owner=str(w.get("kCGWindowOwnerName", "")),
                title=str(w.get("kCGWindowName", "")),
                x=float(bounds.get("X", 0)),
                y=float(bounds.get("Y", 0)),
                width=float(bounds.get("Width", 0)),
                height=float(bounds.get("Height", 0)),
            )
        )
    return out


def find_window(
    app_name: str | None = None,
    window_title: str | None = None,
) -> WindowInfo:
    if app_name is None and window_title is None:
        raise ValueError("Provide at least one of app_name or window_title")

    candidates: list[WindowInfo] = []
    for w in list_windows():
        if w.width < 100 or w.height < 100:
            continue
        if app_name and app_name.lower() not in w.owner.lower():
            continue
        if window_title and window_title.lower() not in w.title.lower():
            continue
        candidates.append(w)

    if not candidates:
        raise RuntimeError(
            f"No window matched app={app_name!r} title={window_title!r}. "
            "Run `python -m harness.capture` to list visible windows."
        )

    candidates.sort(key=lambda w: w.width * w.height, reverse=True)
    return candidates[0]


def capture_window(window: WindowInfo) -> Capture:
    cg_image = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window.window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )
    if cg_image is None:
        raise RuntimeError(
            f"CGWindowListCreateImage returned None for window {window.window_id}. "
            "Is Screen Recording permission granted to this process?"
        )

    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    bytes_per_row = CGImageGetBytesPerRow(cg_image)
    raw = CGDataProviderCopyData(CGImageGetDataProvider(cg_image))
    image = Image.frombytes(
        "RGBA", (width, height), bytes(raw), "raw", "BGRA", bytes_per_row
    )
    return Capture(image=image, window=window)


def _main() -> None:
    for w in list_windows():
        if not w.owner:
            continue
        print(
            f"[{w.window_id:>7}] {w.owner!r} :: {w.title!r}  "
            f"{int(w.width)}x{int(w.height)} @ ({int(w.x)},{int(w.y)})"
        )


if __name__ == "__main__":
    _main()
