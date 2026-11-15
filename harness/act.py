"""Synthetic mouse / keyboard input via CoreGraphics event taps.

Requires Accessibility permission for the process running this code
(System Settings → Privacy & Security → Accessibility).

Coordinates here are in global screen points (not pixels). Use
`image_to_screen` to translate model-returned pixel coordinates from a
captured image back into screen points.
"""
import time
from dataclasses import dataclass

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventCreateMouseEvent,
    CGEventCreateScrollWheelEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    CGWarpMouseCursorPosition,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGMouseButtonLeft,
    kCGScrollEventUnitPixel,
    kCGSessionEventTap,
)

# Session-level tap: events traverse the same pipeline as real user input.
# HID-level is filtered by some Catalyst/iOS-on-Mac apps.
_TAP_LOCATION = kCGSessionEventTap

from .capture import Capture


def accessibility_granted() -> bool:
    """True if this process is Accessibility-trusted. Required for CGEvent taps."""
    try:
        from ApplicationServices import AXIsProcessTrusted
    except ImportError:
        return True  # fail open if we can't introspect
    return bool(AXIsProcessTrusted())


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def image_to_screen(img_x: float, img_y: float, capture: Capture) -> Point:
    """Translate a pixel coordinate in a captured image to a screen-point coordinate."""
    return Point(
        x=capture.window.x + img_x / capture.scale_x,
        y=capture.window.y + img_y / capture.scale_y,
    )


def tap(x: float, y: float, hold_ms: int = 80) -> None:
    pt = (x, y)
    # Hard-warp the cursor first so the OS registers the pointer at the target
    # before the click events arrive. Catalyst apps can ignore clicks whose
    # location doesn't match the current cursor.
    CGWarpMouseCursorPosition(pt)
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventMouseMoved, pt, kCGMouseButtonLeft),
    )
    time.sleep(0.02)
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, pt, kCGMouseButtonLeft),
    )
    time.sleep(hold_ms / 1000)
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, pt, kCGMouseButtonLeft),
    )


def drag(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    steps: int = 20,
    duration_ms: int = 300,
) -> None:
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x1, y1), kCGMouseButtonLeft),
    )
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x1, y1), kCGMouseButtonLeft),
    )
    step_sleep = (duration_ms / 1000) / max(steps, 1)
    for i in range(1, steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        CGEventPost(
            _TAP_LOCATION,
            CGEventCreateMouseEvent(None, kCGEventLeftMouseDragged, (x, y), kCGMouseButtonLeft),
        )
        time.sleep(step_sleep)
    CGEventPost(
        _TAP_LOCATION,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x2, y2), kCGMouseButtonLeft),
    )


def scroll(dx: int, dy: int) -> None:
    """Pixel-unit scroll at the current pointer location.

    Positive dy scrolls content down (wheel up); negative scrolls content up.
    """
    ev = CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 2, dy, dx)
    CGEventPost(_TAP_LOCATION, ev)


def type_text(text: str, per_char_delay_ms: int = 15) -> None:
    """Type a Unicode string via synthetic keyboard events.

    Uses CGEventKeyboardSetUnicodeString so we don't need a per-key keycode map,
    which is important for iOS apps that may have non-standard text fields.
    """
    for ch in text:
        down = CGEventCreateKeyboardEvent(None, 0, True)
        CGEventKeyboardSetUnicodeString(down, len(ch), ch)
        CGEventPost(_TAP_LOCATION, down)
        up = CGEventCreateKeyboardEvent(None, 0, False)
        CGEventKeyboardSetUnicodeString(up, len(ch), ch)
        CGEventPost(_TAP_LOCATION, up)
        if per_char_delay_ms:
            time.sleep(per_char_delay_ms / 1000)
