"""Gemini vision client: send image + prompt, receive a parsed JSON object."""
import io
import logging
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from .config import Config

log = logging.getLogger(__name__)


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(e: Exception) -> bool:
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(e, genai_errors.ServerError):
        return True
    if isinstance(e, genai_errors.ClientError) and code in _RETRYABLE_STATUS:
        return True
    # Raw HTTP timeouts and transient connection errors from httpx surface
    # directly as httpx exceptions; the SDK does not wrap them in ServerError.
    # Retry those as transient network blips.
    try:
        import httpx
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
            return True
    except ImportError:
        pass
    # Fallback: match on the message text for the common timeout wording.
    msg = str(e).lower()
    if "read operation timed out" in msg or "timeout" in msg and "connect" not in msg:
        return True
    return False


def enable_debug_logging(wire_level: bool = True) -> None:
    """Turn on timing + wire-level logs. Call once at program start."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if wire_level:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)
        logging.getLogger("google_genai").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


class GeminiClient:
    def __init__(
        self,
        config: Config,
        timeout_s: float = 20.0,
        max_attempts: int = 5,
        initial_backoff_s: float = 1.5,
    ) -> None:
        self._client = genai.Client(
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self._model = config.model
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._initial_backoff_s = initial_backoff_s
        log.info(
            "GeminiClient ready model=%s timeout=%.1fs retries<=%d",
            self._model, timeout_s, max_attempts - 1,
        )

    def ask[T: BaseModel](
        self,
        prompt: str,
        image: Image.Image,
        schema: type[T],
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> T:
        t0 = time.monotonic()
        png = _png_bytes(image)
        log.info(
            "encode %dx%d -> %.1f KB PNG (%.2fs)",
            image.width, image.height, len(png) / 1024, time.monotonic() - t0,
        )

        contents = [
            types.Part.from_bytes(data=png, mime_type="image/png"),
            prompt,
        ]
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        log.info(
            "request model=%s schema=%s prompt_chars=%d",
            self._model, schema.__name__, len(prompt),
        )
        response = self._generate_with_retry(contents, cfg)

        parsed = response.parsed
        if parsed is None:
            log.warning("response.parsed was None; falling back to text parse")
            log.debug("raw text: %s", response.text)
            parsed = schema.model_validate_json(response.text)
        log.info("total %.2fs", time.monotonic() - t0)
        return parsed

    def ask_with_video[T: BaseModel](
        self,
        prompt: str,
        video_path: Path,
        schema: type[T],
        image: Image.Image | None = None,
        system_instruction: str | None = None,
        temperature: float | None = None,
        delete_after: bool = True,
    ) -> T:
        """Upload a video, optionally pair with an image, send with prompt.

        Blocks until Gemini reports processing is complete, then returns a
        parsed Pydantic instance. Deletes the uploaded file afterwards by
        default to avoid accumulating storage.
        """
        t0 = time.monotonic()
        size_mb = video_path.stat().st_size / 1e6
        log.info("uploading video: %s (%.1f MB)", video_path, size_mb)
        uploaded = self._client.files.upload(file=str(video_path))
        try:
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self._client.files.get(name=uploaded.name)
            if uploaded.state.name == "FAILED":
                raise RuntimeError(f"Gemini video processing FAILED: {uploaded}")
            log.info(
                "video processed in %.1fs state=%s",
                time.monotonic() - t0, uploaded.state.name,
            )

            contents: list = [uploaded]
            if image is not None:
                png = _png_bytes(image)
                log.info("final-frame PNG %.1f KB", len(png) / 1024)
                contents.append(types.Part.from_bytes(data=png, mime_type="image/png"))
            contents.append(prompt)

            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                system_instruction=system_instruction,
                temperature=temperature,
            )
            log.info("request model=%s schema=%s (with video)", self._model, schema.__name__)
            response = self._generate_with_retry(contents, cfg)
            parsed = response.parsed
            if parsed is None:
                log.warning("response.parsed was None; falling back to text parse")
                log.debug("raw text: %s", response.text)
                parsed = schema.model_validate_json(response.text)
            log.info("total %.2fs", time.monotonic() - t0)
            return parsed
        finally:
            if delete_after:
                try:
                    self._client.files.delete(name=uploaded.name)
                except Exception as e:
                    log.warning("failed to delete uploaded video: %s", e)

    def _generate_with_retry(self, contents, cfg):
        backoff = self._initial_backoff_s
        last_err: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            t = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=cfg,
                )
                log.info("response attempt=%d in %.2fs", attempt, time.monotonic() - t)
                return response
            except Exception as e:
                last_err = e
                elapsed = time.monotonic() - t
                if not _is_retryable(e) or attempt == self._max_attempts:
                    log.error(
                        "request failed (attempt %d/%d) after %.2fs: %s",
                        attempt, self._max_attempts, elapsed, e,
                    )
                    raise
                log.warning(
                    "attempt %d/%d failed after %.2fs (%s); retrying in %.1fs",
                    attempt, self._max_attempts, elapsed, type(e).__name__, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        assert last_err is not None
        raise last_err
