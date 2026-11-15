import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    model: str
    target_app: str | None
    target_window_title: str | None
    login_email: str | None
    login_password: str | None


def load_config() -> Config:
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your-gemini-api-key":
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return Config(
        gemini_api_key=key,
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        target_app=os.getenv("TARGET_APP") or None,
        target_window_title=os.getenv("TARGET_WINDOW_TITLE") or None,
        login_email=os.getenv("LOGIN_EMAIL") or None,
        login_password=os.getenv("LOGIN_PASSWORD") or None,
    )
