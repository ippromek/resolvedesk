from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Tiny .env loader so the project has no extra dependency for it."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


# Env var each provider reads. Unknown providers use "<NAME>_API_KEY".
_PROVIDER_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PPLX_API_KEY",
}


def provider_key_variable(provider: str) -> str:
    return _PROVIDER_KEY_VARS.get(provider, f"{provider.upper()}_API_KEY")


def ensure_provider_credentials(settings: Settings) -> None:
    """Refuse to start a live provider when its key variable is unset.

    The message names the variable only. It never includes the key.
    """
    if settings.llm_provider == "offline":
        return
    variable = provider_key_variable(settings.llm_provider)
    if not (os.getenv(variable) or "").strip():
        raise RuntimeError(f"LLM_PROVIDER is {settings.llm_provider} but {variable} is not set. Refusing to start.")


@dataclass(frozen=True)
class Settings:
    # "offline" runs deterministic rules instead of an LLM: tests, CI, and a demo
    # that still works when the conference Wi-Fi does not.
    llm_provider: str = "offline"
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float = 0
    llm_timeout_s: float = 30
    llm_max_retries: int = 2
    db_path: Path = ROOT / "data" / "resolvedesk.db"
    checkpoint_path: Path = ROOT / "data" / "checkpoints.db"
    kb_dir: Path = ROOT / "kb"
    enterprise_dir: Path = ROOT / "data" / "enterprise"
    demo_mode: bool = False
    demo_dir: Path = ROOT / "data" / "demo"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Read process env at call time so a missing key fails at startup."""
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "offline"),
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        db_path=Path(os.getenv("DB_PATH", str(ROOT / "data" / "resolvedesk.db"))),
        checkpoint_path=Path(os.getenv("CHECKPOINT_PATH", str(ROOT / "data" / "checkpoints.db"))),
        demo_mode=_env_flag("DEMO_MODE"),
        demo_dir=Path(os.getenv("DEMO_DIR", str(ROOT / "data" / "demo"))),
    )
    ensure_provider_credentials(settings)
    return settings
