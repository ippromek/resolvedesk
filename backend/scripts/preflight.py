"""Checklist for the laptop before a live demo. Exits non-zero on any FAIL."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.config import ROOT, get_settings
from app.prompts import INVESTIGATE_PROMPT_VERSION, PROMPT_VERSION

SNAPSHOT = ROOT / "data" / "demo" / "snapshot.json"
CHECKS: list[tuple[str, str]] = []


def _record(level: str, message: str) -> None:
    CHECKS.append((level, message))
    print(f"{level:<4}  {message}")


def _get_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _port_number(name: str, fallback: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        print(f"{name} is not a port number ({raw}); using {fallback}")
        return fallback


def _port_from_url(url: str, fallback: int) -> int:
    parsed = urllib.parse.urlparse(url)
    if parsed.port:
        return parsed.port
    return fallback


def _listener_pids(port: int) -> list[int]:
    try:
        netstat = subprocess.check_output(["netstat", "-ano"], text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Could not list listeners: {exc}")
        return []
    pids: list[int] = []
    for line in netstat.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        local = parts[1]
        if local.endswith(f":{port}"):
            try:
                pids.append(int(parts[-1]))
            except ValueError:
                continue
    return pids


def _is_resolvedesk(port: int) -> bool:
    """True when this port is serving ResolveDesk, not another app on the same port."""
    try:
        health = _get_json(f"http://127.0.0.1:{port}/api/health", timeout=2)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        health = None
    if isinstance(health, dict) and "demo_mode" in health and "prompt_version" in health:
        return True
    for host in ("127.0.0.1", "localhost"):
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/", timeout=2) as response:
                body = response.read(4000).decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        if "ResolveDesk" in body:
            return True
    return False


def check_api(api: str) -> dict[str, Any] | None:
    try:
        health = _get_json(api.rstrip("/") + "/api/health", timeout=5)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        _record("FAIL", f"API not reachable at {api} ({type(exc).__name__})")
        return None
    _record("PASS", f"API reachable at {api}")
    model = str(health.get("model", ""))
    prompt = str(health.get("prompt_version", ""))
    investigate = str(health.get("investigate_prompt_version", ""))
    _record("PASS", f"model {model}")
    if prompt == PROMPT_VERSION and investigate == INVESTIGATE_PROMPT_VERSION:
        _record("PASS", f"prompts draft {prompt}, investigate {investigate}")
    else:
        _record(
            "FAIL",
            f"prompts draft {prompt} (expected {PROMPT_VERSION}), investigate {investigate} (expected {INVESTIGATE_PROMPT_VERSION})",
        )
    if health.get("demo_mode") is True:
        _record("PASS", "demo mode on")
    else:
        _record("FAIL", "demo mode is off")
    return health if isinstance(health, dict) else None


def check_snapshot(health: dict[str, Any] | None) -> None:
    if not SNAPSHOT.exists():
        _record("FAIL", f"snapshot missing at {SNAPSHOT}")
        return
    try:
        manifest = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _record("FAIL", f"snapshot.json could not be read ({type(exc).__name__})")
        return
    cases = manifest.get("cases") or []
    _record("PASS", f"snapshot present ({len(cases)} cases, {manifest.get('model')})")
    db = SNAPSHOT.parent / "snapshot.db"
    ckpt = SNAPSHOT.parent / "snapshot-ckpt.db"
    if not db.exists() or not ckpt.exists():
        _record("FAIL", "snapshot.db or snapshot-ckpt.db is missing")
    if health and manifest.get("model") != health.get("model"):
        _record(
            "WARN", f"snapshot model {manifest.get('model')} does not match the running model {health.get('model')}"
        )
    else:
        _record("PASS", "snapshot model matches the running model")


def check_model(skip: bool) -> None:
    if skip:
        _record("PASS", "model call skipped (--no-model)")
        return
    try:
        settings = get_settings()
    except RuntimeError as exc:
        _record("FAIL", str(exc))
        return
    if settings.llm_provider == "offline":
        _record("FAIL", "LLM_PROVIDER is offline, so the cheap model call was not made")
        return
    from app.brain import make_chat_model

    started = time.perf_counter()
    try:
        make_chat_model(settings).invoke("Reply with the single word OK.")
    except Exception as exc:  # noqa: BLE001  # any provider failure is recorded as its type, then the checklist continues
        _record("FAIL", f"model call failed ({type(exc).__name__})")
        return
    elapsed = time.perf_counter() - started
    if elapsed > 15:
        _record("FAIL", f"model call took {elapsed:.1f}s (limit 15s)")
        return
    _record("PASS", f"model call succeeded in {elapsed:.1f}s")


def check_frontend(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = getattr(response, "status", 200)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _record("FAIL", f"frontend not reachable at {url} ({type(exc).__name__})")
        return
    if status >= 500:
        _record("FAIL", f"frontend {url} returned {status}")
        return
    _record("PASS", f"frontend reachable at {url}")


def check_ports(api_port: int, frontend_port: int) -> None:
    for port in (api_port, frontend_port):
        pids = _listener_pids(port)
        if not pids:
            _record("PASS", f"port {port} is free")
            continue
        if _is_resolvedesk(port):
            _record("PASS", f"port {port} is ResolveDesk (pid {pids[0]})")
            continue
        for pid in pids:
            _record("FAIL", f"port {port} is used by another process (not ResolveDesk) (pid {pid})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResolveDesk demo pre-flight.")
    api_port = _port_number("API_PORT", 8100)
    frontend_port = _port_number("VITE_PORT", 5180)
    parser.add_argument("--api", default=f"http://127.0.0.1:{api_port}")
    parser.add_argument("--frontend", default=f"http://localhost:{frontend_port}")
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args(argv)
    CHECKS.clear()
    health = check_api(args.api)
    check_snapshot(health)
    check_model(args.no_model)
    check_frontend(args.frontend)
    check_ports(_port_from_url(args.api, api_port), _port_from_url(args.frontend, frontend_port))
    failed = sum(1 for level, _message in CHECKS if level == "FAIL")
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
