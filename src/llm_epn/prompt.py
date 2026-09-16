"""Small terminal selector with a numbered-input fallback."""
from __future__ import annotations

import os
import sys


class PromptCanceled(ValueError):
    """Selection canceled without changing settings."""


def interactive_available() -> bool:
    try:
        import curses
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
        return os.environ.get("TERM") != "dumb"
    except (ImportError, OSError, ValueError):
        return False


def choose_option(options: list[str], prompt: str, default: int = 0) -> int:
    import curses
    from pick import pick as _pick
    if not options:
        raise ValueError("No options to choose from")
    default = max(0, min(default, len(options) - 1))
    if not interactive_available():
        return _choose_plain(options, prompt, default)
    try:
        option, index = _pick(
            options,
            title=f"{prompt}: arrows/j/k move; Enter selects; q/Esc cancels",
            default_index=default,
            quit_keys={ord("q"), ord("Q"), 27},
        )
    except KeyboardInterrupt:
        raise PromptCanceled("Selection canceled; settings unchanged") from None
    if index < 0 or option is None:
        raise PromptCanceled("Selection canceled; settings unchanged")
    return index


def _choose_plain(options: list[str], prompt: str, default: int) -> int:
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}" + (" (current)" if index - 1 == default else ""))
    while True:
        try:
            answer = input(f"Choose {prompt} number [Enter keeps current], q to cancel: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise PromptCanceled("Selection canceled; settings unchanged") from None
        if answer.lower() == "q":
            raise PromptCanceled("Selection canceled; settings unchanged")
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("Choose one of the listed options.")
