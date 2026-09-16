"""Small terminal selector with a numbered-input fallback."""
from __future__ import annotations

import os
import select
import sys


class PromptCanceled(ValueError):
    """Selection canceled without changing settings."""


def interactive_available() -> bool:
    try:
        import termios
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            return False
        termios.tcgetattr(sys.stdin.fileno())
        return os.environ.get("TERM") != "dumb"
    except (ImportError, OSError, ValueError):
        return False


def _read_raw_key(timeout: float | None = None) -> str | None:
    """Read bytes directly: TextIO buffering can hide queued escape bytes."""
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], timeout)[0]:
        return None
    first = os.read(fd, 1)
    if not first:
        return None
    sequence = first
    if first == b"\x1b":
        while len(sequence) < 16 and select.select([fd], [], [], 0.08)[0]:
            byte = os.read(fd, 1)
            if not byte:
                break
            sequence += byte
            if len(sequence) >= 3 and 0x40 <= byte[0] <= 0x7e:
                break
    return sequence.decode("ascii", "replace")


def choose_option(options: list[str], prompt: str, default: int = 0) -> int:
    if not options:
        raise ValueError("No options to choose from")
    default = max(0, min(default, len(options) - 1))
    if not interactive_available():
        return _choose_plain(options, prompt, default)
    import shutil
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    selection = default
    try:
        tty.setraw(fd)
        sys.stdout.write("\x1b[?1049h\x1b[?25l")
        while True:
            columns, rows = shutil.get_terminal_size((80, 24))
            count = max(1, rows - 3)
            start = min(max(0, selection - count + 1), max(0, len(options) - count))
            lines = [f"{'>' if i == selection else ' '} {options[i]}"[:max(1, columns - 1)]
                     for i in range(start, min(len(options), start + count))]
            lines.append(f"{prompt}: arrows/j/k move; Enter selects; q/Esc cancels"[:max(1, columns - 1)])
            sys.stdout.write("\x1b[H\x1b[2J" + "\r\n".join(lines))
            sys.stdout.flush()
            key = _read_raw_key()
            if key is None or key in ("q", "\x03", "\x04", "\x1b"):
                raise PromptCanceled("Selection canceled; settings unchanged")
            if key in ("\r", "\n"):
                return selection
            if key in ("\x1b[A", "\x1bOA", "k"):
                selection = (selection - 1) % len(options)
            elif key in ("\x1b[B", "\x1bOB", "j"):
                selection = (selection + 1) % len(options)
            elif key in ("\x1b[H", "\x1bOH", "\x1b[1~"):
                selection = 0
            elif key in ("\x1b[F", "\x1bOF", "\x1b[4~"):
                selection = len(options) - 1
            elif key == "\x1b[5~":
                selection = max(0, selection - count)
            elif key == "\x1b[6~":
                selection = min(len(options) - 1, selection + count)
    except KeyboardInterrupt:
        raise PromptCanceled("Selection canceled; settings unchanged") from None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


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
