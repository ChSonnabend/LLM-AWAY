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
    if not options:
        raise ValueError("No options to choose from")
    default = max(0, min(default, len(options) - 1))
    if not interactive_available():
        return _choose_plain(options, prompt, default)
    try:
        index = _choose_curses(options, f"{prompt}: arrows/up-down move; Enter selects; q/Esc cancels", default)
    except KeyboardInterrupt:
        raise PromptCanceled("Selection canceled; settings unchanged") from None
    if index < 0:
        raise PromptCanceled("Selection canceled; settings unchanged")
    return index


def _choose_curses(options: list[str], title: str, default: int) -> int:
    import curses

    def _main(stdscr) -> int:
        # Some ncurses builds (notably the one bundled with macOS Python)
        # do not decode arrow-key escape sequences even with keypad(True),
        # returning the leading ESC instead. Parse the raw sequences here so
        # Up/Down work regardless of the terminfo/ncurses behavior.
        stdscr.keypad(True)
        # The primary key read blocks so the menu does not busy-spin when the
        # user is idle. Continuation bytes of an escape sequence are read with
        # a short nodelay window so a lone ESC is not mistaken for an arrow.
        try:
            curses.curs_set(0)
        except Exception:
            pass

        def next_key():
            # Returns one of: "up", "down", "home", "end", "enter", "cancel",
            # or None (ignorable key). A lone ESC means cancel; ESC followed
            # by [ + a letter is an arrow/home/end sequence.
            stdscr.nodelay(False)
            c = stdscr.getch()
            if c == -1:  # no key (should not block, but be safe)
                return None
            if c == 27:  # ESC: wait briefly to see if it starts an arrow seq.
                stdscr.nodelay(True)
                n = stdscr.getch()
                if n == -1:
                    return "cancel"
                if n == ord("["):
                    m = stdscr.getch()
                    arrows = {ord("A"): "up", ord("B"): "down", ord("F"): "home", ord("H"): "home"}
                    if m in arrows:
                        return arrows[m]
                    if m == ord("6") and stdscr.getch() == ord("~"):
                        return "end"
                    return "cancel"
                return "cancel"
            if c in (curses.KEY_UP, ord("k")):
                return "up"
            if c in (curses.KEY_DOWN, ord("j")):
                return "down"
            if c in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return "enter"
            if c in (ord("q"), ord("Q")):
                return "cancel"
            if c in (3, 4):  # Ctrl-C / Ctrl-D
                return "cancel"
            return None

        index = default
        while True:
            stdscr.erase()
            try:
                stdscr.addnstr(0, 0, title, stdscr.getmaxyx()[1] - 1)
            except Exception:
                pass
            for i, option in enumerate(options):
                marker = "* " if i == index else "  "
                try:
                    stdscr.addnstr(2 + i, 0, marker + str(option), stdscr.getmaxyx()[1] - 1)
                except Exception:
                    pass
            stdscr.refresh()
            key = next_key()
            if key is None:
                continue
            if key == "up":
                index = (index - 1) % len(options)
            elif key == "down":
                index = (index + 1) % len(options)
            elif key == "home":
                index = 0
            elif key == "end":
                index = len(options) - 1
            elif key == "enter":
                return index
            elif key == "cancel":
                return -1

    return curses.wrapper(_main)


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
