"""Inline terminal menus: arrows or numbers, preserving answered questions."""
from __future__ import annotations

import os
import sys


class PromptCanceled(ValueError):
    """Selection canceled without changing settings."""


def interactive_available() -> bool:
    try:
        import termios
        return sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
    except (ImportError, OSError, ValueError):
        return False


def choose_option(options: list[str], prompt: str, default: int = 0) -> int:
    if not options:
        raise ValueError('No options to choose from')
    default = max(0, min(default, len(options) - 1))
    try:
        index = (_choose_inline(options, prompt, default) if interactive_available()
                 else _choose_plain(options, prompt, default))
    except (KeyboardInterrupt, EOFError):
        raise PromptCanceled('Selection canceled; settings unchanged') from None
    if index < 0:
        raise PromptCanceled('Selection canceled; settings unchanged')
    print(f'{prompt}: {options[index]}')
    return index


def _choose_inline(options, prompt, default):
    import select
    import shutil
    import termios
    import tty
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    index, digits, rows = default, '', 0

    def clear():
        nonlocal rows
        if rows:
            sys.stdout.write(f'\x1b[{rows}A\r\x1b[J')
            rows = 0

    def read_key():
        key = os.read(fd, 1)
        if not key:
            raise EOFError
        if key != b'\x1b':
            return key
        sequence = b''
        while select.select([fd], [], [], 0.08)[0]:
            sequence += os.read(fd, 1)
            if len(sequence) > 1 and (sequence[-1:] in b'ABCDHF~' or len(sequence) >= 8):
                break
        # tmux mouse tracking emits SGR (\x1b[<b;x;yM/m) and X10 (\x1b[M+3)
        # reports; ignore them so drags don't cancel inline menus.
        if sequence.startswith(b'[M'):
            while select.select([fd], [], [], 0.05)[0]:
                extra = os.read(fd, 1)
                if not extra:
                    break
                sequence += extra
                if len(sequence) >= 6:
                    break
            return b''
        if sequence.startswith(b'[<'):
            return b''
        return {b'[A': b'up', b'OA': b'up', b'[B': b'down', b'OB': b'down',
                b'[H': b'home', b'OH': b'home', b'[1~': b'home',
                b'[F': b'end', b'OF': b'end', b'[4~': b'end'}.get(sequence, b'escape' if not sequence else b'')

    try:
        tty.setcbreak(fd)
        sys.stdout.write('\x1b[?25l')
        while True:
            clear()
            width, height = shutil.get_terminal_size((80, 24))
            count = max(1, min(len(options), height - 5))
            start = max(0, min(index - count // 2, len(options) - count))
            lines = ['\x1b[1;97;44m ' + prompt +
                     '  ↑/↓ or number, Enter; q/Esc cancels  [' + digits + '] \x1b[0m']
            for i in range(start, start + count):
                marker = '\x1b[1;92m▸\x1b[0m' if i == index else ' '
                number = f'\x1b[2m{i+1:>2}.\x1b[0m'
                option = options[i]
                lines.append(f' {marker} {number} {option}')
            for line in lines:
                # Keep each menu entry on one row, including Unicode wide text.
                import unicodedata
                clean, cells = '', 0
                text = str(line)
                pos = 0
                while pos < len(text):
                    if text[pos] == '\x1b':
                        end = pos + 1
                        while end < len(text) and not text[end].isalpha():
                            end += 1
                        if end < len(text):
                            end += 1
                            clean += text[pos:end]
                            pos = end
                            continue
                    char = text[pos]
                    pos += 1
                    if unicodedata.category(char).startswith('C'):
                        char = ' '
                    size = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in 'WF' else 1)
                    if cells + size > max(1, width - 1):
                        break
                    clean += char
                    cells += size
                sys.stdout.write(clean + '\n')
            rows = len(lines)
            sys.stdout.flush()
            key = read_key()
            if key in (b'up', b'k', b'down', b'j', b'home', b'end'):
                digits = ''
                index = (0 if key == b'home' else len(options)-1 if key == b'end'
                         else (index + (-1 if key in (b'up', b'k') else 1)) % len(options))
            elif key in (b'\r', b'\n'):
                if not digits or 1 <= int(digits) <= len(options):
                    return index
            elif key in (b'q', b'Q', b'escape', b'\x03', b'\x04', b''):
                if key == b'':
                    continue
                return -1
            elif key in (b'\x7f', b'\x08'):
                digits = digits[:-1]
            elif key in b'0123456789' and len(key) == 1:
                digits = (digits + key.decode())[:len(str(len(options)))]
            if digits and 1 <= int(digits) <= len(options):
                index = int(digits) - 1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        clear()
        sys.stdout.write('\x1b[?25h')
        sys.stdout.flush()


def _choose_plain(options: list[str], prompt: str, default: int) -> int:
    for index, option in enumerate(options, 1):
        print(f'  {index}. {option}' + (' (current)' if index - 1 == default else ''))
    while True:
        answer = input(f'{prompt} number [Enter keeps current], q to cancel: ').strip()
        if answer.lower() == 'q':
            return -1
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print('Choose one of the listed options.')
