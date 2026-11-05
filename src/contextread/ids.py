"""Opaque, roughly time-ordered identifiers (cuid-like): base36 milliseconds + random."""

import secrets
import string
import time

_ALPHABET = string.digits + string.ascii_lowercase


def _base36(n: int) -> str:
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = _ALPHABET[r] + out
    return out or "0"


def new_id(prefix: str = "") -> str:
    """25 characters after the prefix: 'c' + 9 time chars + 15 random chars."""
    rand = "".join(secrets.choice(_ALPHABET) for _ in range(15))
    return f"{prefix}c{_base36(time.time_ns() // 1_000_000).rjust(9, '0')}{rand}"


def new_doc_id() -> str:
    return "doc_" + secrets.token_hex(12)
