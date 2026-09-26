"""Base-65536 encoding with the byte loop in Mojo.

The public surface mirrors `base65536` 0.1.1: `encode` and `decode`, returning
the same strings and raising the same `ValueError` messages, so this package
imports alongside the real one and the parity tests compare the two directly.

The code-point allocation is data and lives in `tables.py`; the transform over
it is a loop, and that loop is the compiled part. The reference implementation
walks the payload one Python object at a time and builds the result through an
`io.StringIO` or `io.BytesIO`, so what this port removes is per-character
interpreter and per-character stream work rather than arithmetic.
"""

from __future__ import annotations

from . import _lib
from .tables import BLOCK_START, build_b2_map

__all__ = ["BLOCK_START", "B2", "block_start", "decode", "encode"]

# The reference decoder keys its inverse map on the code point, not on the high
# byte. Exposing the same shape makes the tables comparable in a test without
# either package reaching into the other's internals.
B2 = build_b2_map()

block_start = _lib.block_start
encode = _lib.encode
decode = _lib.decode
