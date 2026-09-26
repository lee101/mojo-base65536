"""mojo-base65536: Base-65536 encoding with the byte loop in Mojo.

Installable alongside the real `base65536` package, which the parity tests
compare against.
"""

from .core import B2, BLOCK_START, block_start, decode, encode

__all__ = ["B2", "BLOCK_START", "block_start", "decode", "encode"]
__version__ = "0.1.0"
