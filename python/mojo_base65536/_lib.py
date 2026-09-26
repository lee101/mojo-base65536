"""ctypes bridge to the compiled base-65536 kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults. Scratch buffers are allocated here, on the Python
side, which keeps the exported Mojo symbols free of allocation and of `raises`.

Code points cross as a `uint32` array, and the string conversion is done in
bulk. That is the second half of the port: upstream calls `unichr` once per
code point on the way out and `ord` once per character on the way in, which for
a megabyte of payload is a million Python-level calls. A `uint32` buffer plus
one `utf-32-le` codec call replaces all of them.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

from .tables import ODD_INDEX, build_decode_table, build_encode_table

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-base65536.so"

_ENCODE_TABLE = np.array(build_encode_table(), dtype=np.uint32)
_DECODE_TABLE = np.array(build_decode_table(), dtype=np.int32)
_MAX_CODE_POINT = 0x10FFFF


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    i = ctypes.c_int64
    lib.b65536_encode.restype = i
    lib.b65536_encode.argtypes = [i, i, i, i, i]
    lib.b65536_decode.restype = i
    lib.b65536_decode.argtypes = [i, i, i, i, i, i, i]
    return lib


lib = _load()

_STATUS_MESSAGES = {
    1: "Invalid Base-65536 code point: {index}",
    2: "Base-65536 sequence continued after final byte",
    3: "base65536 decode output buffer too small",
}


def _status(slot: np.ndarray) -> tuple[int, int]:
    return int(slot[0]), int(slot[1])


def code_points(data: bytes) -> list[int]:
    """The code points `encode` would emit, without building the string."""
    n = len(data)
    if n == 0:
        return []
    src = np.frombuffer(data, dtype=np.uint8)
    cps = np.zeros((n + 1) // 2, dtype=np.uint32)
    written = lib.b65536_encode(
        src.ctypes.data, n, _ENCODE_TABLE.ctypes.data, cps.ctypes.data, cps.size
    )
    if written < 0:
        raise RuntimeError("base65536 encode scratch too small")
    return [int(v) for v in cps[:written]]


def encode(value: bytes) -> str:
    """Encodes bytes to a Base-65536 string."""
    cps = code_points(value)
    if not cps:
        return ""
    return np.array(cps, dtype=np.uint32).tobytes().decode("utf-32-le")


def _code_point_array(value: str) -> np.ndarray:
    """Bulk UTF-32 view of a string.

    `surrogatepass` keeps lone surrogates as their own code units instead of
    raising a `UnicodeEncodeError` here. The reference reaches them through
    `ord(ch)`, sees 0xD800, and reports an invalid code point; letting the codec
    reject them keeps that error and its message.
    """
    return np.frombuffer(value.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)


def decode(value: str) -> bytes:
    """Decodes bytes from a Base-65536 string."""
    if not isinstance(value, str):
        raise TypeError("base65536.decode expects a str")
    cps = _code_point_array(value)
    n = int(cps.size)
    if n == 0:
        return b""
    out = np.zeros(n * 2, dtype=np.uint8)
    status = np.zeros(2, dtype=np.int32)
    written = lib.b65536_decode(
        cps.ctypes.data, n, _DECODE_TABLE.ctypes.data, _DECODE_TABLE.size,
        out.ctypes.data, out.size, status.ctypes.data,
    )
    if written < 0:
        code, index = _status(status)
        message = _STATUS_MESSAGES.get(code, "base65536 decode failed")
        if code == 1:
            raise ValueError(message.format(index=int(cps[index])))
        raise ValueError(message)
    return out.tobytes()[:written]


def block_start(b2: int) -> int:
    """Code-point base for a second byte, or the odd-length block."""
    return int(_ENCODE_TABLE[b2 if b2 >= 0 else ODD_INDEX])
