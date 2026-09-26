"""Correctness-gated benchmark for mojo-base65536.

The reference is the real `base65536` package. A NumPy formulation is also
timed, because a `take` plus a reshape is the fastest fair equivalent and a
port that only beat a per-byte Python loop would be claiming the easy win. Both
are reported, and every case checks the output against the reference string
before timing.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import base65536 as real  # noqa: E402

import mojo_base65536 as mine  # noqa: E402
from mojo_base65536 import _lib  # noqa: E402

_ENCODE_TABLE = _lib._ENCODE_TABLE
_DECODE_TABLE = _lib._DECODE_TABLE
ODD = int(_ENCODE_TABLE[256])


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _numpy_encode(payload: bytes) -> str:
    """The fastest fair NumPy equivalent: take plus a bulk UTF-32 decode."""
    src = np.frombuffer(payload, dtype=np.uint8)
    if src.size == 0:
        return ""
    count = (src.size + 1) // 2
    pad = np.zeros(count * 2, dtype=np.uint8)
    pad[: src.size] = src
    second_idx = np.zeros(count, dtype=np.int64)
    second_idx[:] = pad[1::2]
    if src.size % 2:
        second_idx[count - 1] = 256  # odd payload: no second byte
    cps = (_ENCODE_TABLE[second_idx].astype(np.int64) + pad[::2]).astype(np.uint32)
    return cps.tobytes().decode("utf-32-le")


def bench_encode(n: int, repeats: int):
    payload = os.urandom(n)
    expect = real.encode(payload)
    assert mine.encode(payload) == expect, "encode mismatch"
    assert _numpy_encode(payload) == expect, "numpy reference mismatch"
    return (
        f"encode {n}",
        _time(lambda: real.encode(payload), repeats),
        _time(lambda: mine.encode(payload), repeats),
        _time(lambda: _numpy_encode(payload), repeats),
    )


def _numpy_decode(text: str) -> bytes:
    """A complete NumPy decode, not just the UTF-32 widening.

    The inverse is a gather on the high byte, so the whole thing vectorises; the
    awkward part is that a block is one or two bytes, which is handled by
    materialising pairs and masking the half of a single-byte block away.
    """
    cps = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
    if cps.size == 0:
        return b""
    b1 = (cps & 0xFF).astype(np.uint8)
    b2 = _DECODE_TABLE[(cps >> 8).astype(np.int64)]
    if int((b2 < -1).any()):
        raise ValueError("Invalid Base-65536 code point")
    single = b2 < 0
    if int(single.sum()) > 1:
        raise ValueError("Base-65536 sequence continued after final byte")
    pairs = np.empty((cps.size, 2), dtype=np.uint8)
    pairs[:, 0] = b1
    pairs[:, 1] = np.where(single, 0, b2).astype(np.uint8)
    width = np.where(single, 1, 2)
    flat = pairs.reshape(-1)
    keep = np.repeat(width == 2, 2) | (np.arange(cps.size * 2) % 2 == 0)
    return flat[keep].tobytes()


def bench_decode(n: int, repeats: int):
    text = real.encode(os.urandom(n))
    expect = real.decode(text)
    assert mine.decode(text) == expect, "decode mismatch"
    assert _numpy_decode(text) == expect, "numpy reference mismatch"
    return (
        f"decode {n}",
        _time(lambda: real.decode(text), repeats),
        _time(lambda: mine.decode(text), repeats),
        _time(lambda: _numpy_decode(text), repeats),
    )


def main():
    plan = [(16, 2000), (256, 1000), (4096, 300), (65536, 60), (1048576, 8)]
    print(
        f"{'case':<16}{'base65536':>13}{'mojo':>13}{'numpy ref':>13}"
        f"{'vs up':>9}{'vs numpy':>10}"
    )
    print("-" * 74)
    for n, repeats in plan:
        for label, ref, ours, numpy_ref in (
            bench_encode(n, repeats),
            bench_decode(n, repeats),
        ):
            print(
                f"{label:<16}{ref*1e3:>11.3f}ms{ours*1e3:>11.3f}ms"
                f"{numpy_ref*1e3:>11.3f}ms{ref/ours:>8.2f}x"
                f"{numpy_ref/ours:>9.2f}x"
            )
    print()
    print("vs up: real base65536 / mojo. vs numpy: numpy reference / mojo.")
    print("Both numpy references are full codecs: a take-based encode and a")
    print("gather-and-mask decode. They are the fastest fair equivalent, not a")
    print("strawman.")


if __name__ == "__main__":
    main()
