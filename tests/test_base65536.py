"""Parity against the real `base65536` package, plus structural checks on the table.

Base-65536 is exact byte and code-point work: there is no floating point
anywhere in it, so every comparison here is equality, and nothing is asserted
to a tolerance.
"""

import os

import base65536 as real
import numpy as np
import pytest

import mojo_base65536 as mine
from mojo_base65536 import _lib

ODD = mine.BLOCK_START[256]


def test_tables_match_upstream():
    """The allocation is data, and the wrong data produces a self-consistent
    codec that simply does not interoperate. So it is compared, not trusted."""
    assert list(mine.BLOCK_START[:256]) == [
        real.core.BLOCK_START[i] for i in range(256)
    ]
    assert mine.BLOCK_START[256] == real.core.BLOCK_START[-1]
    assert mine.B2 == real.core.B2
    assert mine.block_start(-1) == real.core.BLOCK_START[-1]
    for b2 in range(256):
        assert mine.block_start(b2) == real.core.BLOCK_START[b2]


def test_table_has_the_structure_the_codec_relies_on():
    """Two properties make the scheme invertible.

    A zero low byte is what lets `code_point & 0xFF` recover the first byte, and
    a distinct high byte is what lets the second byte be recovered from the code
    point alone. Break either and the codec becomes ambiguous rather than merely
    wrong, which is much harder to notice.
    """
    starts = list(mine.BLOCK_START)
    assert all(v % 256 == 0 for v in starts)
    assert len({v >> 8 for v in starts}) == len(starts)
    assert len(starts) == 257


def test_every_two_byte_value_round_trips():
    """All 65 536 byte pairs, not a sample.

    The pair is the whole alphabet of the codec, so a table entry that is off by
    one, swapped, or missing shows up here and nowhere else.
    """
    for b1 in range(256):
        row = bytes([b1]) + bytes(range(256))
        encoded = mine.encode(row)
        assert encoded == real.encode(row)
        assert mine.decode(encoded) == row
        assert len(encoded) == (len(row) + 1) // 2


def test_every_single_byte_value_round_trips():
    for b in range(256):
        encoded = mine.encode(bytes([b]))
        assert encoded == real.encode(bytes([b]))
        assert len(encoded) == 1
        assert ord(encoded) == ODD + b
        assert mine.decode(encoded) == bytes([b])


def test_low_byte_recovers_the_first_byte():
    payload = os.urandom(64)
    for index, code_point in enumerate(_lib.code_points(payload)):
        assert code_point & 0xFF == payload[2 * index]
        if 2 * index + 1 < len(payload):
            assert code_point - (code_point & 0xFF) == mine.block_start(
                payload[2 * index + 1]
            )
        else:
            assert code_point - (code_point & 0xFF) == ODD


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33])
def test_encode_matches_upstream(n):
    payload = os.urandom(n)
    assert mine.encode(payload) == real.encode(payload)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 17, 100, 1000, 4096, 65537])
def test_round_trip_matches_upstream(n):
    payload = os.urandom(n)
    encoded = mine.encode(payload)
    assert encoded == real.encode(payload)
    assert mine.decode(encoded) == payload
    assert real.decode(encoded) == payload


def test_empty_payload_gives_the_empty_string():
    assert mine.encode(b"") == ""
    assert real.encode(b"") == ""
    assert mine.decode("") == b""
    assert real.decode("") == b""


def test_code_points_helper_agrees_with_the_string():
    payload = os.urandom(37)
    assert _lib.code_points(payload) == [ord(c) for c in mine.encode(payload)]
    assert _lib.code_points(b"") == []


def test_unknown_code_point_raises_the_reference_message():
    for text in ("A", "A" * 4, chr(0x4E2D) + "A", chr(0x110000 - 1)):
        with pytest.raises(ValueError) as ours:
            mine.decode(text)
        with pytest.raises(ValueError) as theirs:
            real.decode(text)
        assert str(ours.value) == str(theirs.value)


def test_code_point_beyond_the_table_is_rejected():
    """A scalar near U+10FFFF has a legal high byte, so a scalar-range check
    alone still lets it index far past the end of the 646-entry inverse table.
    A Python string cannot hold a scalar above U+10FFFF at all, so those code
    points are unreachable from this API; the ones below are not."""
    for code_point in (0x10FFFF, 0x10FF00, 0x10F000, 0x2000, 0xFFFF):
        text = chr(code_point)
        with pytest.raises(ValueError) as ours:
            mine.decode(text)
        with pytest.raises(ValueError) as theirs:
            real.decode(text)
        assert str(ours.value) == str(theirs.value)


def test_lone_surrogate_is_an_invalid_code_point():
    """The reference sees 0xD800 through `ord` and reports it as invalid.

    Encoding the string to UTF-32 first would raise a different error entirely,
    so the surrogate has to survive the conversion to reach the codec.
    """
    with pytest.raises(ValueError) as ours:
        mine.decode(chr(0xD800))
    with pytest.raises(ValueError) as theirs:
        real.decode(chr(0xD800))
    assert str(ours.value) == str(theirs.value)
    assert "55296" in str(ours.value)


def test_a_second_odd_length_block_is_rejected():
    text = chr(ODD + 65) + chr(ODD + 66)
    with pytest.raises(ValueError) as ours:
        mine.decode(text)
    with pytest.raises(ValueError) as theirs:
        real.decode(text)
    assert str(ours.value) == str(theirs.value)
    assert str(ours.value) == "Base-65536 sequence continued after final byte"


def test_one_odd_length_block_anywhere_is_accepted():
    """The reference tracks a `done` flag, not a position.

    An odd block in the middle of the string is accepted and its byte is
    emitted in place, even though the string is no longer the encoding of the
    bytes it produces. A stricter implementation would reject this, so matching
    the reference here is a deliberate choice rather than an oversight.
    """
    odd = chr(ODD + 65)
    double = chr(mine.block_start(7) + 9)
    for text in (odd + double, double + odd, odd):
        assert mine.decode(text) == real.decode(text)
    assert mine.decode(odd + double) == bytes([65, 9, 7])
    assert mine.decode(double + odd) == bytes([9, 7, 65])


def test_decode_rejects_bytes():
    with pytest.raises(TypeError):
        mine.decode(b"abc")
    with pytest.raises(TypeError):
        real.decode(b"abc")


def test_random_payloads_match_upstream():
    rng = np.random.default_rng(20260926)
    for _ in range(300):
        n = int(rng.integers(0, 400))
        payload = bytes(rng.integers(0, 256, size=n, dtype="uint8"))
        encoded = mine.encode(payload)
        assert encoded == real.encode(payload)
        assert mine.decode(encoded) == payload


def test_payloads_with_repeated_and_extreme_bytes_match_upstream():
    for payload in (
        b"\x00" * 101,
        b"\xff" * 101,
        bytes(range(256)) * 3,
        b"\x00\xff" * 64,
        b"\xff\x00" * 64,
    ):
        assert mine.encode(payload) == real.encode(payload)
        assert mine.decode(mine.encode(payload)) == payload
