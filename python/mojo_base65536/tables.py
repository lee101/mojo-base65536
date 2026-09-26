"""The base-65536 code-point allocation.

These 257 values are data, not code: they are the Unicode code points the
base-65536 scheme reserves for its two-byte blocks, plus the one extra
block used for a trailing odd byte. They are embedded here rather than
imported from `base65536` so this package stands on its own; the inverse
table the decoder needs is derived from them at import time, so the two
can never drift apart.

Provenance: the allocation published in `base65536.core.BLOCK_START`
(base65536 0.1.1). Each value is a multiple of 256, which is what makes
`code_point & 0xFF` recover the first byte, and each has a distinct high
byte, which is what makes the block recoverable from the code point alone.
"""

from __future__ import annotations

# Index 0..255 is the block start for a second byte b2; index 256 is the
# block used when the payload has an odd length and there is no b2.
BLOCK_START: tuple[int, ...] = (
    13312, 13568, 13824, 14080, 14336, 14592, 14848, 15104, 15360, 15616, 15872,
    16128, 16384, 16640, 16896, 17152, 17408, 17664, 17920, 18176, 18432, 18688,
    18944, 19200, 19456, 19968, 20224, 20480, 20736, 20992, 21248, 21504, 21760,
    22016, 22272, 22528, 22784, 23040, 23296, 23552, 23808, 24064, 24320, 24576,
    24832, 25088, 25344, 25600, 25856, 26112, 26368, 26624, 26880, 27136, 27392,
    27648, 27904, 28160, 28416, 28672, 28928, 29184, 29440, 29696, 29952, 30208,
    30464, 30720, 30976, 31232, 31488, 31744, 32000, 32256, 32512, 32768, 33024,
    33280, 33536, 33792, 34048, 34304, 34560, 34816, 35072, 35328, 35584, 35840,
    36096, 36352, 36608, 36864, 37120, 37376, 37632, 37888, 38144, 38400, 38656,
    38912, 39168, 39424, 39680, 39936, 40192, 40448, 41216, 41472, 41728, 42240,
    67072, 73728, 73984, 74240, 77824, 78080, 78336, 78592, 82944, 83200, 92160,
    92416, 131072, 131328, 131584, 131840, 132096, 132352, 132608, 132864,
    133120, 133376, 133632, 133888, 134144, 134400, 134656, 134912, 135168,
    135424, 135680, 135936, 136192, 136448, 136704, 136960, 137216, 137472,
    137728, 137984, 138240, 138496, 138752, 139008, 139264, 139520, 139776,
    140032, 140288, 140544, 140800, 141056, 141312, 141568, 141824, 142080,
    142336, 142592, 142848, 143104, 143360, 143616, 143872, 144128, 144384,
    144640, 144896, 145152, 145408, 145664, 145920, 146176, 146432, 146688,
    146944, 147200, 147456, 147712, 147968, 148224, 148480, 148736, 148992,
    149248, 149504, 149760, 150016, 150272, 150528, 150784, 151040, 151296,
    151552, 151808, 152064, 152320, 152576, 152832, 153088, 153344, 153600,
    153856, 154112, 154368, 154624, 154880, 155136, 155392, 155648, 155904,
    156160, 156416, 156672, 156928, 157184, 157440, 157696, 157952, 158208,
    158464, 158720, 158976, 159232, 159488, 159744, 160000, 160256, 160512,
    160768, 161024, 161280, 161536, 161792, 162048, 162304, 162560, 162816,
    163072, 163328, 163584, 163840, 164096, 164352, 164608, 164864, 165120, 5376
)

ODD_INDEX = 256


def block_start(b2: int) -> int:
    """Code-point base for a second byte, or the odd-length block."""
    return BLOCK_START[b2 if b2 >= 0 else ODD_INDEX]


def build_encode_table() -> list[int]:
    """`b2 -> block start`, with the odd-length block parked at index 256."""
    return list(BLOCK_START)


def build_decode_table() -> list[int]:
    """`code_point >> 8 -> b2`, with -1 for the odd block and -2 for unknown.

    Derived from BLOCK_START rather than stored, so the encoder and the
    decoder cannot disagree about which code point carries which byte. A
    collision would mean two byte pairs sharing a code point, which would
    make the codec ambiguous rather than merely wrong.
    """
    top = max(BLOCK_START) >> 8
    table = [-2] * (top + 1)
    for b2, start in enumerate(BLOCK_START):
        slot = start >> 8
        if table[slot] != -2:
            raise ValueError(f"code point block {slot} is claimed twice")
        table[slot] = -1 if b2 == ODD_INDEX else b2
    return table


def build_b2_map() -> dict[int, int]:
    """`code_point -> b2`, the reference decoder's own `B2` shape."""
    return {start: (-1 if i == ODD_INDEX else i) for i, start in enumerate(BLOCK_START)}

