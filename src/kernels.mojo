"""Base-65536 block mapping in compiled Mojo.

The codec is a table-driven byte-pair transform: two bytes become one Unicode
code point by adding the second byte to a 256-entry block start, and the
inverse recovers the pair from the code point's high and low bytes. There is no
arithmetic beyond an 8-bit add and two lookups, and it would be dishonest to
pretend otherwise; what the port removes is the per-character Python work, not
a mathematical bottleneck.

The decoder does more than map. It carries the reference implementation's
`done` flag, so a single-byte block is legal exactly once and only before any
further block, and it validates surrogates and out-of-range scalars in the same
pass, so an invalid payload is rejected before a single byte is handed back.
Both of those are in the loop because putting them anywhere else means a second
walk.

Buffers cross the C ABI as 64-bit addresses and are rebuilt inside each body,
because `@export` rejects a function whose parameter types are inferred.
Nothing here allocates, so no exported symbol is `raises`.
"""

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32 = Pointer[UInt32, AnyOrigin[mut=True]]
comptime I32 = Pointer[Int32, AnyOrigin[mut=True]]

comptime ODD_INDEX = 256


def bu8(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def bu32(addr: Int) -> U32:
    return U32(unsafe_from_address=addr)


def bi32(addr: Int) -> I32:
    return I32(unsafe_from_address=addr)


@export("b65536_encode")
def b65536_encode(src_addr: Int, n: Int, table_addr: Int, cps_addr: Int,
                  out_cap: Int) abi("C") -> Int:
    """Map `n` bytes to ceil(n/2) code points, written as uint32.

    `table` is the 257-entry block start table: index 0..255 is the block for a
    second byte, index 256 is the block for a trailing odd byte with no second
    byte at all. Returns the number of code points written, or -1 if `out_cap`
    is too small.
    """
    if n <= 0:
        return 0
    if (n + 1) // 2 > out_cap:
        return -1
    var src = bu8(src_addr)
    var tbl = bu32(table_addr)
    var cps = bu32(cps_addr)
    var k = 0
    var i = 0
    while i < n:
        var base = tbl[unsafe_offset=ODD_INDEX]
        if i + 1 < n:
            base = tbl[unsafe_offset=src[unsafe_offset=i + 1]]
        cps[unsafe_offset=k] = base + UInt32(src[unsafe_offset=i])
        k += 1
        i += 2
    return k


@export("b65536_decode")
def b65536_decode(cps_addr: Int, n: Int, inv_addr: Int, table_len: Int,
                  obuf_addr: Int, out_cap: Int, status_addr: Int) abi("C") -> Int:
    """Map `n` code points back to bytes, validating as it goes.

    `inv` is indexed by `code_point >> 8` and holds the second byte, or -1 for
    the odd-length block, or -2 for a code point the codec does not define.
    `table_len` bounds that index: a code point near U+10FFFF indexes far past
    the end of a 646-entry table, and a scalar range check alone does not catch
    it, because the high byte of a valid scalar is still a legal integer.
    `status` receives a two-entry pair: 0 ok, 1 unknown code point, 2 a second
    odd-length block, 3 output capacity exceeded. Returns the number of bytes
    written, or -1 on any of those.
    """
    var st = bi32(status_addr)
    st[unsafe_offset=0] = 0
    st[unsafe_offset=1] = 0
    if n <= 0:
        return 0
    if n > out_cap:
        # Even the smallest block is one byte, so a cap below n is impossible.
        if out_cap < n:
            st[unsafe_offset=0] = 3
            return -1
    var cps = bu32(cps_addr)
    var inv = bi32(inv_addr)
    var ob = bu8(obuf_addr)
    var k = 0
    var done = False
    for i in range(n):
        var cp = cps[unsafe_offset=i]
        if cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF):
            st[unsafe_offset=0] = 1
            st[unsafe_offset=1] = Int32(i)
            return -1
        if Int32(cp >> 8) >= Int32(table_len):
            st[unsafe_offset=0] = 1
            st[unsafe_offset=1] = Int32(i)
            return -1
        var b1 = Int32(cp & 0xFF)
        var b2 = inv[unsafe_offset=Int32(cp >> 8)]
        if b2 == -2:
            st[unsafe_offset=0] = 1
            st[unsafe_offset=1] = Int32(i)
            return -1
        if b2 == -1:
            # The reference allows one odd-length block and only before any
            # other block; a second one is a truncated-then-resumed sequence.
            if done:
                st[unsafe_offset=0] = 2
                st[unsafe_offset=1] = Int32(i)
                return -1
            done = True
            if k >= out_cap:
                st[unsafe_offset=0] = 3
                return -1
            ob[unsafe_offset=k] = UInt8(b1)
            k += 1
        else:
            if k + 2 > out_cap:
                st[unsafe_offset=0] = 3
                return -1
            ob[unsafe_offset=k] = UInt8(b1)
            ob[unsafe_offset=k + 1] = UInt8(b2)
            k += 2
    return k
