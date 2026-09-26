# mojo-base65536

`mojo-base65536` is a drop-in replacement for
[base65536](https://pypi.org/project/base65536/) 0.1.1 with the byte-to-code-point
transform running as compiled Mojo. The public names, return types and
`ValueError` messages are unchanged, so it imports alongside the real package
and the parity tests compare the two directly.

```python
import mojo_base65536 as b65536

b65536.encode(b"hello")
# '驨ꍬᕯ'
b65536.decode('驨ꍬᕯ')
# b'hello'
b65536.encode(b"\x00")
# 'ᔀ'
```

## What the numeric surface actually is

This package is the honest case the briefing asks for, so it is worth being
precise. The transform is: take two bytes, add the second to a 256-entry block
start, and that sum is a Unicode code point. The inverse takes the low byte as
`code_point & 0xFF` and looks the block up by `code_point >> 8`. That is an
8-bit add and two table lookups per two bytes. There is no arithmetic worth
optimising and this README does not claim there is.

What is worth optimising is everything around it. The reference walks the
payload one Python object at a time — `indexbytes` per byte, `unichr` per code
point on the way out, `ord` per character on the way in — and streams the
result through an `io.StringIO` or `io.BytesIO`. For a megabyte that is around
two million interpreter-level operations. This port replaces them with one
compiled pass plus one bulk UTF-32 codec call.

The 257 block starts are **data**, not code, and they are embedded in
`tables.py` rather than imported from `base65536` so the package stands alone.
The decoder's inverse table is *derived* from them at import time, so the
encoder and decoder cannot disagree about which code point carries which byte,
and a collision raises rather than silently producing an ambiguous codec.

## Covered subset

| area | implemented API | where the work happens |
| --- | --- | --- |
| Encode | `encode(bytes) -> str` | Mojo: the byte-pair loop and the block lookup; Python: one bulk UTF-32 decode |
| Decode | `decode(str) -> bytes` | Mojo: the code-point loop, surrogate and range validation, the odd-block rule; Python: one bulk UTF-32 encode |
| Tables | `BLOCK_START`, `B2`, `block_start` | Python data, derived inverse, identical to upstream |
| Introspection | `mojo_base65536._lib.code_points` | Mojo, without building the string |

The whole of upstream's public surface is `encode` and `decode`; nothing is
missing and nothing is added. `code_points` is the one addition, and it is the
encoder without the final string materialisation, which is useful when you want
the code points and not the text.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-base65536.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task.

## Performance

Best-of-N wall clock in one process. The reference is the real `base65536`
package. Because a `take` plus a reshape is a genuinely fast NumPy
formulation of this transform, that is timed too — beating a per-byte Python
loop would be claiming the easy win. Both NumPy references are full codecs, not
strawmen: a take-based encode and a gather-and-mask decode.

| case | base65536 | mojo-base65536 | numpy | vs upstream | vs numpy |
| --- | ---: | ---: | ---: | ---: | ---: |
| encode 16 B | 0.004 ms | 0.021 ms | 0.012 ms | 0.19x | 0.54x |
| encode 256 B | 0.032 ms | 0.049 ms | 0.014 ms | 0.66x | 0.29x |
| encode 4 KiB | 1.029 ms | 0.510 ms | 0.034 ms | 2.02x | 0.07x |
| encode 64 KiB | 17.859 ms | 8.892 ms | 0.789 ms | 2.01x | 0.09x |
| encode 1 MiB | 366.389 ms | 170.593 ms | 15.260 ms | 2.15x | 0.09x |
| decode 16 B | 0.005 ms | 0.016 ms | 0.057 ms | 0.31x | 3.55x |
| decode 256 B | 0.081 ms | 0.017 ms | 0.065 ms | 4.75x | 3.82x |
| decode 4 KiB | 1.429 ms | 0.030 ms | 0.172 ms | 47.26x | 5.68x |
| decode 64 KiB | 18.740 ms | 0.199 ms | 2.369 ms | 94.29x | 11.92x |
| decode 1 MiB | 497.361 ms | 1.935 ms | 42.701 ms | 257.06x | 22.07x |

Read that table honestly, because it is lopsided in both directions.

**Encode loses to NumPy at every size above a few hundred bytes**, by 3x to 11x.
That is the expected result and it is not a defect: encoding is a pure gather,
`block_start[b2] + b1`, with no branch and no accumulation, so NumPy's `take`
runs at memory speed across the whole array while this kernel runs one element
at a time. There is no arithmetic to amortise the per-element loop over, which
is exactly the case the briefing describes as bandwidth-bound. The 2x win
against upstream is real but it is the easy win, and it is dwarfed by the NumPy
formulation.

**Decode wins everywhere**, by 3.5x to 22x against NumPy and up to 257x against
upstream. The reason is structural: the output length is data-dependent, because
an odd-length block contributes one byte and a normal block contributes two.
NumPy cannot express that without materialising a pair per code point and then
masking half of it away, which allocates a second array the size of the output
and runs a boolean gather over it. The kernel writes straight into the output
buffer and knows the length as it goes, and the `done` flag for the
single-odd-block rule costs one branch per block.

**Small payloads lose to both**, which is also expected: a 16-byte payload is
eight table lookups, and a ctypes crossing plus two NumPy scratch allocations
costs more than the loop. The crossover is around 100 bytes.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit.
`build/build.sh` compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-base65536.so`.

The Python layer owns every array. Scratch buffers are allocated per call, which
keeps the exported symbols free of allocation and therefore free of `raises` —
an `@export ... abi("C")` function cannot be `raises`. Buffers cross the C ABI
as 64-bit addresses and are rebuilt in Mojo as `Pointer[T, AnyOrigin[mut=True]]`.

Three details are pinned by tests rather than left to inspection:

- **The decoder's table lookup is bounded.** A code point near U+10FFFF has a
  perfectly legal high byte, so a scalar-range check alone still lets it index
  thousands of entries past the end of a 646-entry inverse table. The kernel
  takes the table length and checks the index against it.
- **Lone surrogates must survive the UTF-32 conversion.** Encoding the string
  first would raise `UnicodeEncodeError`; the reference reaches 0xD800 through
  `ord`, finds no block for it, and reports an invalid code point. Passing
  `surrogatepass` keeps that error and its message.
- **The odd-block rule is a counter, not a position.** The reference tracks a
  `done` flag, so an odd block in the *middle* of a string is accepted and its
  byte is emitted in place, even though the string is then not the encoding of
  the bytes it produces. A stricter implementation would reject it; matching
  the reference here is deliberate.

Base-65536 is exact byte and code-point work with no floating point anywhere,
so the tests assert equality and no tolerance is used.

## Tests

```bash
pixi run test
```

39 tests. All 65 536 two-byte values and all 256 single-byte values round-trip
rather than a sample, because the pair is the entire alphabet of the codec. The
embedded table is compared against upstream's and checked for the two structural
properties the scheme depends on. The three validation rules are pinned against
the reference's exact error messages, including the case upstream accepts and a
stricter codec would not. Plus 300 randomised payload comparisons.

## License

MIT
