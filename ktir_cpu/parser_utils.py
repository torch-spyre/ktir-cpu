# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General MLIR attribute block parsing utilities.

No dependencies on parser.py or the dialects package — safe to import from
either side without creating a circular import.
"""

import re
from typing import Any, Dict, List, Optional

_SSA_RE = re.compile(r'%\w+(?:#\d+)?')


def find_ssa_names(text: str) -> list[str]:
    """Find all SSA value references in *text*, including multi-result ``%base#N`` forms."""
    return _SSA_RE.findall(text)


_OUTS_RE = re.compile(r'\bouts\s*\(([^)]+)\)')


def extract_outs_operands(op_text: str) -> list[str]:
    """Extract SSA names from the ``outs(...)`` clause of a linalg op text."""
    m = _OUTS_RE.search(op_text)
    if not m:
        return []
    names = []
    for segment in split_top_level(m.group(1)):
        names.extend(find_ssa_names(segment.split(':')[0]))
    return names


def parse_multi_result_lhs(lhs_text: str) -> list[str]:
    """Parse the LHS of a multi-result MLIR assignment.

    Accepts:
      bundled form  ``"%g:2"``    -> ``["%g#0", "%g#1"]``
      comma form    ``"%x, %y"``  -> ``["%x", "%y"]``
      single        ``"%x"``      -> ``["%x"]``

    Raises ``ValueError`` on malformed input.
    """
    m = re.fullmatch(r'(%\w+):([1-9]\d*)', lhs_text.strip())
    if m:
        base, n = m.group(1), int(m.group(2))
        return [f"{base}#{i}" for i in range(n)]
    parts = [p.strip() for p in lhs_text.split(",")]
    if all(re.fullmatch(r'%\w+', p) for p in parts):
        return parts
    raise ValueError(f"cannot parse multi-result LHS: {lhs_text!r}")


def parse_tensor_type(type_str: str) -> Optional[Dict]:
    """Parse a tensor type string, returning shape and dtype if it matches.

    Args:
        type_str: Type string (e.g., "tensor<256xf16>")

    Returns:
        {"shape": tuple, "dtype": str} if tensor type, else None

    Walks the leading dim prefix by matching digit-tokens followed by 'x',
    taking the remainder as dtype. Handles dtypes containing 'x' (e.g.
    ``index``). Dynamic dims (``?``) are silently dropped, matching prior
    behaviour.
    """
    m = re.match(r'tensor<([^>]+)>', type_str)
    if not m:
        return None
    inner = m.group(1).strip()
    # Match all ``NNN x`` dim tokens from the left. The dtype cannot start
    # with a digit followed by 'x', so the pattern terminates at the right
    # boundary even when the dtype itself contains 'x' (e.g. ``index``).
    # Dynamic dims (``?``) are skipped, matching prior behaviour.
    prefix = re.match(r'^((?:\d+\s*x\s*|[?]\s*x\s*)+)', inner)
    if not prefix:
        return None
    dims = [int(d) for d in re.findall(r'(\d+)\s*x', prefix.group(1))]
    if not dims:
        return None
    dtype = inner[prefix.end():].split(',')[0].strip()
    if not dtype:
        return None
    return {"shape": tuple(dims), "dtype": dtype}

def parse_numeric(s: str, dtype: Optional[str] = None) -> Any:
    """Parse a numeric string to a Python int or float.

    Handles: integers, floats, scientific notation, hex constants.
    When *dtype* is a float type (f16, f32, bf16) and the literal is hex,
    the value is reinterpreted as an IEEE 754 bit pattern.

    All float types are widened to Python ``float`` (64-bit double) and
    all integers to Python ``int`` (arbitrary precision) — the caller
    is responsible for narrowing to the target dtype at use-site.
    """
    import numpy as np

    s = s.strip()

    if s.startswith('0x') or s.startswith('0X'):
        bits = int(s, 16)
        # Float types: reinterpret hex as IEEE 754 bit pattern (e.g. 0xFC00 : f16 → -inf)
        # Integer/index types: keep as plain int (e.g. 0xFF800000 : i32 → 4286578688)
        if dtype == 'f32':
            return float(np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])
        elif dtype == 'f16':
            return float(np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0])
        elif dtype == 'bf16':
            # bf16 is the upper 16 bits of an f32 (1 sign + 8 exp + 7 mantissa),
            # unlike f16 which has its own layout (1 sign + 5 exp + 10 mantissa).
            # Shift left into a 32-bit word and reinterpret as f32.
            bits_32 = (bits & 0xFFFF) << 16
            return float(np.array([bits_32], dtype=np.uint32).view(np.float32)[0])
        return bits

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    return 0


def parse_dense_payload(payload: str, elem_dtype: Optional[str] = None):
    """Parse the content already extracted from inside ``dense<...>``.

    Returns ``(value, is_list)``:
    - scalar payload ``0.0``       → ``(scalar, False)``
    - list payload   ``[16, 32]``  → ``([16, 32], True)``
    """
    payload = payload.strip()
    is_list = payload.startswith("[")
    if is_list:
        assert payload.endswith("]"), f"malformed dense list payload: {payload!r}"
        parts = [p.strip() for p in payload[1:-1].split(",") if p.strip()]
        return [parse_numeric(p, dtype=elem_dtype) for p in parts], True
    return parse_numeric(payload, dtype=elem_dtype), False


def _extract_bracket_content(op_text: str, brackets: str = '{}') -> Optional[str]:
    """Return the content inside the outermost matched bracket pair.

    Handles nested brackets of the same kind.  Returns ``None`` when no
    matching pair is found.
    """
    open_ch, close_ch = brackets[0], brackets[1]
    start = op_text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(op_text)):
        if op_text[i] == open_ch:
            depth += 1
        elif op_text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return op_text[start + 1:i]
    return None


def parse_attr_block(op_text: str, aliases: Optional[Dict] = None,
                     brackets: str = '{}') -> Dict:
    """Extract key=value pairs from the outermost bracketed attribute block.

    By default extracts from ``{ ... }``; pass ``brackets='[]'`` for
    ``[ ... ]``.

    Values are returned as Python scalars (int/float/list/str).  Handles:

    - ``keyword<...>`` values (e.g. ``affine_map<...>``, ``affine_set<...>``,
      ``#ktdp.spyre_memory_space<HBM>``): ``<``/``>`` depth is counted while
      skipping ``>=`` and ``->`` operators so constraint expressions like
      ``d0 >= 0`` do not prematurely close the value.
    - ``#alias`` references: resolved via *aliases* when provided.
    - Plain tokens, integers, floats, and ``[...]`` lists.

    Args:
        op_text:  Full operation text (or the bracketed block text).
        aliases:  Optional ``"#name" -> verbatim string`` mapping from
                  ``IRModule.aliases``.
        brackets: Two-character string of open/close bracket chars.

    Returns:
        ``{key: value}`` dict, or ``{}`` if no block found.
    """
    block = _extract_bracket_content(op_text, brackets)
    if block is None:
        return {}

    result: Dict = {}
    pos = 0
    while pos < len(block):
        # Skip whitespace and commas between entries
        while pos < len(block) and block[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(block):
            break

        # Parse key: word characters and dots up to '='
        key_m = re.match(r'[\w.]+', block[pos:])
        if not key_m:
            pos += 1
            continue
        key = key_m.group(0)
        pos += len(key)

        # Consume optional whitespace and '='
        eq_m = re.match(r'\s*=\s*', block[pos:])
        if not eq_m:
            continue
        pos += eq_m.end()

        # Extract raw value string; skip entry if value is absent
        raw, consumed = _extract_attr_value(block[pos:], aliases)
        pos += consumed
        if not raw:
            continue

        result[key] = _coerce_attr_value(raw)

    return result


def parse_attr_list(op_text: str, aliases: Optional[Dict] = None,
                    brackets: str = '[]') -> List:
    """Extract a list of values from the outermost bracketed block.

    Like :func:`parse_attr_block` but for bare value lists (no keys),
    e.g. ``[affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>]``.

    Returns:
        List of raw value strings, or ``[]`` if no block found.
    """
    block = _extract_bracket_content(op_text, brackets)
    if block is None:
        return []

    result: List = []
    pos = 0
    while pos < len(block):
        while pos < len(block) and block[pos] in ' \t\n\r,':
            pos += 1
        if pos >= len(block):
            break

        raw, consumed = _extract_attr_value(block[pos:], aliases)
        pos += consumed
        result.append(raw)

    return result


def _scan_angle_bracketed(text: str, start: int) -> int:
    """Return the index one past the closing ``>`` of an angle-bracketed span.

    *start* must point at the opening ``<``.  Skips ``>=`` (constraint
    operator) and ``->`` (affine_map arrow) so they are not mistaken for
    closing brackets.  Returns ``len(text)`` if no matching ``>`` is found.
    """
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '>' and i + 1 < len(text) and text[i + 1] == '=':
            i += 2
            continue
        if ch == '-' and i + 1 < len(text) and text[i + 1] == '>':
            i += 2
            continue
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def extract_named_attr(op_text: str, key: str,
                       aliases: Optional[Dict] = None) -> Optional[str]:
    """Extract a single bare ``key = value`` attribute from ``op_text``.

    Sibling to :func:`parse_attr_block` for ops that carry attributes
    *outside* a ``{ ... }`` block — e.g. ops whose attribute list lives
    on the op header itself rather than wrapped in braces:
    ``producer_tiles_per_group = #all_tiles, groups = #one_group``.

    Returns the resolved value string (alias-resolved when applicable),
    or ``None`` if not found.  Caller-driven: caller names the key it
    expects.
    """
    m = re.search(rf'\b{re.escape(key)}\s*=\s*', op_text)
    if not m:
        return None
    rest = op_text[m.end():].lstrip()

    # #alias reference
    if rest.startswith('#'):
        end = re.search(r'[,\n}]|\s+:|\s*->', rest)
        raw = rest[:end.start()].strip() if end else rest.strip()
        return aliases.get(raw, raw) if aliases else raw

    # keyword<...> values — count <> depth, skip >= and ->
    kw_m = re.match(r'[\w.]+<', rest)
    if kw_m:
        end = _scan_angle_bracketed(rest, kw_m.end() - 1)
        return rest[:end]

    # Plain token up to next comma / newline / brace / colon / arrow.
    end = re.search(r'[,\n}]|\s+:|\s*->', rest)
    return rest[:end.start()].strip() if end else rest.strip()


def _extract_attr_value(text: str, aliases: Optional[Dict]) -> tuple:
    """Extract one attribute value from the start of *text*.

    Returns ``(value_str, chars_consumed)``.

    Handles ``#alias`` refs, ``keyword<...>`` values (skipping ``>=``/``->``
    inside the body), and plain tokens.
    """
    stripped = text.lstrip()
    skip = len(text) - len(stripped)

    # #alias reference
    if stripped.startswith('#'):
        end = re.search(r'[,}]', stripped)
        raw = stripped[:end.start()].strip() if end else stripped.strip()
        consumed = end.start() if end else len(stripped)
        resolved = aliases.get(raw, raw) if aliases else raw
        return resolved, skip + consumed

    # keyword<...> values — count <> depth, skip >= and ->
    kw_m = re.match(r'[\w.]+<', stripped)
    if kw_m:
        end = _scan_angle_bracketed(stripped, kw_m.end() - 1)
        return stripped[:end], skip + end

    # Plain token up to next comma or closing brace
    end = re.search(r'[,}]', stripped)
    if end:
        return stripped[:end.start()].strip(), skip + end.start()
    return stripped.strip(), skip + len(stripped)


def split_top_level(text: str) -> list:
    """Split *text* on commas that are not inside parentheses or brackets.

    Commas nested inside ``(...)`` or ``[...]`` are treated as part of the
    current token and do not cause a split.

    Example::

        split_top_level("(%h), ind(%IDX[%m, %k]), (%n)")
        # -> ["(%h)", " ind(%IDX[%m, %k])", " (%n)"]
    """
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _coerce_attr_value(value_str: str):
    """Coerce a raw attribute value string to int, float, list, or str."""
    # Strip MLIR type annotation suffix (e.g. "0 : i32" -> "0")
    value_str = re.sub(r'\s*:\s*\S+$', '', value_str).strip()
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    list_match = re.match(r'\[([^\]]+)\]', value_str)
    if list_match:
        try:
            return [int(x.strip()) for x in list_match.group(1).split(',')]
        except ValueError:
            pass
    return value_str
