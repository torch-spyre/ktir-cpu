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
KTIR parser.

Parses KTIR MLIR text into Python IR structures.
Handles multi-line operations, nested regions (scf.for loop bodies),
and all attribute syntaxes produced by the compiler.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from .ir_types import Operation, IRFunction, IRModule
from .dialects import dispatch_parser, make_parse_context, ParseContext
from .parser_utils import parse_attr_block, parse_tensor_type, parse_numeric


class KTIRParserBase(ABC):
    """Abstract mixin for KTIR parsers.

    Subclasses implement ``parse_module``; ``parse_file`` is provided
    concretely and reads the file before calling ``parse_module``.

    Concrete implementations:
        KTIRParser          — regex + custom dialect parsers (this file)
        MLIRFrontendParser  — mlir.ir walk (ktir_cpu/mlir_frontend/parser.py)
    """

    @abstractmethod
    def parse_module(self, mlir_text: str) -> IRModule: ...

    def parse_file(self, filepath: str) -> IRModule:
        with open(filepath) as f:
            return self.parse_module(f.read())


class KTIRParser(KTIRParserBase):
    """KTIR MLIR parser.

    Parsing proceeds in three phases:

    1. **Module-level pre-scan** — collects named attribute aliases
       (``#name = value``) so that dialect parsers can resolve ``#name``
       references during op parsing.

    2. **Tokenization** (``_tokenize_operations``) — splits the body
       text of a function or region into complete operation strings.
       Multi-line ops are joined, and ``{ }`` blocks are classified
       structurally: blocks whose content contains ``%`` SSA
       references are extracted as **regions** (recursively parsed);
       all others are kept as inline **attribute blocks**.

    3. **Op dispatch** — each operation string is dispatched to a
       dialect-registered parser (``register_parser`` in
       ``dialects/registry.py``).  Unrecognised ops fall through to
       the general-purpose parser.

    To add support for a new operation, register a parser in the
    appropriate dialect module — the tokenizer and dispatcher are
    dialect-agnostic and should not require changes.  Specifically:

    - Do **not** add operation-specific keywords to the tokenizer
      (e.g. continuation-line prefixes).  Multi-line grouping relies
      on type-terminal detection, blank-line boundaries, and SSA
      assignment signals — all of which are structural.
    - Do **not** hardcode op names for region detection.  Regions are
      detected by the ``%`` heuristic; any ``{ }`` block that
      contains SSA references is automatically treated as a region.
    """

    def parse_module(self, mlir_text: str) -> IRModule:
        """Parse KTIR MLIR text into module.

        Args:
            mlir_text: MLIR text

        Returns:
            IRModule with parsed functions
        """
        module = IRModule()

        # Pre-scan for named attribute aliases declared at module scope.
        # These appear before module {} as:
        #   #name = affine_set<...>   (may span multiple lines)
        #   #name = affine_map<...>
        # We use _extract_attr_value to correctly capture multi-line values
        # (e.g. affine_set<... \n ... >) without truncating at the first newline.
        from .parser_utils import _extract_attr_value
        for alias_match in re.finditer(r'^(#[\w.]+)\s*=\s*', mlir_text, re.MULTILINE):
            rest = mlir_text[alias_match.end():]
            value, _ = _extract_attr_value(rest, None)
            module.aliases[alias_match.group(1)] = value.strip()

        # Find each func.func declaration and extract the body using
        # brace counting, which correctly skips the attributes { ... } block.
        for match in re.finditer(r'func\.func\s+@(\w+)\s*\(([^)]*)\)', mlir_text):
            func_name = match.group(1)
            func_header_end = match.end()

            # Extract the full header up to the body-opening brace.
            # There may be an attributes { ... } block before the body { ... }.
            # We skip brace-balanced blocks until we find the body.
            func_body, body_end = self._extract_brace_body(mlir_text, func_header_end)
            if func_body is None:
                continue

            func_header = mlir_text[match.start():body_end]

            # Parse grid attribute
            grid = self._parse_grid_attribute(func_header)

            # Parse function arguments
            args = self._parse_function_args(mlir_text[match.start():func_header_end])

            # Build parse context from the module-level alias table so that
            # dialect parsers can resolve #name references in op attributes.
            parse_ctx = make_parse_context(module.aliases)

            # Parse operations from function body
            operations = self._parse_operations(func_body, parse_ctx)

            func = IRFunction(
                name=func_name,
                arguments=args,
                operations=operations,
                grid=grid
            )

            module.add_function(func)

        return module

    def _extract_brace_body(self, text: str, start: int):
        """Extract the last top-level brace-balanced block after start.

        After func.func @name(args), the text looks like:
            -> rettype attributes { grid = [...] } { body }
        We skip everything (return type, attributes block) and return the
        contents of the last { ... } block, which is the function body.

        Returns:
            (body_text, end_pos) or (None, -1) if not found.
        """
        pos = start
        last_body = None
        last_end = -1

        while pos < len(text):
            ch = text[pos]
            if ch == '{':
                # Find matching close brace with nesting
                depth = 1
                inner_start = pos + 1
                pos += 1
                while pos < len(text) and depth > 0:
                    if text[pos] == '{':
                        depth += 1
                    elif text[pos] == '}':
                        depth -= 1
                    pos += 1
                if depth == 0:
                    last_body = text[inner_start:pos - 1]
                    last_end = pos
                else:
                    return (None, -1)
            elif ch == '}':
                # Hit the closing brace of an outer scope (e.g. module {}).
                # Stop — the last block we found is the function body.
                break
            else:
                pos += 1

        return (last_body, last_end)

    def _parse_grid_attribute(self, func_header: str) -> Tuple[int, int, int]:
        """Parse grid attribute from function header.

        Args:
            func_header: Function header text

        Returns:
            (x, y, z) grid dimensions
        """
        # Pattern: grid = [X, Y, Z] or grid = [X, Y]
        grid_match = re.search(r'grid\s*=\s*\[(\d+),\s*(\d+)(?:,\s*(\d+))?\]', func_header)
        if grid_match:
            x = int(grid_match.group(1))
            y = int(grid_match.group(2))
            z = int(grid_match.group(3)) if grid_match.group(3) else 1
            return (x, y, z)
        return (1, 1, 1)  # Default: single-core when no grid attribute is present

    def _parse_function_args(self, func_header: str) -> List[Tuple[str, str]]:
        """Parse function arguments.

        Args:
            func_header: Function header text

        Returns:
            List of (name, type) tuples
        """
        args = []
        # Pattern: %name: type
        arg_pattern = r'%(\w+)\s*:\s*([^,)]+)'
        for match in re.finditer(arg_pattern, func_header):
            name = "%" + match.group(1)
            arg_type = match.group(2).strip()
            args.append((name, arg_type))
        return args

    # ------------------------------------------------------------------
    # Multi-line joining and region-aware operation parsing
    # ------------------------------------------------------------------

    def _parse_operations(self, body_text: str, parse_ctx: Optional[ParseContext] = None) -> List[Operation]:
        """Parse operations from a function or region body.

        Handles multi-line operations by joining continuation lines, and
        handles nested regions (scf.for loop bodies) by recursively
        extracting brace-delimited blocks that contain operations.

        Args:
            body_text:  Function/region body text (between outer braces)
            parse_ctx:  Parse-time context (alias table, etc.) forwarded to
                        dialect parsers so #name refs are resolved immediately.

        Returns:
            List of operations
        """
        # Step 0: Strip inline comments so downstream methods don't need
        # to worry about % or other tokens appearing inside comments.
        body_text = self._preprocess_text(body_text)

        # Step 1: Tokenize body into complete operation strings.
        # An "operation string" is one or more physical lines that together
        # form a single MLIR operation. Multi-line ops like
        # ktdp.construct_memory_view are joined. Region bodies (the { ... }
        # block of scf.for) are extracted separately and attached to the op.
        op_strings = self._tokenize_operations(body_text)

        # Step 2: Parse each operation string into an Operation object.
        operations = []
        for op_text, regions_text in op_strings:
            op = self._parse_operation_text(op_text, parse_ctx)
            if op:
                # If this operation has region bodies (e.g. scf.for),
                # recursively parse each region.
                for region_body in regions_text:
                    region_ops = self._parse_operations(region_body, parse_ctx)
                    op.regions.append(region_ops)
                operations.append(op)

        return operations

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Normalize MLIR text before tokenization.

        Applied once before ``_tokenize_operations`` so that downstream
        methods (region detection, type-terminal checks) operate on
        clean text.  Currently strips inline comments (``// ...``
        through end-of-line); future preprocessing steps (e.g.
        whitespace normalization) can be added here.
        """
        return re.sub(r'//[^\n]*', '', text)

    def _tokenize_operations(self, body_text: str) -> List[Tuple[str, List[str]]]:
        """Split body text into (operation_text, [region_bodies]) pairs.

        Walks through the body character by character. When we hit a '{',
        we figure out if it starts an inline attribute block or a region body:
        - Attribute blocks: { key = val, ... } on the same logical line
        - Region bodies: { <newline> ops... <newline> } contain full operations

        Returns a list of tuples where:
        - op_text: the operation text with attribute blocks inlined
        - region_bodies: list of region body strings (for scf.for etc.)
        """
        results = []
        lines = body_text.split('\n')
        i = 0
        current_op_lines = []
        current_regions = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # A blank line (or comment) separates operations.  Flush
            # the accumulated op if braces are balanced.
            if not stripped or stripped.startswith('//'):
                if current_op_lines:
                    accumulated = ' '.join(current_op_lines)
                    open_braces = accumulated.count('{') - accumulated.count('}')
                    if open_braces == 0:
                        results.append((accumulated, current_regions))
                        current_op_lines = []
                        current_regions = []
                i += 1
                continue

            # Flush when the accumulated op is structurally complete
            # (type annotation or void terminator) with balanced braces,
            # OR when the incoming line is an SSA assignment (%name = )
            # which unambiguously starts a new operation.
            #
            # Type-terminal flush (adjacent ops without blank line):
            #   %c0 = arith.constant 0 : index   <- complete (: index)
            #   %c1 = arith.constant 1 : index   <- new op
            #
            # SSA flush (prev op has no type terminal, e.g. dimensions=):
            #   linalg.broadcast ... dimensions = [1]   <- no type
            #   %next = arith.subf ...                  <- SSA triggers flush
            #
            # Multi-line ops stay together until complete:
            #   %t = construct_indirect_access_tile   <- no type yet
            #       intermediate_variables(...)        <- continuation
            #       %view[...] { ... } : ... -> type  <- complete
            accumulated = ' '.join(current_op_lines)
            open_braces = accumulated.count('{') - accumulated.count('}')
            # A line starting with '->' is always a result-type
            # continuation (e.g. `: input_types\n  -> result_type`).
            if current_op_lines and open_braces == 0 and not stripped.startswith('->'):
                starts_ssa = re.match(
                    r'(?:%\w+\s*,\s*)*%\w+\s*=\s', stripped
                )
                if self._is_op_complete(accumulated) or starts_ssa:
                    results.append((accumulated, current_regions))
                    current_op_lines = []
                    current_regions = []

            current_op_lines.append(stripped)

            # Check if this line opens a region body.
            # A region { } is distinguished from an attribute { } by
            # peeking inside: regions contain %SSA references,
            # attribute blocks do not.
            if self._line_opens_region(stripped, lines, i):
                # The '{' at the end of this line opens a region.
                # Remove the trailing '{' from the op text.
                current_op_lines[-1] = stripped.rstrip('{').rstrip()
                if not current_op_lines[-1]:
                    current_op_lines.pop()

                # Extract region body from subsequent lines
                region_body, end_line, trailing = self._extract_region_from_lines(lines, i)
                if region_body is not None:
                    current_regions.append(region_body)
                    if trailing:
                        current_op_lines.append(trailing)
                    i = end_line + 1
                    continue
                # If extraction failed, just continue

            i += 1

        # Flush any remaining operation
        if current_op_lines:
            op_text = ' '.join(current_op_lines)
            results.append((op_text, current_regions))

        return results

    # Regex matching a terminal type token at end of op text.
    # Covers: tensor<...>, memref<...>, !ktdp.access_tile<...>,
    #         index, i32, f16, f32, etc.
    _TYPE_TERMINAL_RE = re.compile(
        r'(?::\s|->)\s*.*(?:>|index|[iuf]\d+)\s*$'
    )

    def _is_op_complete(self, accumulated: str) -> bool:
        """Check whether *accumulated* op text is structurally complete.

        An MLIR operation is complete when it has a terminal type
        annotation (``:``, ``->``) or is a void terminator (``return``,
        ``scf.yield``).  Ops that end without a type annotation
        (e.g. ``linalg.reduce ... dimensions = [1]``) rely on a
        trailing blank line to flush instead.
        """
        text = accumulated.rstrip()
        if not text:
            return False
        # Block labels: ^name(%arg: type):
        if text.startswith('^') and text.endswith(':'):
            return True
        # Void terminators (return, *.yield)
        if 'return' in text or '.yield' in text:
            return True
        # Type annotation: `: <type>` or `-> <type>` at end
        if self._TYPE_TERMINAL_RE.search(text):
            return True
        return False

    # MLIR SSA value names: % followed by a digit, letter, or id-punct [$._-].
    _SSA_REF_RE = re.compile(r'%[\w$.]')

    def _line_opens_region(self, stripped_line: str, lines: List[str], line_idx: int) -> bool:
        """Check if the current line opens a region body.

        A region ``{`` is distinguished from an attribute block ``{``
        structurally: attribute blocks contain ``key = value`` pairs
        (no ``%`` SSA references), while region blocks contain
        operations that always reference ``%`` SSA names.

        Comments are already stripped by ``_preprocess_text``, so we
        only need to check for ``%`` followed by a valid SSA name
        character (letter, digit, ``$``, ``.``, ``_``).
        """
        if not stripped_line.endswith('{'):
            return False
        depth = 1
        for j in range(line_idx + 1, len(lines)):
            inner = lines[j]
            depth += inner.count('{') - inner.count('}')
            if self._SSA_REF_RE.search(inner):
                return True
            if depth <= 0:
                break
        return False

    def _extract_region_from_lines(self, lines: List[str], open_brace_line: int) -> Tuple[Optional[str], int, str]:
        """Extract a region body starting from the line that has the opening '{'.

        The opening '{' is at the end of lines[open_brace_line].
        We scan forward to find the matching '}', tracking brace depth.

        Text after the closing '}' on the same line (e.g. `: tensor<16x16xf16>`
        in tensor.generate) is returned as *trailing* so the caller can append
        it to the outer operation text.

        Returns:
            (region_body_text, closing_brace_line_index, trailing_text)
            or (None, -1, "") on failure.
        """
        # The opening '{' is already accounted for (depth starts at 1)
        depth = 1
        body_lines = []
        i = open_brace_line + 1

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Count braces in this line
            for ch in stripped:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # Found the closing brace.
                        close_pos = stripped.rfind('}')
                        # Text before '}' is part of the body
                        before_close = stripped[:close_pos].strip()
                        if before_close:
                            body_lines.append(before_close)
                        # Text after '}' belongs to the outer op
                        # (e.g. `: tensor<16x16xf16>` in tensor.generate)
                        after_close = stripped[close_pos + 1:].strip()
                        return ('\n'.join(body_lines), i, after_close)

            body_lines.append(line)
            i += 1

        return (None, -1, "")

    # ------------------------------------------------------------------
    # Operation parsing
    # ------------------------------------------------------------------

    def _parse_operation_text(self, op_text: str, parse_ctx: Optional[ParseContext] = None) -> Optional[Operation]:
        """Parse a complete operation string into an Operation.

        Dispatches to dialect-registered parsers first, then falls back
        to the general-purpose parser.

        Args:
            op_text:    Complete operation text (may contain spaces from joining)
            parse_ctx:  Parse-time context forwarded to dialect parsers so they
                        can resolve #name alias refs in op attributes.

        Returns:
            Operation or None
        """
        op_text = op_text.strip()
        if not op_text:
            return None

        # Infix shorthand: %result = %lhs [+\-*] %rhs : index
        if re.match(r'%\w+\s*=\s*%\w+\s*[\+\-\*]\s*%\w+', op_text):
            return self._parse_index_binary(op_text)

        # Use an empty ParseContext if none was provided (e.g. in tests that
        # call the parser directly without a module-level alias pre-scan).
        ctx = parse_ctx or make_parse_context({})

        parser_fn = dispatch_parser(op_text)
        if parser_fn:
            return parser_fn(op_text, ctx)

        return self._parse_general_operation(op_text)

    def _parse_index_binary(self, text: str) -> Optional[Operation]:
        """Parse infix index arithmetic: %result = %a OP %b : type

        Converts the shorthand syntax used in KTIR (e.g.
        ``%offset = %core_id * %BLOCK_SIZE : index``) into the equivalent
        arith dialect operation so the interpreter can execute it.
        """
        m = re.match(
            r'(%\w+)\s*=\s*(%\w+)\s*(\*|\+|-)\s*(%\w+)\s*:\s*(\w+)',
            text.strip()
        )
        if not m:
            return None
        op_map = {'*': 'arith.muli', '+': 'arith.addi', '-': 'arith.subi'}
        return Operation(
            result=m.group(1),
            op_type=op_map[m.group(3)],
            operands=[m.group(2), m.group(4)],
            attributes={},
            result_type=m.group(5),
        )

    # ------------------------------------------------------------------
    # General-purpose operation parser (fallback)
    # ------------------------------------------------------------------

    def _parse_general_operation(self, text: str) -> Optional[Operation]:
        """Parse a general operation using pattern matching.

        Handles simple operations like:
            %result = op_type %op1, %op2 : type
            op_type %op1, %op2 : type
            return %result : type
        """
        # Try to match: %result = op_type rest
        op_match = re.match(r'(?:(%\w+)\s*=\s*)?([a-z_][a-z0-9_\.]*)\s*(.*)', text, re.DOTALL)
        if not op_match:
            return None

        result = op_match.group(1)
        op_type = op_match.group(2)
        rest = op_match.group(3).strip()

        # Extract operands: all %name references in the text after op_type,
        # but before any { } attribute blocks.
        operands = self._extract_operands(rest, result)

        # Extract attributes from { ... } blocks.
        # Be careful not to confuse attribute blocks with region blocks
        # (region blocks were already extracted).
        attributes = self._extract_attributes(rest)

        # Extract result type. For operations with "-> type" we use that.
        # Otherwise we look for ": type" at the end.
        result_type = self._extract_result_type(rest)

        # For operations that have result type info, extract shape/dtype
        # and put them in attributes for the interpreter.
        if result_type:
            type_info = parse_tensor_type(result_type)
            if type_info and "shape" not in attributes:
                attributes["_result_shape"] = type_info["shape"]
                attributes["_result_dtype"] = type_info.get("dtype", "f16")

        return Operation(
            result=result,
            op_type=op_type,
            operands=operands,
            attributes=attributes,
            result_type=result_type
        )

    def _extract_operands(self, text: str, result: Optional[str]) -> List[str]:
        """Extract SSA operands from operation text.

        Finds all %name references, excluding:
        - The result name itself
        - References inside { } attribute blocks
        """
        # Remove all { ... } blocks to avoid picking up references inside
        # attribute blocks.
        cleaned = re.sub(r'\{[^}]*\}', '', text)

        operands = re.findall(r'%\w+', cleaned)

        # Remove result name if present
        if result:
            operands = [o for o in operands if o != result]

        return operands

    def _extract_attributes(self, text: str, aliases: Optional[Dict] = None) -> Dict:
        """Extract attributes from the outermost { ... } block in operation text."""
        return parse_attr_block(text, aliases)

    def _extract_result_type(self, text: str) -> Optional[str]:
        """Extract result type from operation text.

        Looks for "-> type" first (for ops like construct_memory_view),
        then falls back to ": type" at the end.
        """
        # Check for "-> type" pattern
        arrow_match = re.search(r'->\s*(.+?)$', text)
        if arrow_match:
            return arrow_match.group(1).strip()

        # Check for ": type" at the end, skipping attribute blocks.
        # We want the last ": type" that is not inside { ... }.
        # Remove all { ... } blocks first.
        cleaned = re.sub(r'\{[^{}]*\}', '', text)
        # Also remove nested braces (for multi-level attributes)
        while '{' in cleaned:
            cleaned = re.sub(r'\{[^{}]*\}', '', cleaned)

        type_match = re.search(r':\s*(.+?)$', cleaned)
        if type_match:
            result_type = type_match.group(1).strip()
            # Filter out things that are clearly not types
            if result_type and not result_type.startswith('%'):
                return result_type

        return None

