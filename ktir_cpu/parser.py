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
from typing import Any, Dict, List, Optional, Tuple
from .ir_types import Operation, IRFunction, IRModule
from .dialects import dispatch_parser, make_parse_context, ParseContext
from .parser_utils import parse_attr_block, parse_tensor_type, parse_numeric

class KTIRParser:
    """KTIR MLIR parser.

    Parses compiler-generated KTIR including multi-line operations
    (e.g. ktdp.construct_memory_view spanning 3-4 lines), nested
    regions (scf.for loop bodies), and compiler-specific constant
    syntax like {1823 : index} or {dense<0.0>}.
    """

    def parse_file(self, filepath: str) -> IRModule:
        """Parse KTIR MLIR file.

        Args:
            filepath: Path to MLIR file

        Returns:
            IRModule
        """
        with open(filepath, 'r') as f:
            text = f.read()
        return self.parse_module(text)

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

            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                # If we have accumulated lines for an operation, a blank line
                # might separate it. But we should only flush if the accumulated
                # op looks complete.
                i += 1
                continue

            # Only flush when the accumulated op text has balanced braces —
            # an open { ... } attribute block must stay on one logical op.
            # parse_attr_block uses the same depth-counting loop internally,
            # so we reuse it here: a balanced block means it found a closing '}'.
            # NOTE: underlying assumption is that the KTIR is well-formed with no 
            #       stray braces.
            accumulated = ' '.join(current_op_lines)
            open_braces = accumulated.count('{') - accumulated.count('}')
            # Don't flush when accumulated ends with '=' — the next line is
            # a continuation (e.g. multi-result: %a, %b, %c =\n  scf.for ...)
            ends_with_eq = accumulated.rstrip().endswith('=')
            if self._is_new_operation(stripped) and current_op_lines and open_braces == 0 and not ends_with_eq:
                # Flush the previous operation
                results.append((accumulated, current_regions))
                current_op_lines = []
                current_regions = []

            current_op_lines.append(stripped)

            # Check if this line opens a region body (scf.for, scf.if).
            # A region starts when a line ends with '{' and the operation
            # is a control flow op (scf.for, scf.if).
            # We need to find the matching '}' across subsequent lines.
            if self._line_opens_region(stripped, current_op_lines):
                # The '{' at the end of this line opens a region.
                # Remove the trailing '{' from the op text.
                current_op_lines[-1] = stripped.rstrip('{').rstrip()
                if not current_op_lines[-1]:
                    current_op_lines.pop()

                # Extract region body from subsequent lines
                region_body, end_line = self._extract_region_from_lines(lines, i)
                if region_body is not None:
                    current_regions.append(region_body)
                    i = end_line + 1
                    continue
                # If extraction failed, just continue

            i += 1

        # Flush any remaining operation
        if current_op_lines:
            op_text = ' '.join(current_op_lines)
            results.append((op_text, current_regions))

        return results

    def _is_new_operation(self, stripped_line: str) -> bool:
        """Check if a stripped line starts a new operation.

        A new operation starts with:
        - %result = op_name ...
        - op_name ...  (where op_name matches dialect.op pattern)
        - return ...
        - scf.for, scf.if, scf.yield

        Continuation lines start with:
        - sizes:, strides:, base_map, access_tile_set
        - : (type annotation on its own line)
        - } (closing brace of attribute block)
        """
        # Continuation patterns: these never start a new operation
        if stripped_line.startswith((
            'sizes:', 'strides:', 'base_map', 'access_tile_set',
            'ins(', 'outs(', 'dimensions', 'permutation',  # linalg ops continuation
        )):
            return False
        if stripped_line.startswith('}'):
            return False
        # A line starting with ':' is a type continuation
        if stripped_line.startswith(':'):
            return False

        # Check for result assignment: %name = op_type
        # Also handles multi-result: %a, %b, %c = op_type  (or split across lines: %a, %b, %c =)
        if re.match(r'(?:%\w+\s*,\s*)*%\w+\s*=\s*[a-z_]', stripped_line):
            return True
        if re.match(r'(?:%\w+\s*,\s*)*%\w+\s*=\s*$', stripped_line):
            return True

        # Infix shorthand: %result = %lhs * %rhs : index
        if re.match(r'%\w+\s*=\s*%\w+\s*[\+\-\*]', stripped_line):
            return True

        # Check for operation without result: op_type operands
        if re.match(r'[a-z_][a-z0-9_\.]*\s', stripped_line):
            return True

        # Stand-alone operation names like "return" or "scf.yield"
        if re.match(r'[a-z_][a-z0-9_\.]*$', stripped_line):
            return True

        return False

    def _line_opens_region(self, stripped_line: str, current_op_lines: List[str]) -> bool:
        """Check if the current line opens a region body for scf.for etc.

        A region is opened when:
        1. The line ends with '{'
        2. The operation is a control flow op (scf.for, scf.if)

        We check the accumulated op lines to see if the operation is scf.for.
        """
        if not stripped_line.endswith('{'):
            return False

        # Look through accumulated lines for the operation type
        op_text = ' '.join(current_op_lines)
        # scf.for opens a region
        if 'scf.for ' in op_text or op_text.startswith('scf.for'):
            return True
        if 'scf.if ' in op_text or op_text.startswith('scf.if'):
            return True
        # linalg.generic: the region opens on the outs(...) { line, not on
        # the first { which is the inline attribute block.
        if 'linalg.generic' in op_text and 'outs(' in op_text:
            return True
        return False

    def _extract_region_from_lines(self, lines: List[str], open_brace_line: int) -> Tuple[Optional[str], int]:
        """Extract a region body starting from the line that has the opening '{'.

        The opening '{' is at the end of lines[open_brace_line].
        We scan forward to find the matching '}', tracking brace depth.
        We return the text between the braces and the line index of the
        closing '}'.

        Returns:
            (region_body_text, closing_brace_line_index) or (None, -1)
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
                        # Found the closing brace. Everything before this
                        # closing brace on this line is part of the body.
                        # But typically the '}' is on its own line.
                        before_close = stripped[:stripped.rfind('}')].strip()
                        if before_close:
                            body_lines.append(before_close)
                        return ('\n'.join(body_lines), i)

            body_lines.append(line)
            i += 1

        return (None, -1)

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

