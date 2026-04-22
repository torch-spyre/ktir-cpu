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
KTIR dialect handler registry.

Importing this package registers every dialect handler via side-effect
imports of the individual dialect modules.
"""

from . import (  # noqa: F401 — imported for registration side effects
    arith_ops,
    ktdp_ops,
    linalg_ops,
    math_ops,
    scf_ops,
    tensor_ops,
)
from .registry import ExecutionEnv, ParseContext, dispatch, dispatch_parser, make_parse_context
