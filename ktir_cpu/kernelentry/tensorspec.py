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

"""The arguments a kernel is called with, written as data rather than as code.

Every kernel in this repository is driven by the same few tensor shapes: a normal
draw, a zeroed output, a broadcast constant, a tiled row, an index vector, and the
scalars the signature declares.  Written as a function per kernel, that vocabulary
was re-spelled eighteen times and each spelling could differ in ways nothing
checked — the seed, the dtype, whether the output was zeroed.  Written as a table,
declaring a kernel is a row, and the ways two kernels can differ are the ways the
vocabulary allows.

A value in an entry's ``tensors`` mapping is one of four things:

* a **spec** from this module — ``normal``, ``zeros``, ``full``, ``tile``,
  ``arange``, ``integers``, ``asarray``, ``param``;
* a **string**, naming a parameter to forward to the kernel unchanged;
* any other **literal**, forwarded as it stands;
* a **callable** ``(params, rng) -> value``, for input a spec cannot express.

The last is the escape hatch, and it is per argument rather than per kernel: RoPE
needs its cos/sin tables built from angles, and gets to declare those two the long
way while its other two arguments stay rows in the table.

Shapes are resolved against the entry's parameters: an ``int`` is itself, a
``str`` is a parameter name, and a callable is evaluated on the parameters for a
dimension that is an expression of them.

**Determinism is a requirement, not a convenience.**  The reference comparison and
the committed cost derivation each rebuild the arguments, and neither may depend on
which ran first, so every random draw comes from a generator seeded by the argument
name.  Seeding per argument rather than per kernel is also what keeps two arguments
of the same kernel independent: one generator drawn twice gives the second tensor
the first one's values wherever their shapes overlap, and a kernel that swapped its
two operands would then still agree with its reference.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

import numpy as np

#: One seed for the whole ledger.  Which value it is does not matter; that it is
#: written down once, and that no declaration gets to choose its own, does.
SEED = 42

_DTYPES = {
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
    "i32": np.int32, "i64": np.int64,
}

Dim = Union[int, str, Callable[[Mapping[str, Any]], int]]
Shape = Union[Dim, Sequence[Dim]]


def _rng(name: str) -> np.random.Generator:
    """A generator for one argument, stable across runs, hosts and orderings."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])


def dtype_of(name: str) -> np.dtype:
    """The numpy dtype a spec's ``dtype=`` string names."""
    try:
        return np.dtype(_DTYPES[name])
    except KeyError:
        raise ValueError(
            f"unknown dtype {name!r}; use one of {', '.join(sorted(_DTYPES))}"
        ) from None


def _scalar(value: Any, params: Mapping[str, Any]) -> Any:
    """A spec's own scalar argument: a parameter name, or a literal."""
    return params[value] if isinstance(value, str) else value


def _shape(shape: Shape, params: Mapping[str, Any]) -> Tuple[int, ...]:
    dims = shape if isinstance(shape, (tuple, list)) else (shape,)
    out = []
    for dim in dims:
        if isinstance(dim, str):
            out.append(int(params[dim]))
        elif callable(dim):
            out.append(int(dim(params)))
        else:
            out.append(int(dim))
    return tuple(out)


#: ``scale=`` for a weight matrix initialised the way the layer would be, which
#: keeps every f16 intermediate downstream of it inside range.  Named rather than
#: written out because the reason for it is the same wherever it appears.
FAN_IN: Callable[[Tuple[int, ...]], float] = lambda shape: 1.0 / math.sqrt(shape[0])


@dataclass(frozen=True)
class normal:
    """A standard normal draw, optionally scaled."""

    shape: Shape
    dtype: str = "f16"
    scale: Union[float, Callable[[Tuple[int, ...]], float]] = 1.0

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        shape = _shape(self.shape, params)
        scale = self.scale(shape) if callable(self.scale) else self.scale
        return (rng.standard_normal(shape) * scale).astype(dtype_of(self.dtype))


@dataclass(frozen=True)
class zeros:
    """A zeroed output tensor."""

    shape: Shape
    dtype: str = "f16"

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        return np.zeros(_shape(self.shape, params), dtype=dtype_of(self.dtype))


@dataclass(frozen=True)
class full:
    """One value everywhere.  *value* may name a parameter."""

    shape: Shape
    value: Any
    dtype: str = "f16"

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        return np.full(_shape(self.shape, params), _scalar(self.value, params),
                       dtype=dtype_of(self.dtype))


@dataclass(frozen=True)
class tile:
    """*row*, repeated — a vector a kernel views at matrix shape.

    The repetition is what makes the reference able to say the tensor is
    position-dependent along one axis and constant along the other.
    """

    row: Any
    reps: Shape

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        return np.tile(self.row(params, rng), _shape(self.reps, params))


@dataclass(frozen=True)
class arange:
    """Consecutive values over *shape*, so a fold's result is hand-checkable."""

    shape: Shape
    start: Any = 0
    dtype: str = "f16"

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        shape = _shape(self.shape, params)
        start = int(_scalar(self.start, params))
        size = int(np.prod(shape)) if shape else 0
        return np.arange(start, start + size,
                         dtype=dtype_of(self.dtype)).reshape(shape)


@dataclass(frozen=True)
class integers:
    """A draw from ``[low, high)`` — an index tensor, not a value tensor.

    *low* and *high* may name parameters.  Drawn over the whole range rather than
    set to identity on purpose: indices that happen to equal their position make
    an indirect access indistinguishable from a direct one.
    """

    shape: Shape
    high: Any
    low: Any = 0
    dtype: str = "i32"

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        return rng.integers(int(_scalar(self.low, params)),
                            int(_scalar(self.high, params)),
                            size=_shape(self.shape, params),
                            dtype=dtype_of(self.dtype))


@dataclass(frozen=True)
class asarray:
    """A literal sequence, or a parameter holding one, as a tensor."""

    value: Any
    dtype: str = "i64"

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
        return np.asarray(_scalar(self.value, params), dtype=dtype_of(self.dtype))


@dataclass(frozen=True)
class param:
    """A scalar argument, at a width the kernel reads it as.

    A bare string in the mapping forwards a parameter as the Python value it is;
    this is for the arguments where the width is part of the signature — an ``i32``
    extent is not an index constant, and the interpreter reads the two differently.
    """

    name: str
    dtype: str = ""

    def __call__(self, params: Mapping[str, Any],
                 rng: np.random.Generator) -> Any:
        value = params[self.name]
        return dtype_of(self.dtype).type(value) if self.dtype else value


def build_tensors(specs: Mapping[str, Any],
                 params: Mapping[str, Any]) -> Dict[str, Any]:
    """The keyword arguments to call the kernel with.

    Called once per run and again for whatever needs a pristine copy: a kernel
    whose output argument aliases its input writes over what it was given, and a
    reference reading that would be comparing the result against itself.
    """
    built: Dict[str, Any] = {}
    for name, spec in specs.items():
        if isinstance(spec, str):
            built[name] = params[spec]
        elif callable(spec):
            built[name] = spec(params, _rng(name))
        else:
            built[name] = spec
    return built


def validate_specs(specs: Mapping[str, Any], params: Mapping[str, Any],
                   where: str) -> None:
    """Reject a mapping that cannot be built, at declaration time.

    A parameter name misspelled in a spec would otherwise surface as a ``KeyError``
    from inside the engine, on the one kernel, at the moment it ran — which reads
    as a fault in the tool rather than a typo in the row.
    """
    for name, spec in specs.items():
        wanted = [spec] if isinstance(spec, str) else []
        if isinstance(spec, param):
            wanted = [spec.name]
        for key in wanted:
            if key not in params:
                raise ValueError(
                    f"{where}: tensors[{name!r}] names parameter {key!r}, which "
                    f"is not in gate_params ({', '.join(sorted(params))})"
                )


__all__ = ["FAN_IN", "SEED", "arange", "asarray", "build_tensors", "dtype_of",
           "full", "integers", "normal", "param", "tile", "validate_specs",
           "zeros"]
