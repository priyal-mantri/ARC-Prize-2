"""Program Representation, Composition, and Execution Engine for ARC-AGI-2 DSL.

Supports typed sequential composition of primitives, execution on ArcGrid and NumPy arrays,
complexity measurement, serialization, and deserialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.representation.grid import ArcGrid
from src.dsl.primitives import PRIMITIVE_REGISTRY, DSL_REGISTRY

logger = logging.getLogger(__name__)


class PrimitiveCall:
    """Represents a single invocation of a DSL primitive with bound arguments."""

    def __init__(
        self,
        primitive_name: str,
        params: Optional[Dict[str, Any]] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Union[Dict[str, Any], Tuple[Tuple[str, Any], ...]]] = None,
    ):
        self.primitive_name = primitive_name
        self.args = args

        # Normalize kwargs from params or kwargs dict/tuple
        kw_dict: Dict[str, Any] = {}
        if isinstance(kwargs, dict):
            kw_dict.update(kwargs)
        elif isinstance(kwargs, (tuple, list)):
            kw_dict.update(dict(kwargs))

        if isinstance(params, dict):
            kw_dict.update(params)

        self.kwargs = tuple(sorted(kw_dict.items()))
        self.params = kw_dict

    @property
    def name(self) -> str:
        return self.primitive_name

    @property
    def complexity(self) -> int:
        """Complexity weight: 1 for primitive + 1 for each bound argument."""
        return 1 + len(self.args) + len(self.kwargs)

    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Executes this primitive call on the input grid array."""
        if self.primitive_name not in PRIMITIVE_REGISTRY:
            raise KeyError(f"Unknown primitive '{self.primitive_name}'")
        func = PRIMITIVE_REGISTRY[self.primitive_name]
        kw = dict(self.kwargs)
        return func(grid, *self.args, **kw)

    def to_code(self) -> str:
        """Returns string representation of this call."""
        arg_strs = [repr(a) for a in self.args]
        kw_strs = [f"{k}={repr(v)}" for k, v in self.kwargs]
        all_args = ", ".join(arg_strs + kw_strs)
        if all_args:
            return f"{self.primitive_name}(g, {all_args})"
        return f"{self.primitive_name}(g)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive": self.primitive_name,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimitiveCall":
        prim = data.get("primitive", data.get("primitive_name", "identity"))
        args = tuple(data.get("args", ()))
        params = data.get("params", data.get("kwargs", {}))
        return cls(primitive_name=prim, args=args, params=params)

    def __repr__(self) -> str:
        return self.to_code()


class DSLProgram:
    """Composable sequential pipeline of DSL operations."""

    def __init__(
        self,
        name: str = "DSLProgram",
        steps: Optional[List[PrimitiveCall]] = None,
        calls: Optional[List[PrimitiveCall]] = None,
        origin: str = "unknown",
    ):
        self.name = name
        self.steps: List[PrimitiveCall] = steps or calls or []
        self.origin = origin

    @property
    def calls(self) -> List[PrimitiveCall]:
        return self.steps

    def add_step(self, step: PrimitiveCall) -> None:
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def complexity(self) -> int:
        """Total program complexity: sum of step complexities."""
        return sum(s.complexity for s in self.steps)

    def execute(self, grid: Union[ArcGrid, np.ndarray]) -> Union[ArcGrid, np.ndarray]:
        """Executes all program steps in sequential pipeline order.

        Accepts either ArcGrid or np.ndarray; returns the corresponding type.
        """
        is_arc_grid = isinstance(grid, ArcGrid)
        curr = grid.to_numpy() if is_arc_grid else np.array(grid, copy=True)

        for step in self.steps:
            curr = step.execute(curr)
            if not isinstance(curr, np.ndarray):
                curr = np.array(curr, dtype=np.int32)

        if is_arc_grid:
            return ArcGrid(np_grid=curr)
        return curr

    def to_code(self) -> str:
        """Produces a standalone Python lambda expression."""
        if not self.steps:
            return "lambda g: g"
        expr = "g"
        for s in self.steps:
            arg_strs = [repr(a) for a in s.args]
            kw_strs = [f"{k}={repr(v)}" for k, v in s.kwargs]
            args_str = ", ".join(arg_strs + kw_strs)
            if args_str:
                expr = f"{s.primitive_name}({expr}, {args_str})"
            else:
                expr = f"{s.primitive_name}({expr})"
        return f"lambda g: {expr}"

    def to_lambda_code(self) -> str:
        return self.to_code()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "steps": [s.to_dict() for s in self.steps],
            "complexity": self.complexity,
            "lambda_code": self.to_code(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSLProgram":
        name = data.get("name", "DSLProgram")
        origin = data.get("origin", "unknown")
        steps = [PrimitiveCall.from_dict(s) for s in data.get("steps", [])]
        return cls(name=name, steps=steps, origin=origin)

    def __repr__(self) -> str:
        return f"DSLProgram({self.name}, steps={self.length}, complexity={self.complexity}, code='{self.to_code()}')"


class LambdaProgram:
    """Wrapper around an arbitrary callable / lambda function."""

    def __init__(
        self,
        name_or_func: Union[str, Callable[[np.ndarray], np.ndarray]],
        func_or_none: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        source_code: str = "",
    ):
        if callable(name_or_func):
            self.name = "lambda_prog"
            self.func = name_or_func
        else:
            self.name = str(name_or_func)
            self.func = func_or_none if func_or_none else (lambda g: g)

        self.source_code = source_code or getattr(self.func, "_source", getattr(self.func, "__doc__", "lambda g: ..."))
        self.complexity = max(1, self.source_code.count("(") if self.source_code else 2)

    def execute(self, grid: Union[ArcGrid, np.ndarray]) -> Union[ArcGrid, np.ndarray]:
        is_arc_grid = isinstance(grid, ArcGrid)
        arr = grid.to_numpy() if is_arc_grid else np.array(grid, copy=True)
        out = self.func(arr)
        if not isinstance(out, np.ndarray):
            out = np.array(out, dtype=np.int32)
        if is_arc_grid:
            return ArcGrid(np_grid=out)
        return out

    def __call__(self, grid: Union[ArcGrid, np.ndarray]) -> Union[ArcGrid, np.ndarray]:
        return self.execute(grid)

    def to_code(self) -> str:
        return self.source_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_code": self.source_code,
            "complexity": self.complexity,
        }

    def __repr__(self) -> str:
        return f"LambdaProgram(name='{self.name}', complexity={self.complexity})"
