from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager


class SourceBuilder:
    """Small indentation-aware builder for inspectable generated source."""

    def __init__(self, indent: str = "    "):
        self._indent = indent
        self._level = 0
        self._lines: list[str] = []

    def line(self, text: str = "") -> None:
        if text:
            self._lines.append(f"{self._indent * self._level}{text}")
        else:
            self._lines.append("")

    def lines(self, values: Iterable[str]) -> None:
        for value in values:
            self.line(value)

    @contextmanager
    def block(self, header: str):
        self.line(f"{header}:")
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    @contextmanager
    def indent(self):
        self._level += 1
        try:
            yield
        finally:
            self._level -= 1

    def render(self) -> str:
        return "\n".join(self._lines)


def emit_triton_imports(builder: SourceBuilder) -> None:
    builder.lines(
        [
            "import triton",
            "import triton.language as tl",
            "",
            "",
        ]
    )


def emit_triton_function_header(builder: SourceBuilder, function_name: str, signature_args: Iterable[str]) -> None:
    builder.lines(
        [
            "@triton.jit",
            f"def {function_name}(",
        ]
    )
    args = tuple(signature_args)
    for idx, arg in enumerate(args):
        suffix = "," if idx < len(args) - 1 else ""
        builder.line(f"    {arg}{suffix}")
    builder.line("):")


def emit_triton_header(builder: SourceBuilder, function_name: str, signature_args: Iterable[str]) -> None:
    emit_triton_imports(builder)
    emit_triton_function_header(builder, function_name, signature_args)
