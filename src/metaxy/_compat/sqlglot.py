"""Compatibility shims for `sqlglot`.

Some sqlglot releases raise `ValueError` when aliases/identifiers are provided
as `bytes`. Certain versions of Ibis/sqlglot on Linux occasionally emit byte
aliases, which caused CI-only failures for Postgres metadata store tests.
This module patches sqlglot to gracefully decode byte aliases before the
original code runs.
"""

from __future__ import annotations

from typing import Any

_patched_sqlglot = False


def _decode_alias(alias: Any) -> Any:
    """Convert bytes aliases to strings expected by sqlglot."""
    if isinstance(alias, (bytes, bytearray)):
        return alias.decode()
    return alias


def _patch_expression_alias() -> None:
    """Monkey patch `Expression.alias` to accept byte aliases."""
    global _patched_sqlglot
    if _patched_sqlglot:
        return

    try:
        from sqlglot import expressions as exp
    except ImportError:
        return

    original_alias_func = exp.alias_

    def alias_(expression, alias=None, *args, **kwargs):
        return original_alias_func(expression, _decode_alias(alias), *args, **kwargs)

    exp.alias_ = alias_  # type: ignore[assignment]
    _patched_sqlglot = True


_patch_expression_alias()

__all__: list[str] = []
