"""Compatibility shims for `sqlglot`.

Some sqlglot releases raise `ValueError` when aliases/identifiers are provided
as `bytes`. Certain versions of Ibis/sqlglot on Linux occasionally emit byte
aliases or identifiers, which caused CI-only failures for Postgres metadata store tests.

This module patches sqlglot to gracefully decode byte identifiers before the
original code runs.
"""

from __future__ import annotations

from typing import Any

_patched_sqlglot = False


def _decode_if_bytes(value: Any) -> Any:
    """Convert bytes values to strings expected by sqlglot."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


def _patch_sqlglot_bytes_handling() -> None:
    """Monkey patch sqlglot functions to accept byte identifiers."""
    global _patched_sqlglot
    if _patched_sqlglot:
        return
    try:
        from sqlglot import expressions as exp
    except ImportError:
        return

    # Patch 1: Expression.alias
    original_alias_func = exp.alias_

    def alias_(expression, alias=None, *args, **kwargs):
        return original_alias_func(expression, _decode_if_bytes(alias), *args, **kwargs)

    exp.alias_ = alias_  # type: ignore[assignment]

    # Patch 2: to_identifier (This is the fix for your CI failure)
    original_to_identifier_func = exp.to_identifier

    def to_identifier(name, quoted=None, copy=True):
        return original_to_identifier_func(
            _decode_if_bytes(name), quoted=quoted, copy=copy
        )

    exp.to_identifier = to_identifier  # type: ignore[assignment]

    _patched_sqlglot = True


_patch_sqlglot_bytes_handling()


__all__: list[str] = []
