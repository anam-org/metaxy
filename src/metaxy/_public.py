"""Public API decorator for documentation whitelisting."""

from __future__ import annotations

from typing import TypeVar

_T = TypeVar("_T")


def public(obj: _T) -> _T:
    """Mark an API as public.

    This decorator sets a `__metaxy_public__` attribute on the decorated object,
    which is detected by the Griffe extension during documentation generation.

    Objects decorated with `@public` will:
    - Appear in API documentation when explicitly referenced
    - Be visible when their parent module is documented

    Objects without `@public` will:
    - Be hidden from documentation even if implicitly included via module documentation
    - Cause the docs build to fail if explicitly referenced via `::: identifier`

    Args:
        obj: The class, function, or other object to mark as public.

    Returns:
        The same object, unchanged except for the `__metaxy_public__` attribute.

    Example:
        ```python
        from metaxy import public


        @public
        class MyFeature(BaseFeature):
            '''This feature will appear in documentation.'''

            pass


        @public
        def my_function():
            '''This function will appear in documentation.'''
            pass
        ```
    """
    obj.__metaxy_public__ = True  # type: ignore[attr-defined]
    return obj
