"""Griffe extension for @experimental decorator detection.

This extension adds warning admonitions to experimental APIs in documentation
based on the presence of the `@experimental` decorator from `metaxy._decorators`.
"""

from __future__ import annotations

import griffe

EXPERIMENTAL_ADMONITION = """\
!!! warning "Experimental"
    This functionality is experimental.

"""


def _has_experimental_decorator(obj: griffe.Object) -> bool:
    """Check if object has @experimental decorator.

    Args:
        obj: The Griffe object to check.

    Returns:
        True if the object has the @experimental decorator.
    """
    if not hasattr(obj, "decorators"):
        return False

    for decorator in obj.decorators:
        if decorator.callable_path == "metaxy._decorators.experimental":
            return True

    return False


class ExperimentalAPIExtension(griffe.Extension):
    """Griffe extension that adds warning admonitions to experimental APIs.

    Objects with the @experimental decorator get a warning admonition
    prepended to their docstring.
    """

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        """Process the entire package after loading to add experimental warnings."""
        self._add_experimental_warnings(pkg)

    def _add_experimental_warnings(self, module: griffe.Module) -> None:
        """Recursively add experimental warnings to module members.

        Args:
            module: The module to process.
        """
        for member in module.members.values():
            if isinstance(member, griffe.Module):
                self._add_experimental_warnings(member)
            elif isinstance(member, griffe.Class | griffe.Function):
                if _has_experimental_decorator(member):
                    self._prepend_warning(member)

    def _prepend_warning(self, obj: griffe.Class | griffe.Function) -> None:
        """Prepend experimental warning admonition to object's docstring.

        Args:
            obj: The object to add the warning to.
        """
        if obj.docstring is None:
            obj.docstring = griffe.Docstring(EXPERIMENTAL_ADMONITION.strip())
        else:
            obj.docstring.value = EXPERIMENTAL_ADMONITION + obj.docstring.value
