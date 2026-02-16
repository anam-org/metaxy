"""Griffe extension for @public decorator detection.

This extension controls API visibility in documentation based on the presence
of the `@public` decorator from `metaxy._decorators`.

Objects decorated with `@public` will appear in documentation.
Objects without the decorator will be removed from the Griffe tree,
hiding them from module documentation.
"""

from __future__ import annotations

import griffe


def _has_public_decorator(obj: griffe.Object) -> bool:
    """Check if object has @public decorator.

    Args:
        obj: The Griffe object to check.

    Returns:
        True if the object has the @public decorator.
    """
    if not hasattr(obj, "decorators"):
        return False

    for decorator in obj.decorators:
        if decorator.callable_path == "metaxy._decorators.public":
            return True

    return False


class PublicAPIExtension(griffe.Extension):
    """Griffe extension that removes non-public objects from documentation.

    Objects without the @public decorator are deleted from the Griffe tree,
    preventing them from appearing in module documentation.
    """

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        """Process the entire package after loading to remove non-public members."""
        self._prune_non_public(pkg)

    def _prune_non_public(self, module: griffe.Module) -> None:
        """Recursively remove non-public members from a module.

        Args:
            module: The module to prune.
        """
        # Collect members to remove (can't modify dict during iteration)
        to_remove: list[str] = []

        for name, member in module.members.items():
            # Recurse into submodules first
            if isinstance(member, griffe.Module):
                self._prune_non_public(member)
            # Remove non-public classes and functions
            elif isinstance(member, griffe.Class | griffe.Function):
                if not _has_public_decorator(member):
                    to_remove.append(name)

        # Remove non-public members
        for name in to_remove:
            del module.members[name]
