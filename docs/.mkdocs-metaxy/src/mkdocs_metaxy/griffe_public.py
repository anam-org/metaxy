"""Griffe extension for @public decorator detection.

This extension controls API visibility in documentation based on the presence
of the `@public` decorator from `metaxy._public`.

Objects decorated with `@public` will have `obj.public = True`, making them
visible in documentation. Objects without the decorator will have
`obj.public = False`, hiding them from implicit module documentation.
"""

from __future__ import annotations

import griffe


class PublicAPIExtension(griffe.Extension):
    """Griffe extension that detects the @public decorator and sets visibility."""

    def on_class_instance(self, *, cls: griffe.Class, **kwargs: object) -> None:
        """Process class instances to check for @public decorator."""
        self._check_public(cls)

    def on_function_instance(self, *, func: griffe.Function, **kwargs: object) -> None:
        """Process function instances to check for @public decorator."""
        self._check_public(func)

    def _check_public(self, obj: griffe.Object) -> None:
        """Check if object has @public decorator and set visibility accordingly.

        Args:
            obj: The Griffe object to check.
        """
        if not hasattr(obj, "decorators"):
            return

        for decorator in obj.decorators:
            # Check for the @public decorator from metaxy._public
            if decorator.callable_path in ("metaxy._public.public",):
                obj.public = True
                return

        # Objects without @public are marked as non-public
        # This hides them from implicit module documentation
        obj.public = False
