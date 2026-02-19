"""Griffe extensions for Pydantic config documentation.

1. Hides single-value Literal discriminator fields from griffe-pydantic rendering.
2. Propagates inherited attributes to child classes so mkdocstrings can resolve
   ``ChildClass.inherited_field`` in per-attribute `:::` blocks.
"""

from __future__ import annotations

import copy
import re

import griffe

# Matches Literal['value'] or Literal["value"] with exactly one value
_SINGLE_LITERAL_RE = re.compile(r"""^Literal\[['"]([^'"]+)['"]\]$""")


class ConfigDiscriminatorExtension(griffe.Extension):
    """Remove single-value Literal attributes and propagate inherited members."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        self._process_module(pkg)

    def _process_module(self, module: griffe.Module) -> None:
        for member in module.members.values():
            if isinstance(member, griffe.Module):
                self._process_module(member)
            elif isinstance(member, griffe.Class):
                self._propagate_inherited_attributes(member)
                self._prune_discriminators(member)

    def _propagate_inherited_attributes(self, cls: griffe.Class) -> None:
        """Copy inherited attributes into the class as real objects, not aliases."""
        try:
            inherited = cls.inherited_members
        except Exception:
            return

        for name, member in inherited.items():
            if name in cls.members:
                continue
            # Resolve aliases to get the actual object, then copy it into the class
            target = member.target if isinstance(member, griffe.Alias) else member
            if not isinstance(target, griffe.Attribute):
                continue
            attr_copy = copy.copy(target)
            attr_copy.parent = cls
            cls.set_member(name, attr_copy)

    def _prune_discriminators(self, cls: griffe.Class) -> None:
        to_remove: list[str] = []
        for name, member in cls.members.items():
            if not isinstance(member, griffe.Attribute):
                continue
            if member.annotation is None:
                continue
            annotation_str = str(member.annotation)
            if _SINGLE_LITERAL_RE.match(annotation_str):
                to_remove.append(name)
        for name in to_remove:
            del cls.members[name]
