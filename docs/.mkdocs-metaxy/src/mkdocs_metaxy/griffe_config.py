"""Griffe extension for hiding single-value Literal discriminator fields.

Pydantic discriminated unions use ``type: Literal["value"]`` fields as
discriminators. These are implementation details and should not appear in
rendered configuration documentation. This extension removes them from
the griffe tree so griffe-pydantic does not render them.
"""

from __future__ import annotations

import re

import griffe

# Matches Literal['value'] or Literal["value"] with exactly one value
_SINGLE_LITERAL_RE = re.compile(r"""^Literal\[['"]([^'"]+)['"]\]$""")


class ConfigDiscriminatorExtension(griffe.Extension):
    """Remove single-value Literal attributes from pydantic model classes."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: object) -> None:
        self._process_module(pkg)

    def _process_module(self, module: griffe.Module) -> None:
        for member in module.members.values():
            if isinstance(member, griffe.Module):
                self._process_module(member)
            elif isinstance(member, griffe.Class):
                self._prune_discriminators(member)

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
