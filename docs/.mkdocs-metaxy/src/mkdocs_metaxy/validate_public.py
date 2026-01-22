"""MkDocs hook to validate that explicitly documented APIs have @public decorator.

This hook scans markdown files for `::: identifier` directives and verifies
that each referenced Python object has the `@public` decorator. If any
documented API lacks the decorator, the build fails with `--strict` mode.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import griffe

log = logging.getLogger("mkdocs.hooks.validate_public")

# Pattern to match mkdocstrings directives like `::: metaxy.SomeClass`
# Excludes custom directives like `metaxy-config`, `metaxy-mcp-tools`, `metaxy-example`
MKDOCSTRINGS_PATTERN = re.compile(r"^::: ([\w.]+)$", re.MULTILINE)

# Identifiers that are not Python objects (custom markdown extensions)
SKIP_IDENTIFIERS = frozenset(
    {
        "metaxy-config",
        "metaxy-mcp-tools",
        "metaxy-example",
    }
)


def _has_public_decorator(obj: griffe.Object) -> bool:
    """Check if a Griffe object has the @public decorator.

    Args:
        obj: The Griffe object to check.

    Returns:
        True if the object has @public decorator, False otherwise.
    """
    if not hasattr(obj, "decorators"):
        # Modules, attributes, etc. don't have decorators
        # They are considered public if explicitly documented
        return True

    for decorator in obj.decorators:
        if decorator.callable_path in ("metaxy._public.public",):
            return True

    return False


def _is_module_documentation(identifier: str, obj: griffe.Object) -> bool:
    """Check if this is module-level documentation (entire module).

    Args:
        identifier: The documented identifier string.
        obj: The resolved Griffe object.

    Returns:
        True if documenting an entire module, False otherwise.
    """
    return isinstance(obj, griffe.Module)


def on_files(files: list, config: dict) -> list:
    """MkDocs hook called after files are collected.

    Validates that all explicitly documented APIs have @public decorator.

    Args:
        files: List of MkDocs File objects.
        config: MkDocs configuration dictionary.

    Returns:
        The unmodified files list.
    """
    # Load the metaxy package for inspection
    loader = griffe.GriffeLoader(search_paths=["src"])

    try:
        loader.load("metaxy")
    except griffe.LoadingError as e:
        log.warning(f"Could not load metaxy package for validation: {e}")
        return files

    # Also load extension packages
    for ext_name in ("metaxy.ext", "metaxy.metadata_store"):
        try:
            loader.load(ext_name)
        except griffe.LoadingError:
            pass

    errors: list[tuple[str, str]] = []  # (file_path, identifier)

    for file in files:
        if not file.src_path.endswith(".md"):
            continue

        file_path = Path(config["docs_dir"]) / file.src_path
        if not file_path.exists():
            continue

        content = file_path.read_text()

        for match in MKDOCSTRINGS_PATTERN.finditer(content):
            identifier = match.group(1)

            # Skip non-Python identifiers
            if identifier in SKIP_IDENTIFIERS or not identifier.startswith("metaxy"):
                continue

            # Resolve the identifier using the modules collection
            try:
                obj = loader.modules_collection[identifier]
            except KeyError:
                # Object not found - this will be caught by mkdocstrings itself
                continue

            # Modules are always allowed (they document their public members)
            if _is_module_documentation(identifier, obj):
                continue

            # Check for @public decorator
            if not _has_public_decorator(obj):
                errors.append((file.src_path, identifier))

    if errors:
        log.error("The following documented APIs are missing the @public decorator:")
        for file_path, identifier in sorted(errors):
            log.error(f"  {file_path}: {identifier}")
        log.error(
            "\nAdd the @public decorator from metaxy._public to these objects, or remove them from the documentation."
        )
        # This will cause mkdocs build --strict to fail
        raise SystemExit(1)

    return files
