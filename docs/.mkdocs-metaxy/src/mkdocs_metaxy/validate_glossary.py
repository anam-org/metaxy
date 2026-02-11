"""MkDocs hook to validate glossary link targets exist.

Parses docs/.glossary.md for URLs in abbreviation definitions
(format: ``*[term]: Description | /path/to/docs``) and warns when
a linked page is missing. Causes ``mkdocs build --strict`` to fail.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.validate_glossary")

GLOSSARY_LINK_PATTERN = re.compile(r"^\*\[.*?\]:.*\|\s*(.+)$", re.MULTILINE)


def on_files(files: list, config: dict) -> list:
    """Validate that glossary link targets resolve to existing doc pages."""
    glossary_path = Path(config["docs_dir"]) / ".glossary.md"

    if not glossary_path.exists():
        return files

    content = glossary_path.read_text()

    known_paths: set[str] = set()
    for f in files:
        if not f.src_path.endswith(".md"):
            continue
        # "guide/concepts/data-versioning.md" -> "guide/concepts/data-versioning"
        path = f.src_path.removesuffix(".md")
        known_paths.add(path)
        # "guide/concepts/index" -> "guide/concepts"
        if path.endswith("/index"):
            known_paths.add(path.removesuffix("/index"))

    for match in GLOSSARY_LINK_PATTERN.finditer(content):
        url = match.group(1).strip()
        line_num = content[: match.start()].count("\n") + 1
        path = url.split("#")[0].strip("/")

        if not path:
            continue

        if path not in known_paths:
            log.warning(f".glossary.md:{line_num}: glossary link target not found: '{url}'")

    return files
