from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from mkdocs.structure.files import Files

log = logging.getLogger("mkdocs")

SLIDES_DIR = Path(__file__).parent / "slides"
SLIDES_ENTRY = "slides-introduction.md"
SLIDES_OUTPUT = "dist"
HIDDEN_NODE_MODULES = SLIDES_DIR / ".node_modules"
EXCLUDED_DOCS = {
    "paper.md",
    "slides/README.md",
    f"slides/{SLIDES_ENTRY}",
}


def _run(command: list[str]) -> None:
    """Run a subprocess command with error handling."""
    subprocess.run(command, check=True, cwd=SLIDES_DIR)


def _restore_node_modules() -> None:
    """Ensure the Slidev node_modules directory is available for Slidev commands."""
    node_modules = SLIDES_DIR / "node_modules"
    if HIDDEN_NODE_MODULES.exists():
        if node_modules.exists():
            shutil.rmtree(node_modules)
        HIDDEN_NODE_MODULES.rename(node_modules)


def _hide_node_modules() -> None:
    """Hide the Slidev node_modules directory so MkDocs doesn't try to process it."""
    node_modules = SLIDES_DIR / "node_modules"
    if not node_modules.exists():
        return
    if HIDDEN_NODE_MODULES.exists():
        shutil.rmtree(HIDDEN_NODE_MODULES)
    node_modules.rename(HIDDEN_NODE_MODULES)


def on_pre_build(config) -> None:  # pragma: no cover - executed by MkDocs
    """Build the Slidev deck before MkDocs collects site files."""
    _restore_node_modules()

    entry_path = SLIDES_DIR / SLIDES_ENTRY

    if not entry_path.exists():
        log.debug("Slidev entry %s not found, skipping Slidev build.", entry_path)
        _hide_node_modules()
        return

    bun = shutil.which("bun")
    if bun is None:
        log.warning("Skipping Slidev build because `bun` executable is not available.")
        _hide_node_modules()
        return

    node_modules = SLIDES_DIR / "node_modules"
    try:
        if not node_modules.exists():
            log.info("Installing Slidev dependencies with `bun install`.")
            _run([bun, "install"])

        output_dir = SLIDES_DIR / SLIDES_OUTPUT
        if output_dir.exists():
            shutil.rmtree(output_dir)

        log.info("Building Slidev slides into %s.", output_dir)
        _run(
            [
                bun,
                "x",
                "slidev",
                "build",
                SLIDES_ENTRY,
                "--out",
                SLIDES_OUTPUT,
                "--base",
                "./",
            ]
        )
    finally:
        _hide_node_modules()


def on_files(files: Files, config) -> Files:  # pragma: no cover - executed by MkDocs
    """Exclude Slidev dependencies from the MkDocs file set."""
    filtered_files = Files(
        [
            f
            for f in files
            if not f.src_path.replace("\\", "/").startswith("slides/node_modules")
            and f.src_path not in EXCLUDED_DOCS
        ]
    )

    removed = len(files) - len(filtered_files)
    if removed:
        log.debug(
            "Excluded %d files under slides/node_modules from MkDocs build.", removed
        )

    return filtered_files


def on_post_build(config) -> None:  # pragma: no cover - executed by MkDocs
    """Restore node_modules after MkDocs is finished."""
    _restore_node_modules()
