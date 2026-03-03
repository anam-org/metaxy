from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from metaxy.config import MetaxyConfig
from mkdocs.structure.files import Files

log = logging.getLogger("mkdocs")

DOCS_DIR = Path(__file__).parent
ROOT_DIR = DOCS_DIR.parent
SLIDES_DIR = ROOT_DIR / "slides" / "2026-introducing-metaxy"
PUBLICATIONS_DIR = ROOT_DIR / "publications" / "2026-introducing-metaxy"
DOCS_SLIDES_DIR = DOCS_DIR / "slides" / "2026-introducing-metaxy"
DOCS_PUBLICATIONS_DIR = DOCS_DIR / "publications" / "2026-introducing-metaxy"
SLIDES_ENTRY = "introducing-metaxy.md"
SLIDES_OUTPUT = "dist"
HIDDEN_NODE_MODULES = SLIDES_DIR / ".node_modules"
PUBLICATION_ASSETS_DIR = (
    DOCS_DIR / "assets" / "publications" / "2026-introducing-metaxy"
)
SLIDES_PUBLIC_IMG_DIR = SLIDES_DIR / "public" / "img"
SHARED_SLIDES_ASSET_SOURCES = {
    "anatomy.svg": PUBLICATION_ASSETS_DIR / "anatomy.svg",
    "feature.svg": PUBLICATION_ASSETS_DIR / "feature.svg",
    "pipeline.svg": PUBLICATION_ASSETS_DIR / "pipeline.svg",
    "metaxy.svg": DOCS_DIR / "assets" / "metaxy.svg",
    "coffee.jpg": DOCS_DIR
    / "assets"
    / "slides"
    / "2026-introducing-metaxy"
    / "coffee.jpg",
    "race.jpg": DOCS_DIR / "assets" / "slides" / "2026-introducing-metaxy" / "race.jpg",
}
EXCLUDED_DOCS = {
    "slides/2026-introducing-metaxy/README.md",
    f"slides/2026-introducing-metaxy/{SLIDES_ENTRY}",
}


def _ensure_metaxy_config() -> None:
    """Ensure MkDocs runs with a configured MetaxyConfig to silence warnings."""
    if not MetaxyConfig.is_set():
        MetaxyConfig.set(MetaxyConfig(project="docs"))


_ensure_metaxy_config()


def _run(command: list[str]) -> None:
    """Run a subprocess command with error handling."""
    subprocess.run(command, check=True, cwd=SLIDES_DIR)


def _remove_stale_slide_root_index() -> None:
    """Drop stale Slidev root index.html that conflicts with README.md -> index.html."""
    stale_index = SLIDES_DIR / "index.html"
    if stale_index.exists() and stale_index.is_file():
        stale_index.unlink()


def _sync_docs_mounts() -> None:
    """Mirror top-level slides/publications into docs/ for MkDocs file discovery."""
    mounts = (
        (SLIDES_DIR, DOCS_SLIDES_DIR),
        (PUBLICATIONS_DIR, DOCS_PUBLICATIONS_DIR),
    )
    for src, dst in mounts:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.copytree(
            src,
            dst,
            symlinks=False,
            ignore=shutil.ignore_patterns("node_modules", ".node_modules"),
        )


def _sync_slides_svg_assets() -> None:
    """Keep shared Slidev assets aligned by symlinking with copy fallback."""
    SLIDES_PUBLIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

    for name, src in SHARED_SLIDES_ASSET_SOURCES.items():
        dst = SLIDES_PUBLIC_IMG_DIR / name

        if not src.exists():
            log.warning("Skipping missing shared Slidev asset source: %s", src)
            continue

        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        try:
            relative_src = Path(os.path.relpath(src, start=dst.parent))
            dst.symlink_to(relative_src)
        except OSError:
            shutil.copy2(src, dst)


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


def _ensure_dependencies(bun_path: str | None, npm_path: str | None) -> None:
    """Install Slidev dependencies if node_modules is missing."""
    node_modules = SLIDES_DIR / "node_modules"
    if node_modules.exists():
        return

    if bun_path:
        log.info("Installing Slidev dependencies with `bun install`.")
        _run([bun_path, "install"])
        return

    if npm_path:
        log.info("Installing Slidev dependencies with `npm install`.")
        _run([npm_path, "install"])
        return

    raise RuntimeError(
        "Unable to install Slidev dependencies: neither `bun` nor `npm` is available."
    )


def _select_runner(bun_path: str | None, npx_path: str | None) -> list[str]:
    """Choose the command used to build the Slidev deck."""
    slidev_args = [
        "slidev",
        "build",
        SLIDES_ENTRY,
        "--out",
        SLIDES_OUTPUT,
        "--base",
        "./",
    ]

    prefer_npx = os.environ.get("METAXY_SLIDES_RUNNER", "").lower() == "npx"

    if prefer_npx and npx_path:
        log.info("Using npx to build Slidev slides (runner=%s).", npx_path)
        return [npx_path, *slidev_args]

    if bun_path:
        log.info(
            "Using bun to build Slidev slides via package script (runner=%s).", bun_path
        )
        return [
            bun_path,
            "run",
            "build",
            "--",
            *slidev_args[2:],  # pass through slide-specific arguments
        ]

    if npx_path:
        log.info("Fallback to npx to build Slidev slides (runner=%s).", npx_path)
        return [npx_path, *slidev_args]

    raise RuntimeError(
        "Neither `bun` nor `npx` is available to build Slidev slides. "
        "Install bun or Node.js (providing npx) to proceed."
    )


def on_pre_build(config) -> None:  # pragma: no cover - executed by MkDocs
    """Build the Slidev deck before MkDocs collects site files."""
    _remove_stale_slide_root_index()
    _sync_slides_svg_assets()
    _sync_docs_mounts()
    _restore_node_modules()

    entry_path = SLIDES_DIR / SLIDES_ENTRY
    if not entry_path.exists():
        log.debug("Slidev entry %s not found, skipping Slidev build.", entry_path)
        _hide_node_modules()
        return

    if os.environ.get("METAXY_SKIP_SLIDEV"):
        log.info(
            "METAXY_SKIP_SLIDEV detected; skipping Slidev build and preserving existing artifacts."
        )
        dist_index = SLIDES_DIR / SLIDES_OUTPUT / "index.html"
        if not dist_index.exists():
            log.warning(
                "Slidev dist output missing at %s; ensure slides are built before skipping.",
                dist_index,
            )
        _hide_node_modules()
        return

    bun = shutil.which("bun")
    npx = shutil.which("npx")
    npm = shutil.which("npm")

    if bun is None and npx is None:
        log.warning(
            "Skipping Slidev build because neither `bun` nor `npx` is available."
        )
        _hide_node_modules()
        return

    try:
        log.info(
            "Slidev build environment: CI=%s, bun=%s, npx=%s, npm=%s",
            os.environ.get("CI"),
            bun,
            npx,
            npm,
        )
        _ensure_dependencies(bun_path=bun, npm_path=npm)

        output_dir = SLIDES_DIR / SLIDES_OUTPUT
        if output_dir.exists():
            shutil.rmtree(output_dir)

        log.info("Building Slidev slides into %s.", output_dir)
        runner = _select_runner(bun_path=bun, npx_path=npx)
        _run(runner)
        _sync_docs_mounts()
    finally:
        _hide_node_modules()


def on_files(files: Files, config) -> Files:  # pragma: no cover - executed by MkDocs
    """Exclude Slidev dependencies from the MkDocs file set."""
    filtered_files = Files(
        [
            f
            for f in files
            if not f.src_path.replace("\\", "/").startswith(
                "slides/2026-introducing-metaxy/node_modules"
            )
            and f.src_path not in EXCLUDED_DOCS
        ]
    )

    removed = len(files) - len(filtered_files)
    if removed:
        log.debug(
            "Excluded %d files under slides/2026-introducing-metaxy/node_modules from MkDocs build.",
            removed,
        )

    return filtered_files


def on_post_build(config) -> None:  # pragma: no cover - executed by MkDocs
    """Restore node_modules after MkDocs is finished."""
    _restore_node_modules()
