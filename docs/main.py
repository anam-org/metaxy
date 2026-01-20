"""MkDocs macros for Metaxy documentation."""

from __future__ import annotations

from pathlib import Path


def define_env(env):
    """Define custom macros for MkDocs."""

    @env.macro
    def child_pages() -> str:
        """Generate a list of direct child pages for the current page.

        Returns:
            Markdown string with links to all direct child pages.

        Usage in markdown:
            {{ child_pages() }}
        """
        # Get the current page path
        page = env.page
        if not page:
            return ""

        page_path = Path(page.file.src_path)
        page_dir = page_path.parent

        # Get the docs directory
        docs_dir = Path(env.conf["docs_dir"])
        current_dir = docs_dir / page_dir

        if not current_dir.exists():
            return ""

        children = []

        # Find all direct child directories that have an index.md
        for item in sorted(current_dir.iterdir()):
            if item.is_dir():
                index_file = item / "index.md"
                if index_file.exists():
                    # Get the title from the first H1 heading in the file
                    title = _get_page_title(index_file) or item.name.replace("-", " ").title()
                    rel_path = f"./{item.name}/index.md"
                    children.append(f"- [{title}]({rel_path})")

        # Find sibling .md files (excluding index.md)
        for item in sorted(current_dir.iterdir()):
            if item.is_file() and item.suffix == ".md" and item.name != "index.md":
                title = _get_page_title(item) or item.stem.replace("-", " ").title()
                rel_path = f"./{item.name}"
                children.append(f"- [{title}]({rel_path})")

        if not children:
            return ""

        return "\n".join(children)


def _get_page_title(file_path: Path) -> str | None:
    """Extract the title (first H1) from a markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    return line[2:].strip()
    except Exception:
        pass
    return None
