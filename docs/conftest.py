"""Sybil conftest for testing markdown code blocks in documentation.

Configures Sybil to parse and test markdown code blocks (```python and ```py)
found in the docs/ directory.

Code blocks using --8<-- snippet includes are expanded before evaluation so
docs execute the same example source that MkDocs renders.
"""

import re
from pathlib import Path

from metaxy.models.feature import FeatureGraph
from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.abstract.codeblock import AbstractCodeBlockParser
from sybil.parsers.markdown.lexers import (
    DirectiveInHTMLCommentLexer,
    RawFencedCodeBlockLexer,
)
from sybil.parsers.markdown.skip import SkipParser
from sybil.typing import Evaluator

import metaxy as mx

REPO_ROOT = Path(__file__).resolve().parent.parent
SNIPPET_BASE_PATHS = (
    REPO_ROOT / "examples",
    REPO_ROOT / "docs" / "snippets",
    REPO_ROOT / "docs",
)


# Workaround for pytest-cases compatibility issue
# pytest-cases patches getfixtureclosure and has an assertion that fails for SybilItem
# because SybilItem doesn't have the same structure as regular test functions.
# We patch pytest-cases to skip its assertion for non-function items.
def _patch_pytest_cases():
    try:
        import pytest_cases.plugin as pc_plugin
    except ImportError:
        return

    if not hasattr(pc_plugin, "_getfixtureclosure"):
        return

    original_getfixtureclosure = pc_plugin._getfixtureclosure

    def patched_getfixtureclosure(
        fm, fixturenames, parentnode, ignore_args=frozenset()
    ):
        # Check if this is a SybilItem or similar non-function item
        # These don't have proper funcargs and cause assertion failures in pytest-cases
        from sybil.integration.pytest import SybilItem

        if isinstance(parentnode, SybilItem):
            # Bypass pytest-cases entirely for Sybil items - use pytest's native implementation
            if pc_plugin.PYTEST8_OR_GREATER:
                return fm.__class__.getfixtureclosure(
                    fm, parentnode, fixturenames, ignore_args=ignore_args
                )
            elif pc_plugin.PYTEST46_OR_GREATER:
                return fm.__class__.getfixtureclosure(
                    fm, fixturenames, parentnode, ignore_args=ignore_args
                )
            else:
                return fm.__class__.getfixtureclosure(fm, fixturenames, parentnode)

        return original_getfixtureclosure(fm, fixturenames, parentnode, ignore_args)

    pc_plugin._getfixtureclosure = patched_getfixtureclosure  # type: ignore[invalid-assignment]

    # Also patch the wrapper if it exists
    if pc_plugin.PYTEST8_OR_GREATER:

        def patched_wrapper(fm, parentnode, initialnames, ignore_args):
            return patched_getfixtureclosure(
                fm,
                fixturenames=initialnames,
                parentnode=parentnode,
                ignore_args=ignore_args,
            )

        pc_plugin.getfixtureclosure = patched_wrapper  # type: ignore[invalid-assignment]


_patch_pytest_cases()


class FencedCodeBlockWithAttributesLexer(RawFencedCodeBlockLexer):
    """A lexer that handles fenced code blocks with optional attributes.

    Supports patterns like:
    - ```python
    - ```python {title="test.py"}
    - ```py {title="test.py" hl_lines="1"}
    """

    def __init__(self) -> None:
        # Match language (word chars only) followed by optional attributes
        # The language is captured separately from any attributes
        super().__init__(
            info_pattern=re.compile(
                r"(?P<language>\w+)(?:[^\n]*)?\n",
                re.MULTILINE,
            ),
            # Map 'language' to 'arguments' as expected by AbstractCodeBlockParser
            mapping={"language": "arguments", "source": "source"},
        )


class CodeBlockWithAttributesParser(AbstractCodeBlockParser):
    """CodeBlockParser that handles MkDocs-style attributes like {title="..."}.

    Extends the default CodeBlockParser to support fenced code blocks with
    additional attributes that MkDocs uses for styling.
    """

    def __init__(
        self, language: str | None = None, evaluator: Evaluator | None = None
    ) -> None:
        super().__init__(
            [
                FencedCodeBlockWithAttributesLexer(),
                DirectiveInHTMLCommentLexer(
                    directive=r"(invisible-)?code(-block)?",
                    arguments=".+",
                ),
            ],
            language,
            evaluator,
        )


class SnippetExpandingEvaluator(PythonEvaluator):
    """PythonEvaluator that expands MkDocs snippet includes before execution."""

    SNIPPET_PATTERN = re.compile(r'^(?P<indent>\s*)--8<--\s*"(?P<target>[^"]+)"\s*$')
    SECTION_START_PATTERN = re.compile(r"#\s*--8<--\s*\[start:(?P<name>[^\]]+)\]\s*$")
    SECTION_END_PATTERN = re.compile(r"#\s*--8<--\s*\[end:(?P<name>[^\]]+)\]\s*$")

    def _resolve_snippet_path(self, target_path: str, *, document_path: Path) -> Path:
        candidate_paths = []

        if Path(target_path).is_absolute():
            candidate_paths.append(Path(target_path))
        else:
            candidate_paths.append((document_path.parent / target_path).resolve())
            candidate_paths.extend(
                (base_path / target_path).resolve() for base_path in SNIPPET_BASE_PATHS
            )

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(
            f"Unable to resolve snippet path {target_path!r}; searched: {searched}"
        )

    def _extract_section(
        self, source_lines: list[str], section_name: str, snippet_path: Path
    ) -> list[str]:
        start_index: int | None = None

        for index, line in enumerate(source_lines):
            start_match = self.SECTION_START_PATTERN.match(line)
            if start_match and start_match.group("name") == section_name:
                start_index = index + 1
                break

        if start_index is None:
            raise ValueError(
                f"Snippet section {section_name!r} not found in {snippet_path}"
            )

        for index in range(start_index, len(source_lines)):
            end_match = self.SECTION_END_PATTERN.match(source_lines[index])
            if end_match and end_match.group("name") == section_name:
                return source_lines[start_index:index]

        raise ValueError(
            f"Snippet section {section_name!r} was not closed in {snippet_path}"
        )

    def _extract_range(
        self,
        source_lines: list[str],
        start_line: int,
        end_line: int,
        snippet_path: Path,
    ) -> list[str]:
        if start_line < 1 or end_line < start_line:
            raise ValueError(
                f"Invalid snippet line range {start_line}:{end_line} in {snippet_path}"
            )
        if end_line > len(source_lines):
            raise ValueError(
                f"Snippet line range {start_line}:{end_line} exceeds {snippet_path} length {len(source_lines)}"
            )
        return source_lines[start_line - 1 : end_line]

    def _read_snippet(self, target: str, *, document_path: Path) -> str:
        line_range_match = re.fullmatch(
            r"(?P<path>.+):(?P<start>\d+):(?P<end>\d+)", target
        )

        if line_range_match:
            snippet_path = self._resolve_snippet_path(
                line_range_match.group("path"), document_path=document_path
            )
            source_lines = snippet_path.read_text(encoding="utf-8").splitlines()
            snippet_lines = self._extract_range(
                source_lines,
                int(line_range_match.group("start")),
                int(line_range_match.group("end")),
                snippet_path,
            )
            return "\n".join(snippet_lines)

        path_part, separator, suffix = target.rpartition(":")

        if separator and path_part:
            snippet_path = self._resolve_snippet_path(
                path_part, document_path=document_path
            )
            source_lines = snippet_path.read_text(encoding="utf-8").splitlines()
            snippet_lines = self._extract_section(source_lines, suffix, snippet_path)
            return "\n".join(snippet_lines)

        snippet_path = self._resolve_snippet_path(target, document_path=document_path)
        return snippet_path.read_text(encoding="utf-8")

    def _expand_snippets(self, source: str, *, document_path: Path) -> str:
        expanded_lines: list[str] = []

        for line in source.splitlines():
            match = self.SNIPPET_PATTERN.match(line)
            if not match:
                expanded_lines.append(line)
                continue

            indent = match.group("indent")
            snippet_source = self._read_snippet(
                match.group("target"), document_path=document_path
            )
            snippet_lines = snippet_source.splitlines()
            expanded_lines.extend(
                f"{indent}{snippet_line}" if snippet_line else ""
                for snippet_line in snippet_lines
            )

        expanded_source = "\n".join(expanded_lines)
        if source.endswith("\n"):
            expanded_source += "\n"
        return expanded_source

    def __call__(self, example):
        example.parsed = type(example.parsed)(
            self._expand_snippets(
                str(example.parsed), document_path=Path(example.path)
            ),
            example.parsed.offset,
            example.parsed.line_offset,
        )
        return super().__call__(example)


def sybil_setup(namespace):
    """Set up an isolated FeatureGraph for each markdown document.

    Pre-populated symbols:
        - mx: The metaxy module (import metaxy as mx)
        - nw: The narwhals module
        - graph: The active FeatureGraph instance for this document
        - store: A DeltaMetadataStore with sample MyFeature data
        - empty_store: An empty DeltaMetadataStore in a temporary directory
        - config: A MetaxyConfig with "dev" and "prod" stores configured
        - MyFeature: A pre-defined feature class for examples

    Examples must use `import metaxy as mx` and access classes via `mx.FeatureSpec`,
    `mx.BaseFeature`, etc. Direct imports like `from metaxy import FeatureSpec` are
    banned in documentation examples.
    """
    import narwhals as nw
    import polars as pl
    from metaxy.models import feature as feature_module
    from metaxy_testing.doctest_fixtures import (
        ChildFeature,
        DocsStoreFixtures,
        MyFeature,
        ParentFeature,
        register_doctest_fixtures,
        sample_data,
    )

    # Create isolated graph and enter its context
    isolated_graph = FeatureGraph()
    context_manager = isolated_graph.use()
    context_manager.__enter__()

    # Register pre-defined fixtures to the isolated graph
    register_doctest_fixtures(isolated_graph)

    # Set up store fixtures
    store_fixtures = DocsStoreFixtures(isolated_graph)
    store_fixtures.setup()

    # Override the module-level 'graph' variable
    namespace["_original_module_graph"] = feature_module.graph
    feature_module.graph = isolated_graph

    # Internal state
    namespace["_sybil_graph"] = isolated_graph
    namespace["_sybil_context_manager"] = context_manager
    namespace["_store_fixtures"] = store_fixtures

    # Pre-populated symbols for examples
    # Only `mx` and `nw` are allowed - examples must use `import metaxy as mx`
    # Direct imports like `from metaxy import FeatureSpec` are banned in docs
    namespace["mx"] = mx
    namespace["nw"] = nw
    namespace["pl"] = pl
    namespace["__name__"] = "docs"
    namespace["__package__"] = "docs"
    namespace["graph"] = isolated_graph
    namespace["MyFeature"] = MyFeature
    namespace["ParentFeature"] = ParentFeature
    namespace["ChildFeature"] = ChildFeature
    namespace["df"] = sample_data
    # `store` is pre-populated with MyFeature data for examples that need existing data
    namespace["store"] = store_fixtures.store_with_data
    namespace["empty_store"] = store_fixtures.store
    namespace["config"] = store_fixtures.config


def sybil_teardown(namespace):
    """Clean up the isolated FeatureGraph context and store fixtures."""
    # Clean up store fixtures
    store_fixtures = namespace.get("_store_fixtures")
    if store_fixtures is not None:
        store_fixtures.teardown()

    # Restore the original module-level graph
    original_graph = namespace.get("_original_module_graph")
    if original_graph is not None:
        from metaxy.models import feature as feature_module

        feature_module.graph = original_graph

    context_manager = namespace.get("_sybil_context_manager")
    if context_manager:
        context_manager.__exit__(None, None, None)


# Create parsers for both ```python and ```py code blocks
# SkipParser must come before other parsers to handle skip directives
# Use CodeBlockWithAttributesParser to handle MkDocs-style attributes like {title="..."}
python_evaluator = SnippetExpandingEvaluator()
parsers = [
    SkipParser(),
    CodeBlockWithAttributesParser(language="python", evaluator=python_evaluator),
    CodeBlockWithAttributesParser(language="py", evaluator=python_evaluator),
]

# Configure Sybil to parse markdown files in docs/
_sybil_collect_file = Sybil(
    parsers=parsers,
    patterns=["**/*.md"],
    setup=sybil_setup,
    teardown=sybil_teardown,
    excludes=[
        # Integration pages that require external infrastructure
        "integrations/metadata-stores/databases/clickhouse.md",  # Requires ClickHouse server
        "integrations/metadata-stores/databases/bigquery.md",  # Requires BigQuery
        "integrations/compute/ray.md",  # Requires Ray cluster
        "integrations/ai/*",  # AI integrations
        # Plugin pages with complex setup requirements
        "integrations/plugins/sqlalchemy.md",  # Requires Alembic context
        "integrations/plugins/sqlmodel.md",  # Requires SQLModel setup
        # Slides use Slidev magic-move syntax with illustrative code blocks
        "slides/**",
        "docs/slides/**",
        "**/slides/**",
        # Internal docs
        ".mkdocs-metaxy/*",
    ],
).pytest()


def pytest_collect_file(file_path, parent):
    """Skip Slidev markdown docs and delegate all other markdown to Sybil."""
    path_str = str(file_path).replace("\\", "/")
    if "/slides/" in path_str and path_str.endswith(".md"):
        return None
    return _sybil_collect_file(file_path, parent)
