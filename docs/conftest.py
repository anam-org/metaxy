"""Sybil conftest for testing markdown code blocks in documentation.

Configures Sybil to parse and test markdown code blocks (```python and ```py)
found in the docs/ directory.

Code blocks using --8<-- snippet includes are automatically skipped since
they are tested via their source files.
"""

import re

from sybil import Sybil
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.abstract.codeblock import AbstractCodeBlockParser
from sybil.parsers.markdown.lexers import DirectiveInHTMLCommentLexer, RawFencedCodeBlockLexer
from sybil.parsers.markdown.skip import SkipParser
from sybil.typing import Evaluator

import metaxy as mx
from metaxy.models.feature import FeatureGraph


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

    def patched_getfixtureclosure(fm, fixturenames, parentnode, ignore_args=frozenset()):
        # Check if this is a SybilItem or similar non-function item
        # These don't have proper funcargs and cause assertion failures in pytest-cases
        from sybil.integration.pytest import SybilItem

        if isinstance(parentnode, SybilItem):
            # Bypass pytest-cases entirely for Sybil items - use pytest's native implementation
            if pc_plugin.PYTEST8_OR_GREATER:
                return fm.__class__.getfixtureclosure(fm, parentnode, fixturenames, ignore_args=ignore_args)
            elif pc_plugin.PYTEST46_OR_GREATER:
                return fm.__class__.getfixtureclosure(fm, fixturenames, parentnode, ignore_args=ignore_args)
            else:
                return fm.__class__.getfixtureclosure(fm, fixturenames, parentnode)

        return original_getfixtureclosure(fm, fixturenames, parentnode, ignore_args)

    pc_plugin._getfixtureclosure = patched_getfixtureclosure  # type: ignore[invalid-assignment]

    # Also patch the wrapper if it exists
    if pc_plugin.PYTEST8_OR_GREATER:

        def patched_wrapper(fm, parentnode, initialnames, ignore_args):
            return patched_getfixtureclosure(
                fm, fixturenames=initialnames, parentnode=parentnode, ignore_args=ignore_args
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

    def __init__(self, language: str | None = None, evaluator: Evaluator | None = None) -> None:
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


class SnippetSkippingEvaluator(PythonEvaluator):
    """PythonEvaluator that skips code blocks containing --8<-- snippet includes."""

    # Pattern to detect snippet includes
    SNIPPET_PATTERN = re.compile(r'--8<--\s*"')

    def __call__(self, example):
        # Skip code blocks that contain snippet includes
        if self.SNIPPET_PATTERN.search(example.parsed):
            return None  # Skip this example
        # Delegate to parent for actual execution
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

    from metaxy._testing.doctest_fixtures import (
        DocsStoreFixtures,
        MyFeature,
        register_doctest_fixtures,
    )
    from metaxy.models import feature as feature_module

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
    namespace["graph"] = isolated_graph
    namespace["MyFeature"] = MyFeature
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
python_evaluator = SnippetSkippingEvaluator()
parsers = [
    SkipParser(),
    CodeBlockWithAttributesParser(language="python", evaluator=python_evaluator),
    CodeBlockWithAttributesParser(language="py", evaluator=python_evaluator),
]

# Configure Sybil to parse markdown files in docs/
pytest_collect_file = Sybil(
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
        # Internal docs
        ".mkdocs-metaxy/*",
    ],
).pytest()
