"""Sybil conftest for docstring code block testing.

Configures Sybil to parse and test markdown code blocks (```python and ```py)
found in docstrings throughout the src/metaxy codebase.

Import style enforcement:
    All docstring examples must use `import metaxy as mx` instead of
    `from metaxy import ...`. The `mx` module is pre-populated in the
    namespace along with common symbols for convenience.
"""

from doctest import ELLIPSIS

from sybil import Sybil
from sybil.document import PythonDocStringDocument
from sybil.evaluators.python import PythonEvaluator
from sybil.parsers.markdown.codeblock import CodeBlockParser
from sybil.parsers.markdown.skip import SkipParser
from sybil.parsers.rest.doctest import DocTestParser

import metaxy as mx


# Workaround for pytest-cases compatibility issue with SybilItem.
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


class ImportStyleCheckingEvaluator(PythonEvaluator):
    """PythonEvaluator that checks import style before execution."""

    def __call__(self, example):
        # Check for forbidden import style
        if "from metaxy import" in example.parsed:
            return (
                "Use 'import metaxy as mx' instead of 'from metaxy import'.\n"
                "The 'mx' module is pre-populated in the doctest namespace."
            )
        # Delegate to parent for actual execution
        return super().__call__(example)


def sybil_setup(namespace):
    """Set up an isolated FeatureGraph for each docstring document.

    This provides a clean graph context so examples don't need boilerplate.
    Examples can define features and they'll be registered to this isolated graph.

    Pre-populated symbols:
        - mx: The metaxy module (import metaxy as mx)
        - graph: The active FeatureGraph instance for this document
        - MyFeature: A pre-defined feature class with key "my/feature"
        - store: An empty DeltaMetadataStore in a temporary directory
        - store_with_data: A DeltaMetadataStore with sample MyFeature data
        - config: A MetaxyConfig with "dev" and "prod" stores configured
    """
    from metaxy_testing.doctest_fixtures import (
        DocsStoreFixtures,
        MyFeature,
        register_doctest_fixtures,
    )

    from metaxy.models import feature as feature_module
    from metaxy.models.feature import FeatureGraph

    # Create isolated graph and enter its context
    isolated_graph = FeatureGraph()
    context_manager = isolated_graph.use()
    context_manager.__enter__()

    # Register pre-defined fixtures to the isolated graph
    register_doctest_fixtures(isolated_graph)

    # Set up store fixtures
    store_fixtures = DocsStoreFixtures(isolated_graph)
    store_fixtures.setup()

    # IMPORTANT: Override the module-level 'graph' variable so docstring examples
    # in feature.py use our isolated graph instead of the global singleton.
    # We restore it in teardown.
    namespace["_original_module_graph"] = feature_module.graph
    feature_module.graph = isolated_graph

    # Internal state
    namespace["_sybil_graph"] = isolated_graph
    namespace["_sybil_context_manager"] = context_manager
    namespace["_store_fixtures"] = store_fixtures

    # Pre-populated symbols for examples
    namespace["mx"] = mx
    namespace["graph"] = isolated_graph
    namespace["MyFeature"] = MyFeature
    namespace["store"] = store_fixtures.store
    namespace["store_with_data"] = store_fixtures.store_with_data
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


# SkipParser must come before other parsers to handle skip directives.
python_evaluator = ImportStyleCheckingEvaluator()
parsers = [
    SkipParser(),
    DocTestParser(optionflags=ELLIPSIS),
    CodeBlockParser(language="python", evaluator=python_evaluator),
    CodeBlockParser(language="py", evaluator=python_evaluator),
]

# Configure Sybil to extract docstrings from Python files and parse code blocks
# Uses relative patterns since this conftest is in src/metaxy/
pytest_collect_file = Sybil(
    parsers=parsers,
    patterns=["**/*.py"],
    document_types={".py": PythonDocStringDocument},
    setup=sybil_setup,
    teardown=sybil_teardown,
    excludes=[
        # Sybil uses PurePath.match() which requires patterns to match from the right side
        "ext/mcp/*",  # MCP module has import conflict with 'mcp' package
        "*/mcp/*",  # Deeper nested mcp files
        "_testing/*",  # Internal testing utilities - examples not meant to be executed
        "_testing/*/*",  # Nested _testing subdirectories (parametric, etc)
        "ext/dagster/*",  # Dagster examples require dagster context
        "*/dagster/*",  # Deeper nested dagster files
        "ext/ray/*",  # Ray examples require specific setup
        "*/ray/*",  # Deeper nested ray files
        "metadata_store/system/*",  # System storage examples need store setup
        "*/system/*",  # Deeper nested system files
        "conftest.py",  # Don't test this file itself
    ],
).pytest()
