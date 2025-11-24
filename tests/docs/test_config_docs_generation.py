"""Tests for config documentation generation."""

from __future__ import annotations

from mkdocs_metaxy.config.markdown_ext import MetaxyConfigPreprocessor
from mkdocs_metaxy.config_generator import (
    extract_field_info,
    generate_individual_field_doc,
)

from metaxy._testing.config import SamplePluginConfig


def test_generate_toml_code_block() -> None:
    """Test the helper method for generating TOML code blocks."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = 'store = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)

    # Join lines to see full output
    result = "\n".join(lines)

    # Should have blank line before fence
    assert lines[0] == ""

    # Should have opening fence with proper indentation
    assert lines[1] == "    ```toml"

    # Should have blank line after opening fence
    assert lines[2] == ""

    # Should have content with proper indentation
    assert lines[3] == '    store = "dev"'

    # Should have closing fence
    assert lines[4] == "    ```"

    # Should have blank line after fence
    assert lines[5] == ""

    # Should NOT have literal \n in the string
    assert r"\n" not in result


def test_generate_toml_code_block_multiline() -> None:
    """Test code block generation with multiline content."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = '[tool.metaxy]\nstore = "dev"\nstores = {}'
    lines = generate_toml_code_block(content, indent_level=1)

    result = "\n".join(lines)

    # Should have all content lines properly indented
    assert "    [tool.metaxy]" in result
    assert '    store = "dev"' in result
    assert "    stores = {}" in result

    # Should have proper fences
    assert "    ```toml" in result
    assert result.count("    ```") == 2  # Opening and closing


def test_generate_toml_code_block_exact_output() -> None:
    """Test exact code block output as a single string."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = 'store = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)
    result = "\n".join(lines)

    # Expected exact output with all newlines and indentation
    # IMPORTANT: Blank line after opening fence is required for proper rendering!
    expected = """
    ```toml

    store = "dev"
    ```
"""

    assert result == expected, f"Got:\n{repr(result)}\n\nExpected:\n{repr(expected)}"

    # Verify structure more explicitly
    assert len(lines) == 6, f"Expected 6 lines, got {len(lines)}: {lines}"
    assert lines[0] == "", f"Line 0 should be empty, got: {repr(lines[0])}"
    assert lines[1] == "    ```toml", (
        f"Line 1 should be '    ```toml', got: {repr(lines[1])}"
    )
    assert lines[2] == "", (
        f"Line 2 should be empty (blank after fence), got: {repr(lines[2])}"
    )
    assert lines[3] == '    store = "dev"', (
        f"Line 3 should be content, got: {repr(lines[3])}"
    )
    assert lines[4] == "    ```", f"Line 4 should be '    ```', got: {repr(lines[4])}"
    assert lines[5] == "", f"Line 5 should be empty, got: {repr(lines[5])}"

    # Verify no content on same line as fence marker
    for i, line in enumerate(lines):
        if "```toml" in line:
            assert line.strip() == "```toml", (
                f"Line {i} has content after fence marker: {repr(line)}"
            )

    # Verify blank line immediately after opening fence
    fence_idx = next(i for i, line in enumerate(lines) if "```toml" in line)
    assert lines[fence_idx + 1] == "", (
        f"Line after opening fence should be blank, got: {repr(lines[fence_idx + 1])}"
    )


def test_generate_toml_code_block_multiline_exact_output() -> None:
    """Test exact multiline code block output as a single string."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = '[tool.metaxy]\nstore = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)
    result = "\n".join(lines)

    # Expected exact output - note the blank line after opening fence!
    expected = """
    ```toml

    [tool.metaxy]
    store = "dev"
    ```
"""

    assert result == expected, f"Got:\n{repr(result)}\n\nExpected:\n{repr(expected)}"

    # Verify blank line after opening fence
    assert lines[2] == "", (
        f"Line 2 should be blank after opening fence, got: {repr(lines[2])}"
    )

    # Verify [tool.metaxy] is not on same line as fence
    fence_line = next(line for line in lines if "```toml" in line)
    assert "[tool.metaxy]" not in fence_line, (
        "Section header should not be on same line as fence"
    )


def test_full_field_doc_exact_toml_blocks() -> None:
    """Test that a complete field doc has properly formatted TOML code blocks."""
    from mkdocs_metaxy.config_generator import (
        extract_field_info,
        generate_individual_field_doc,
    )

    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    doc = generate_individual_field_doc(enable_field, header_level=3)

    # Check that metaxy.toml tab has exact format WITH blank line after opening fence
    assert (
        '=== "metaxy.toml"\n\n    ```toml\n\n    enable = false\n    ```\n\n' in doc
    ), f"metaxy.toml block not formatted correctly. Full doc:\n{doc}"

    # Check that pyproject.toml tab has exact format WITH blank line after opening fence
    assert (
        '=== "pyproject.toml"\n\n    ```toml\n\n    [tool.metaxy]\n    enable = false\n    ```\n\n'
        in doc
    ), f"pyproject.toml block not formatted correctly. Full doc:\n{doc}"

    # Should NOT have any literal backslash-n sequences
    assert r"\n" not in doc
    assert "\\n" not in doc

    # Verify content is NOT on same line as fence
    assert "```toml\n    [tool.metaxy]" not in doc, (
        "Content should not be on same line as opening fence"
    )
    assert "```toml\n    enable" not in doc, (
        "Content should not be on same line as opening fence"
    )


def test_extract_field_info_metaxy_config() -> None:
    """Test extracting field info from MetaxyConfig."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)

    # Should have multiple fields
    assert len(fields) > 0

    # Should have store field
    store_fields = [f for f in fields if f["name"] == "store"]
    assert len(store_fields) == 1
    assert store_fields[0]["default"] == "dev"


def test_extract_field_info_sqlmodel_config() -> None:
    """Test extracting field info from SQLModelPluginConfig."""
    from metaxy.ext.sqlmodel import SQLModelPluginConfig

    fields = extract_field_info(SQLModelPluginConfig)

    # Should have fields
    assert len(fields) > 0

    # Should have enable field
    enable_fields = [f for f in fields if f["name"] == "enable"]
    assert len(enable_fields) == 1


def test_generate_field_doc_has_proper_code_fences() -> None:
    """Test that generated docs have properly formatted code fences."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)
    store_field = next(f for f in fields if f["name"] == "store")

    doc = generate_individual_field_doc(store_field, header_level=2)

    # Should have code fences
    assert "```toml" in doc
    assert "```" in doc

    # Should have tabs
    assert '=== "metaxy.toml"' in doc
    assert '=== "pyproject.toml"' in doc
    assert '=== "Environment Variable"' in doc

    # Should not have literal newlines in output (checking for the two characters \ and n)
    literal_backslash_n = r"\n"  # This is the literal two-character string
    assert literal_backslash_n not in doc, "Found literal backslash-n in output"

    # Should not have triple blank lines
    assert "\n\n\n" not in doc, "Found triple newlines"

    # Code fence should be on its own line
    lines = doc.split("\n")
    toml_fence_indices = [i for i, line in enumerate(lines) if "```toml" in line]
    assert len(toml_fence_indices) >= 2  # At least 2 tabs

    # Check that code fence has proper indentation (4 spaces for pymdownx.tabbed)
    for idx in toml_fence_indices:
        # Line should start with 4 spaces followed by ```toml
        assert lines[idx] == "    ```toml", f"Line {idx}: {repr(lines[idx])}"
        # Previous line should be blank or have less indentation
        assert lines[idx - 1].strip() == "", f"Line {idx - 1}: {repr(lines[idx - 1])}"


def test_sqlmodel_config_respects_env_prefix() -> None:
    """Test that SQLModel config uses correct env_prefix."""
    from metaxy.ext.sqlmodel import SQLModelPluginConfig

    # Get model_config (it's a dict in pydantic-settings)
    model_config = SQLModelPluginConfig.model_config
    if isinstance(model_config, dict):
        env_prefix = model_config.get("env_prefix", "METAXY_")
    else:
        env_prefix = getattr(model_config, "env_prefix", "METAXY_")

    # Should have custom prefix
    assert env_prefix == "METAXY_EXT__SQLMODEL_"

    # Generate docs
    fields = extract_field_info(SQLModelPluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    doc = generate_individual_field_doc(
        enable_field,
        env_prefix=env_prefix,
        header_level=2,
    )

    # Should use correct env prefix (note: field name gets uppercased and delimiter added)
    assert "METAXY_EXT__SQLMODEL_ENABLE" in doc
    # Should NOT have double prefix
    assert "METAXY_EXT__SQLMODEL_EXT__SQLMODEL" not in doc


def test_preprocessor_processes_directive() -> None:
    """Test that preprocessor correctly processes metaxy-config directive."""
    preprocessor = MetaxyConfigPreprocessor(None)

    # Test markdown with directive (no closing :::)
    markdown_lines = [
        "::: metaxy-config",
        "    class: metaxy.config.MetaxyConfig",
        "    header_level: 2",
        "",
        "Some text after",
    ]

    result = preprocessor.run(markdown_lines)
    result_text = "\n".join(result)

    # Should have generated field docs (field names are wrapped in backticks)
    assert "## `store`" in result_text or "## Store" in result_text
    assert "```toml" in result_text

    # Should preserve text after directive
    assert "Some text after" in result_text


def test_no_literal_backslash_n_in_output() -> None:
    """Test that there are no literal backslash-n in the generated markdown."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)

    literal_backslash_n = r"\n"  # Literal two-character string

    for field in fields:
        if field["is_nested"]:
            continue

        doc = generate_individual_field_doc(field, header_level=2)

        # Should not have literal newlines
        assert literal_backslash_n not in doc, (
            f"Field {field['name']} has literal backslash-n in output"
        )

        # Should not have triple blank lines
        assert "\n\n\n" not in doc, f"Field {field['name']} has triple newlines"


def test_preprocessor_handles_empty_directive() -> None:
    """Test that preprocessor handles directives with no fields gracefully."""
    preprocessor = MetaxyConfigPreprocessor(None)

    # Test markdown with directive but invalid class
    markdown_lines = [
        "::: metaxy-config",
        "    class: nonexistent.module.Class",
        "    header_level: 2",
        "",
        "Some text after",
    ]

    result = preprocessor.run(markdown_lines)
    result_text = "\n".join(result)

    # Should have error admonition
    assert "error" in result_text.lower()
    assert "failed to import" in result_text.lower()

    # Should preserve text after directive
    assert "Some text after" in result_text


def test_code_fence_indentation_consistency() -> None:
    """Test that all code fences have consistent indentation across tabs."""
    from metaxy.config import MetaxyConfig

    fields = extract_field_info(MetaxyConfig)
    store_field = next(f for f in fields if f["name"] == "store")

    doc = generate_individual_field_doc(store_field, header_level=2)
    lines = doc.split("\n")

    # Find all tab markers
    tab_indices = [i for i, line in enumerate(lines) if line.startswith('=== "')]

    for tab_idx in tab_indices:
        # Find the code fence after this tab
        fence_idx = None
        for i in range(tab_idx + 1, min(tab_idx + 5, len(lines))):
            if "```toml" in lines[i]:
                fence_idx = i
                break

        if fence_idx:
            # Check that the fence has 4-space indentation
            assert lines[fence_idx].startswith("    "), (
                f"Code fence at line {fence_idx} doesn't start with 4 spaces: {repr(lines[fence_idx])}"
            )

            # Check closing fence
            close_fence_idx = None
            for i in range(fence_idx + 1, min(fence_idx + 10, len(lines))):
                if lines[i].strip() == "```":
                    close_fence_idx = i
                    break

            if close_fence_idx:
                assert lines[close_fence_idx].startswith("    "), (
                    f"Closing fence at line {close_fence_idx} doesn't start with 4 spaces: {repr(lines[close_fence_idx])}"
                )


def test_sample_plugin_config_structure() -> None:
    """Test that SamplePluginConfig generates correct structure."""
    fields = extract_field_info(SamplePluginConfig)

    # Should have expected fields
    field_names = {f["name"] for f in fields if not f["is_nested"]}
    assert "enable" in field_names
    assert "name" in field_names
    assert "port" in field_names
    assert "debug" in field_names
    assert "optional_setting" in field_names


def test_sample_plugin_env_prefix() -> None:
    """Test that SamplePluginConfig uses correct env_prefix."""
    model_config = SamplePluginConfig.model_config
    # SettingsConfigDict is a TypedDict, so we access it like a dict
    if isinstance(model_config, dict):
        env_prefix = model_config.get("env_prefix", "METAXY_")
    else:
        env_prefix = getattr(model_config, "env_prefix", "METAXY_")

    assert env_prefix == "SAMPLE_PLUGIN_"

    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    doc = generate_individual_field_doc(
        enable_field,
        env_prefix=env_prefix,
        header_level=3,
    )

    # Should use correct prefix
    assert "SAMPLE_PLUGIN_ENABLE" in doc
    assert "SAMPLE_PLUGIN__ENABLE" not in doc  # No double underscore for top-level


def test_header_level_creates_correct_nesting() -> None:
    """Test that header levels create correct TOC nesting."""
    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    # Test with h3 (should nest under h2 parent)
    doc_h3 = generate_individual_field_doc(enable_field, header_level=3)
    assert doc_h3.startswith("### `enable`")

    # Test with h2 (would be top-level)
    doc_h2 = generate_individual_field_doc(enable_field, header_level=2)
    assert doc_h2.startswith("## `enable`")

    # Test with h4 (would nest under h3 parent)
    doc_h4 = generate_individual_field_doc(enable_field, header_level=4)
    assert doc_h4.startswith("#### `enable`")


def test_full_markdown_generation() -> None:
    """Test complete markdown generation for SamplePluginConfig."""
    preprocessor = MetaxyConfigPreprocessor(None)

    # Simulate a documentation page
    markdown_lines = [
        "# Sample Plugin Configuration",
        "",
        "## Configuration Options",
        "",
        "::: metaxy-config",
        "    class: metaxy._testing.config.SamplePluginConfig",
        "    header_level: 3",
        "",
        "## Additional Information",
        "",
        "More content here.",
    ]

    result = preprocessor.run(markdown_lines)
    result_text = "\n".join(result)

    # Should have the title
    assert "# Sample Plugin Configuration" in result_text

    # Should have generated field docs with h3 (field names are wrapped in backticks)
    assert "### `enable`" in result_text
    assert "### `name`" in result_text
    assert "### `port`" in result_text

    # Should have proper code fences
    assert "```toml" in result_text

    # Should have proper env vars with SAMPLE_PLUGIN_ prefix
    assert "SAMPLE_PLUGIN_ENABLE" in result_text
    assert "SAMPLE_PLUGIN_NAME" in result_text

    # Should preserve content after directive
    assert "## Additional Information" in result_text
    assert "More content here." in result_text


def test_markdown_ast_has_valid_toml_code_blocks() -> None:
    """Test that generated markdown produces valid TOML code blocks in AST."""
    from typing import Any, cast

    import mistune

    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    doc = generate_individual_field_doc(enable_field, header_level=3)

    # Parse markdown to AST - the 4-space indentation from pymdownx.tabbed
    # will cause mistune to parse these as indented code blocks
    markdown = mistune.create_markdown(renderer=None)
    tokens: list[dict[str, Any]] = cast(list[dict[str, Any]], markdown(doc))

    # Extract all code blocks from AST
    code_blocks = [token for token in tokens if token.get("type") == "block_code"]

    # Should have at least 2 code blocks (metaxy.toml and pyproject.toml tabs)
    assert len(code_blocks) >= 2, (
        f"Expected at least 2 code blocks, got {len(code_blocks)}"
    )

    # Check that code blocks contain TOML-like content
    for block in code_blocks:
        # Code block should have actual content (not empty)
        raw = block.get("raw", "").strip()
        assert len(raw) > 0, "Code block has no content"

        # With 4-space indent (pymdownx.tabbed), mistune treats fenced blocks as indented
        # The raw content will include the fence markers and actual content
        # or just the content if truly indented. Either way, check for TOML-like content.
        assert "=" in raw or "```toml" in raw, (
            f"Code block doesn't look TOML-related: {raw[:100]}"
        )


def test_markdown_ast_headers_at_correct_level() -> None:
    """Test that field headers are at the correct level in the AST."""
    from typing import Any, cast

    import mistune

    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    # Test h3
    doc_h3 = generate_individual_field_doc(enable_field, header_level=3)
    markdown = mistune.create_markdown(renderer=None)
    tokens_h3: list[dict[str, Any]] = cast(list[dict[str, Any]], markdown(doc_h3))

    # Find heading tokens
    headings_h3 = [token for token in tokens_h3 if token.get("type") == "heading"]
    assert len(headings_h3) > 0, "No headings found in h3 document"

    # First heading should be level 3
    first_heading = headings_h3[0]
    level = first_heading.get("attrs", {}).get("level")
    assert level == 3, f"Expected level 3, got {level}"

    # Heading text should be the field name (may be in code span due to backticks)
    heading_children = first_heading.get("children", [])
    assert len(heading_children) > 0, "Heading has no children"
    heading_text = "".join(
        child.get("raw", "")
        for child in heading_children
        if child.get("type") in ("text", "codespan")
    )
    assert "enable" in heading_text.lower(), (
        f"Heading doesn't contain field name: {heading_text}"
    )

    # Test h2
    doc_h2 = generate_individual_field_doc(enable_field, header_level=2)
    tokens_h2: list[dict[str, Any]] = cast(list[dict[str, Any]], markdown(doc_h2))
    headings_h2 = [token for token in tokens_h2 if token.get("type") == "heading"]
    level_h2 = headings_h2[0].get("attrs", {}).get("level")
    assert level_h2 == 2, f"Expected level 2, got {level_h2}"


def test_markdown_ast_full_document_structure() -> None:
    """Test the complete document structure using AST."""
    from typing import Any, cast

    import mistune

    preprocessor = MetaxyConfigPreprocessor(None)

    # Simulate a documentation page
    markdown_lines = [
        "# Configuration",
        "",
        "## Fields",
        "",
        "::: metaxy-config",
        "    class: metaxy._testing.config.SamplePluginConfig",
        "    header_level: 3",
        "",
        "## More Info",
        "",
        "Additional content.",
    ]

    result = preprocessor.run(markdown_lines)
    result_text = "\n".join(result)

    # Parse to AST
    markdown = mistune.create_markdown(renderer=None)
    tokens: list[dict[str, Any]] = cast(list[dict[str, Any]], markdown(result_text))

    # Extract structure
    headings = [token for token in tokens if token.get("type") == "heading"]
    code_blocks = [token for token in tokens if token.get("type") == "block_code"]

    # Should have:
    # - h1: Configuration
    # - h2: Fields
    # - h3: enable, name, port, debug, optional_setting (5 fields)
    # - h2: More Info
    assert len(headings) >= 7, f"Expected at least 7 headings, got {len(headings)}"

    # First heading should be h1
    level_0 = headings[0].get("attrs", {}).get("level")
    assert level_0 == 1, f"Expected h1, got h{level_0}"

    # Second heading should be h2 (Fields)
    level_1 = headings[1].get("attrs", {}).get("level")
    assert level_1 == 2, f"Expected h2, got h{level_1}"

    # Next 5 headings should be h3 (field names)
    for i in range(2, 7):
        level = headings[i].get("attrs", {}).get("level")
        assert level == 3, f"Heading {i} should be h3, got h{level}"

    # Last heading should be h2 (More Info)
    level_7 = headings[7].get("attrs", {}).get("level")
    assert level_7 == 2, f"Expected h2, got h{level_7}"

    # Should have multiple TOML code blocks (at least one per field)
    # With 4-space indent for pymdownx.tabbed, these become indented code blocks
    assert len(code_blocks) >= 5, (
        f"Expected at least 5 code blocks (one per field), got {len(code_blocks)}"
    )

    # Check blocks contain TOML-like content
    for block in code_blocks:
        raw = block.get("raw", "").strip()
        # Either contains TOML code or fence markers
        assert "=" in raw or "```toml" in raw or "[tool.metaxy]" in raw, (
            f"Code block doesn't look TOML-related: {raw[:100]}"
        )


def test_markdown_ast_no_literal_code_fence_in_text() -> None:
    """Test that code fence markers don't appear as literal text in the AST."""
    from typing import Any, cast

    import mistune

    fields = extract_field_info(SamplePluginConfig)
    enable_field = next(f for f in fields if f["name"] == "enable")

    doc = generate_individual_field_doc(enable_field, header_level=3)

    # Parse to AST
    markdown = mistune.create_markdown(renderer=None)
    tokens: list[dict[str, Any]] = cast(list[dict[str, Any]], markdown(doc))

    # Extract all text content from paragraphs
    paragraphs = [token for token in tokens if token.get("type") == "paragraph"]

    for para in paragraphs:
        children = para.get("children", [])
        for child in children:
            if child.get("type") == "text":
                text = child.get("raw", "")
                # Should NOT contain literal code fence markers
                assert "```toml" not in text, (
                    f"Found literal '```toml' in paragraph text: {text}"
                )
                assert "```" not in text or text.strip() == "```", (
                    f"Found literal code fence in text: {text}"
                )


def test_generate_toml_code_block_no_content_on_fence_line() -> None:
    """Test that content doesn't appear on the same line as opening fence."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = '[tool.metaxy]\nstore = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)

    # Find the opening fence line
    fence_line_idx = next(i for i, line in enumerate(lines) if "```toml" in line)

    # The fence line should ONLY contain the fence marker (and indentation)
    fence_line = lines[fence_line_idx]
    assert fence_line.strip() == "```toml", (
        f"Opening fence has content on same line: {repr(fence_line)}"
    )

    # The line after the fence should be blank (critical for rendering)
    if fence_line_idx + 1 < len(lines):
        next_line = lines[fence_line_idx + 1]
        assert next_line == "", (
            f"Expected blank line after opening fence for proper rendering, got: {repr(next_line)}"
        )

    # The content should be on the line after the blank line
    if fence_line_idx + 2 < len(lines):
        content_line = lines[fence_line_idx + 2]
        assert "[tool.metaxy]" in content_line, (
            f"Expected content after blank line, got: {repr(content_line)}"
        )


def test_generate_toml_code_block_no_extra_blank_lines() -> None:
    """Test that there is exactly one blank line after opening fence (required for rendering)."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = '[tool.metaxy]\nstore = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)

    # Find fence positions
    opening_fence_idx = next(i for i, line in enumerate(lines) if "```toml" in line)
    closing_fence_idx = next(
        i for i in range(opening_fence_idx + 1, len(lines)) if lines[i].strip() == "```"
    )

    # Count blank lines between fences (these are INSIDE the code block)
    blank_lines_inside = [
        i
        for i in range(opening_fence_idx + 1, closing_fence_idx)
        if lines[i].strip() == ""
    ]

    # There should be EXACTLY ONE blank line immediately after opening fence (required for rendering)
    # (content has no blank lines in it, so only the one after fence)
    assert len(blank_lines_inside) == 1, (
        f"Expected exactly 1 blank line (after fence) inside code block, found {len(blank_lines_inside)} at positions {blank_lines_inside}"
    )

    # The blank line should be immediately after the opening fence
    assert blank_lines_inside[0] == opening_fence_idx + 1, (
        f"Blank line should be immediately after opening fence at position {opening_fence_idx + 1}, found at {blank_lines_inside[0]}"
    )


def test_generate_toml_code_block_exactly_one_blank_before_and_after() -> None:
    """Test that there is exactly one blank line before and after the code block."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = 'store = "dev"'
    lines = generate_toml_code_block(content, indent_level=1)

    # Expected structure: ["", "    ```toml", "", "    store = \"dev\"", "    ```", ""]
    # (note the blank line after opening fence for proper rendering)
    assert len(lines) == 6, f"Expected 6 lines, got {len(lines)}: {lines}"

    # First line should be blank
    assert lines[0] == "", f"First line should be blank, got: {repr(lines[0])}"

    # Last line should be blank
    assert lines[-1] == "", f"Last line should be blank, got: {repr(lines[-1])}"

    # Second line should be opening fence
    assert lines[1] == "    ```toml", (
        f"Second line should be fence, got: {repr(lines[1])}"
    )

    # Third line should be blank (after opening fence)
    assert lines[2] == "", f"Third line should be blank, got: {repr(lines[2])}"

    # Second-to-last line should be closing fence
    assert lines[-2] == "    ```", (
        f"Second-to-last line should be fence, got: {repr(lines[-2])}"
    )


def test_generate_toml_code_block_no_duplicate_blank_lines() -> None:
    """Test that the function doesn't produce duplicate blank lines."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    content = '[tool.metaxy]\nstore = "dev"\nstores = {}'
    lines = generate_toml_code_block(content, indent_level=1)

    # Convert to string to check for patterns
    result = "\n".join(lines)

    # Should not have triple newlines anywhere (which would be duplicate blank lines)
    assert "\n\n\n" not in result, (
        "Found triple newlines (duplicate blank lines) in output"
    )

    # Count total blank lines
    blank_count = sum(1 for line in lines if line == "")

    # Should have exactly 3 blank lines: one before fence, one after opening fence, one after closing fence
    assert blank_count == 3, (
        f"Expected exactly 3 blank lines, found {blank_count}: {[i for i, line in enumerate(lines) if line == '']}"
    )


def test_generate_toml_code_block_multiline_no_internal_blanks() -> None:
    """Test that multiline content doesn't get extra blank lines inserted (except the required one after opening fence)."""
    from mkdocs_metaxy.config_generator import generate_toml_code_block

    # Content with NO blank lines in it
    content = '[tool.metaxy]\nstore = "dev"\nstores = {}'
    lines = generate_toml_code_block(content, indent_level=1)

    # Expected output: ["", "    ```toml", "", "    [tool.metaxy]", "    store = \"dev\"", "    stores = {}", "    ```", ""]
    # (note the blank line after opening fence for proper rendering)
    expected_length = (
        8  # 1 blank + 1 fence + 1 blank (after fence) + 3 content + 1 fence + 1 blank
    )
    assert len(lines) == expected_length, (
        f"Expected {expected_length} lines, got {len(lines)}: {lines}"
    )

    # Find the content lines (between fences)
    opening_fence_idx = next(i for i, line in enumerate(lines) if "```toml" in line)
    closing_fence_idx = next(
        i for i in range(opening_fence_idx + 1, len(lines)) if lines[i].strip() == "```"
    )

    content_lines = lines[opening_fence_idx + 1 : closing_fence_idx]

    # First content line should be blank (required for rendering)
    assert content_lines[0] == "", "First line after fence should be blank"

    # All other content lines should be non-empty (properly indented)
    for i, line in enumerate(content_lines[1:], start=1):
        assert line.strip() != "", (
            f"Content line {i} is blank, should have content: {repr(line)}"
        )
        assert line.startswith("    "), (
            f"Content line {i} doesn't have proper indentation: {repr(line)}"
        )
