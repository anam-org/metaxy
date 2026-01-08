# mkdocs-metaxy

MkDocs plugins and extensions for Metaxy documentation.

## Plugins

### metaxy-examples

An MkDocs plugin for displaying Metaxy example content in documentation. This plugin provides markdown directives for:

- Displaying example scenarios from `.example.yaml` runbooks
- Showing Python source files at different stages of evolution
- Applying patches to demonstrate code changes
- Displaying diff patches with syntax highlighting

#### Installation

The plugin is installed automatically as part of the Metaxy documentation build:

```bash
uv pip install -e docs/.mkdocs-metaxy/
```

#### Configuration

Add the plugin to your `mkdocs.yml`:

```yaml
plugins:
  - metaxy-examples:
      examples_dir: "../examples"  # Path relative to docs/ directory
```

#### Usage

The plugin provides the following directives:

##### 1. GitHub Source Link

Display a link to the example source code on GitHub:

```markdown
::: metaxy-example source-link
example: basic
:::
```

Parameters:

- `example` (required): Example name
- `button` (optional): Display as button (default: true) or inline link (false)
- `text` (optional): Custom link text (default: "View Example Source on GitHub")

Examples:

```markdown
# As a button (default)

::: metaxy-example source-link
example: basic
:::

# As an inline link

::: metaxy-example source-link
example: basic
button: false
:::

# With custom text

::: metaxy-example source-link
example: basic
text: "Browse Example Code"
:::

# Alternative directive name

::: metaxy-example github
example: basic
:::
```

##### 2. Display Scenarios

Show the scenarios from a runbook:

```markdown
::: metaxy-example scenarios
example: basic
:::
```

This displays:

- Scenario names and descriptions
- Lists of steps in each scenario
- Command descriptions and assertions

##### 3. Display Source Files

Show a Python source file with syntax highlighting and collapsible wrapper:

```markdown
::: metaxy-example file
example: basic
path: src/example_basic/features.py
:::
```

Parameters:

- `example` (required): Example name (e.g., "basic" for "example-basic")
- `path` (required): File path relative to example directory
- `linenos` (optional): Show line numbers (default: true)
- `patches` (optional): List of patches to apply before displaying

##### 4. Display Files After Patches

Show how a file looks after applying patches:

```markdown
::: metaxy-example file
example: basic
path: src/example_basic/features.py
patches: ["patches/01_update_parent_algorithm.patch"]
:::
```

The plugin will:

1. Create a temporary copy of the example
2. Apply the specified patches in order
3. Display the modified file with syntax highlighting
4. Clean up the temporary files

##### 5. Display Patches

Show the diff patch itself:

```markdown
::: metaxy-example patch
example: basic
path: patches/01_update_parent_algorithm.patch
:::
```

This displays the patch with diff syntax highlighting.

#### Features

- **Automatic patch application**: Applies patches in a temporary directory without modifying the original files
- **Syntax highlighting**: Uses Pygments for beautiful code and diff highlighting
- **Line numbers**: Optional line numbers for code files
- **Collapsible code blocks**: All code snippets are wrapped in collapsible `<details>` elements
- **GitHub integration**: Direct links to example source code on GitHub
- **Error handling**: Clear error messages if examples, files, or patches are not found
- **Custom styling**: CSS classes for easy customization
- **Runbook integration**: Reads directly from `.example.yaml` files

#### Styling

The plugin adds custom CSS classes that you can style:

```css
/* Scenario containers */
.metaxy-scenarios
.metaxy-scenario
.scenario-description
.scenario-steps

/* Code file containers */
.metaxy-code-file
.code-file-header
.file-path
.file-stage
.code-file-content

/* Patch containers */
.metaxy-patch
.patch-header
.patch-path
.patch-content

/* Error messages */
.metaxy-error
```

See `docs/css/metaxy_examples.css` for the default styling.

#### Example

See `docs/learn/examples.md` for a complete working example demonstrating all features.

#### Architecture

The plugin consists of:

1. **plugin.py**: MkDocs plugin that registers the markdown extension
2. **markdown_ext.py**: Markdown preprocessor that parses directives
3. **core.py**: Loader that wraps `metaxy._testing.Runbook` for MkDocs integration
4. **renderer.py**: HTML rendering with syntax highlighting

The plugin uses the official runbook system from `metaxy._testing` to load and parse `.example.yaml` files, ensuring consistency between test execution and documentation.

The plugin hooks into MkDocs' build process to:

1. Register as an MkDocs plugin
2. Add a markdown extension to process directives
3. Parse directive blocks in markdown files
4. Load runbooks using `metaxy._testing.Runbook`
5. Apply patches to temporary copies of examples
6. Render HTML with syntax highlighting
7. Inject the HTML into the final documentation

#### Development

The plugin is located in `docs/.mkdocs-metaxy/src/mkdocs_metaxy/examples/`.

To modify the plugin:

1. Edit the source files
2. Rebuild the docs: `uv run mkdocs build`
3. Check the output in `site/`

The plugin is automatically reinstalled when you run `uv run mkdocs build` or `uv run mkdocs serve`.
