import cyclopts

from metaxy.cli.console import console, data_console, error_console

# List subcommand app
app = cyclopts.App(
    name="list",  # pyrefly: ignore[unexpected-keyword]
    help="List Metaxy entities",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
    error_console=error_console,  # pyrefly: ignore[unexpected-keyword]
)


@app.command()
def features():
    """
    List Metaxy features.
    """
    from metaxy import get_feature_by_key
    from metaxy.cli.context import AppContext
    from metaxy.models.plan import FQFieldKey

    context = AppContext.get()
    graph = context.graph

    for feature_key, feature_spec in graph.feature_specs_by_key.items():
        if (
            context.project
            and get_feature_by_key(feature_key).project != context.project
        ):
            continue
        data_console.print("---")
        version = graph.get_feature_version(feature_key)
        data_console.print(f"{feature_key} (version\n{version})")
        if feature_spec.deps:
            data_console.print("  Feature Dependencies:")
            for dep in feature_spec.deps:
                data_console.print(f"    {dep}")
        data_console.print("  Fields:")
        for field_key, field_spec in feature_spec.fields_by_key.items():
            field_version = graph.get_field_version(
                FQFieldKey(feature=feature_key, field=field_key)
            )
            data_console.print(
                f"    {field_spec.key.to_string()} (code_version {field_spec.code_version}, version\n{field_version})"
            )
