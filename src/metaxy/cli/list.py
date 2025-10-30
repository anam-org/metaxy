import cyclopts
from rich.console import Console

# Rich console for formatted output
console = Console()

# Migrations subcommand app
app = cyclopts.App(
    name="list",  # pyrefly: ignore[unexpected-keyword]
    help="List Metaxy entities",  # pyrefly: ignore[unexpected-keyword]
    console=console,  # pyrefly: ignore[unexpected-keyword]
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
        console.print("---")
        console.print(
            f"{feature_key} (version {graph.get_feature_version(feature_key)})"
        )
        if feature_spec.deps:
            console.print("  Feature Dependencies:")
            for dep in feature_spec.deps:
                console.print(f"    {dep}")
        console.print("  Fields:")
        for field_key, field_spec in feature_spec.fields_by_key.items():
            console.print(
                f"    {field_spec.key.to_string()} (code_version {field_spec.code_version}, version {graph.get_field_version(FQFieldKey(feature=feature_key, field=field_key))})"
            )
