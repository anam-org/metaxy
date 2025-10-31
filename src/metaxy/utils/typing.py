class _CodeVersionDescriptor:
    """Descriptor returning this feature's field-only code version hash.

    The hash is cached on the feature spec (`FeatureSpec.field_code_version_hash`)
    and excludes any dependency information, allowing callers to distinguish
    between "my code changed" and "one of my dependencies changed".
    """

    def __get__(self, instance, owner) -> str:
        if owner.spec is None:
            raise ValueError(
                f"Feature '{owner.__name__}' has no spec; cannot compute code_version."
            )
        return owner.spec.field_code_version_hash
