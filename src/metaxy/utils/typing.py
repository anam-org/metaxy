class _CodeVersionDescriptor:
    """Descriptor that returns field-only code version hashes."""

    def __get__(self, instance, owner) -> str:
        if owner.spec is None:
            raise ValueError(
                f"Feature '{owner.__name__}' has no spec; cannot compute code_version."
            )
        return owner.spec.field_code_version_hash
