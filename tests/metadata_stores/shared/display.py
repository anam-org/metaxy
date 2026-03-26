"""Display test pack for metadata stores."""

from metaxy.metadata_store import MetadataStore


class DisplayTests:
    """Tests for display(), repr(), and name attribute."""

    def test_name_is_none_by_default(self, store: MetadataStore) -> None:
        assert store.name is None

    def test_display_without_name(self, store: MetadataStore) -> None:
        display = store.display()
        assert not display.startswith("[")
        assert store.__class__.__name__ in display

    def test_name_property_returns_configured_name(self, named_store: MetadataStore) -> None:
        assert named_store.name is not None

    def test_display_without_name_even_when_configured(self, named_store: MetadataStore) -> None:
        display = named_store.display()
        assert not display.startswith("[")
        assert named_store.__class__.__name__ in display

    def test_repr_includes_name(self, named_store: MetadataStore) -> None:
        repr_str = repr(named_store)
        assert repr_str.startswith(f"[{named_store.name}]")
        assert named_store.__class__.__name__ in repr_str
        assert named_store.display() in repr_str
