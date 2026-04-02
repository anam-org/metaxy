"""Display test pack for metadata stores."""

from metaxy.metadata_store import MetadataStore


class DisplayTests:
    """Tests for display() and name attribute.

    Subclasses may set ``expected_display_contains`` to a string that must
    appear in ``store.display()`` (e.g. the engine class name).
    """

    expected_display_contains: str | None = None

    def test_name_is_none_by_default(self, store: MetadataStore) -> None:
        assert store.name is None

    def test_display_without_name(self, store: MetadataStore) -> None:
        display = store.display()
        assert display
        assert not display.startswith("[")
        if self.expected_display_contains is not None:
            assert self.expected_display_contains in display
