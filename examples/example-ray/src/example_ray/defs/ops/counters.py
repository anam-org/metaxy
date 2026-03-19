"""Pipeline counters for operational metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass

# --8<-- [start:counters]


@dataclass
class PipelineCounters:
    """Tracks operational metrics surfaced as Dagster output metadata."""

    input_count: int = 0
    selected_count: int = 0
    processed_count: int = 0
    failed_count: int = 0
    total_processing_seconds: float = 0.0

    def to_metadata(self) -> dict[str, int | float]:
        """Convert counters to a dict suitable for Dagster output metadata."""
        return asdict(self)


# --8<-- [end:counters]
