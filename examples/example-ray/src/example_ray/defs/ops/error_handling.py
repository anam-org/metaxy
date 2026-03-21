"""Failure policy and error collection for production pipelines."""

from __future__ import annotations

from dataclasses import dataclass

# --8<-- [start:failure_policy]


@dataclass
class FailurePolicy:
    """Defines acceptable failure thresholds for a pipeline run."""

    max_failed_fraction: float = 0.05

    def check(
        self,
        total: int,
        failed: int,
        error_samples: list[str] | None = None,
    ) -> None:
        """Raise if the failure rate exceeds the threshold."""
        if total == 0:
            return

        suffix = f". Sample errors: {error_samples[:3]}" if error_samples else ""

        if failed == total:
            raise RuntimeError(f"All {total} samples failed{suffix}")

        fraction = failed / total
        if fraction > self.max_failed_fraction:
            msg = (
                f"{failed}/{total} samples failed "
                f"({fraction:.1%} > {self.max_failed_fraction:.1%}){suffix}"
            )
            raise RuntimeError(msg)


# --8<-- [end:failure_policy]
