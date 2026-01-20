"""Pipeline orchestrator for recompute example.

This script runs both compute_parent and compute_child in sequence,
demonstrating the full recomputation workflow.
"""

import subprocess
import sys
from pathlib import Path

example_dir = Path(__file__).parent

print("Pipeline")
print("=" * 60)

# Step 1: Compute parent feature
print("\n[1/2] Computing parent feature...")
result = subprocess.run(
    [sys.executable, str(example_dir / "compute_parent.py")],
    capture_output=False,
)
if result.returncode != 0:
    print("✗ Parent computation failed")
    sys.exit(1)

# Step 2: Compute child feature
print("\n[2/2] Computing child feature...")
result = subprocess.run(
    [sys.executable, str(example_dir / "compute_child.py")],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print("✗ Child computation failed")
    print(result.stderr)
    sys.exit(1)

# Print child output (contains idempotence messages)
print(result.stdout)

# Check for idempotence - when both new and changed are 0
if "Identified: 0 new samples, 0 samples with new provenance_by_field" in result.stdout:
    print("No changes detected (idempotent)")

if "changed samples" in result.stdout and "0 samples with new provenance_by_field" not in result.stdout:
    print("Note: Recomputation occurred due to algorithm change")

print("\n✅ Pipeline complete!")
