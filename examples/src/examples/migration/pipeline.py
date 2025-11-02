"""Pipeline orchestrator for migration example.

This script runs compute_parent and compute_child in sequence.
The STAGE environment variable determines which feature version is loaded.
"""

import os
import subprocess
import sys
from pathlib import Path

stage = os.environ.get("STAGE", "1")
example_dir = Path(__file__).parent

print(f"Pipeline STAGE={stage}")
print("=" * 60)

# Step 1: Compute parent feature
print("\n[1/2] Computing parent feature...")
result = subprocess.run(
    [sys.executable, str(example_dir / "compute_parent.py")],
    capture_output=False,
    env=os.environ.copy(),
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
    env=os.environ.copy(),
)
if result.returncode != 0:
    print("✗ Child computation failed")
    print(result.stderr)
    sys.exit(1)

# Print child output
print(result.stdout)

# Check for idempotence - when both new and changed are 0
if (
    "Identified: 0 new samples, 0 samples with new metaxy_provenance_by_field"
    in result.stdout
):
    print("No changes detected (idempotent or migration worked correctly)")

if (
    "changed samples" in result.stdout
    and "0 samples with new metaxy_provenance_by_field" not in result.stdout
):
    print("Note: Recomputation occurred (expected if algorithm changed)")

print(f"\n✅ Stage {stage} pipeline complete!")
