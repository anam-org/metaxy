"""Orchestrator for the DuckLake example.

Running this script exercises the attachment, join/diff, and calculator demos.
"""

from .demo import run_demo

if __name__ == "__main__":
    print("🐤 DuckLake/Narwhals integration walkthrough")
    print("=" * 60)
    run_demo()
    print("✅ Demo complete")
