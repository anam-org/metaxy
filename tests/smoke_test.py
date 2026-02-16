"""Smoke test to verify the package can be imported and basic features work."""

import subprocess
import sys

import metaxy as mx


def main() -> None:
    # Verify version is accessible
    assert hasattr(mx, "__version__"), "Missing __version__"
    print(f"metaxy version: {mx.__version__}")

    # Verify core imports work
    from metaxy import BaseFeature, FeatureGraph

    print(f"BaseFeature: {BaseFeature}")
    print(f"FeatureGraph: {FeatureGraph}")

    # Verify CLI works
    result = subprocess.run([sys.executable, "-m", "metaxy.cli.app", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI --help failed: {result.stderr}"
    print("CLI --help: OK")

    print("Smoke test passed!")


if __name__ == "__main__":
    main()
