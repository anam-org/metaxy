"""Generate configuration reference documentation."""

import sys
from pathlib import Path

# Add project source to path so we can import metaxy
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


if __name__ == "__main__":
    # This won't be called by mkdocs-gen-files, but allows manual testing
    pass
