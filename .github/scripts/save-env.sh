#!/usr/bin/env bash
# Saves the current PATH and LD_LIBRARY_PATH before Nix modifies them
# This allows us to restore the original environment after Nix operations

set -euo pipefail

# Save original PATH
echo "ORIGINAL_PATH=$PATH" >> "$GITHUB_ENV"

# Save original LD_LIBRARY_PATH if it exists
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo "ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
else
    echo "ORIGINAL_LD_LIBRARY_PATH=" >> "$GITHUB_ENV"
fi

echo "Environment variables saved before Nix installation"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
