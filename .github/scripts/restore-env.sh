#!/usr/bin/env bash
# Restores the original PATH and LD_LIBRARY_PATH that were saved before Nix installation
# This ensures non-Nix tools in subsequent steps work correctly

set -euo pipefail

# Restore original PATH
if [ -n "${ORIGINAL_PATH:-}" ]; then
    echo "Restoring original PATH"
    echo "PATH=$ORIGINAL_PATH" >> "$GITHUB_ENV"
    export PATH="$ORIGINAL_PATH"
else
    echo "Warning: ORIGINAL_PATH not found, keeping current PATH"
fi

# Restore original LD_LIBRARY_PATH
if [ -n "${ORIGINAL_LD_LIBRARY_PATH+x}" ]; then
    echo "Restoring original LD_LIBRARY_PATH"
    if [ -z "$ORIGINAL_LD_LIBRARY_PATH" ]; then
        # It was originally unset, so unset it
        unset LD_LIBRARY_PATH
        echo "LD_LIBRARY_PATH=" >> "$GITHUB_ENV"
    else
        echo "LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH" >> "$GITHUB_ENV"
        export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
    fi
else
    echo "Warning: ORIGINAL_LD_LIBRARY_PATH not found, keeping current LD_LIBRARY_PATH"
fi

# Unset Nix-specific environment variables
echo "Unsetting Nix-specific environment variables"
unset NIX_BUILD_CORES NIX_STORE NIX_PATH || true

echo "Environment restored to pre-Nix state"
echo "Current PATH: $PATH"
echo "Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<not set>}"
