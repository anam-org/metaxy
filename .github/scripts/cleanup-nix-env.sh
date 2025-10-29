#!/usr/bin/env bash
# Filters out /nix/store paths from PATH and LD_LIBRARY_PATH
# Used to cleanup Nix environment in GitHub Actions to prevent conflicts with non-Nix tools

set -euo pipefail

# Function to filter out /nix/store paths from a colon-separated path variable
filter_nix_store() {
    local input_path="$1"
    local filtered_path=""

    # Handle empty input
    if [ -z "$input_path" ]; then
        echo ""
        return
    fi

    # Convert path to array using IFS
    IFS=':' read -ra PATHS <<< "$input_path"

    for path in "${PATHS[@]}"; do
        # Skip paths that contain /nix/store
        if [[ ! "$path" =~ /nix/store ]]; then
            if [ -z "$filtered_path" ]; then
                filtered_path="$path"
            else
                filtered_path="$filtered_path:$path"
            fi
        fi
    done

    echo "$filtered_path"
}

# Filter PATH
filtered_path=$(filter_nix_store "$PATH")
echo "PATH=$filtered_path" >> "$GITHUB_ENV"

# Filter LD_LIBRARY_PATH if it exists
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    filtered_ld_library_path=$(filter_nix_store "$LD_LIBRARY_PATH")
    echo "LD_LIBRARY_PATH=$filtered_ld_library_path" >> "$GITHUB_ENV"
fi

# Unset Nix-specific environment variables
unset NIX_BUILD_CORES NIX_STORE || true
