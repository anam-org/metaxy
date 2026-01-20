#!/bin/bash
# Load direnv environment and execute the command
eval "$(direnv export bash 2>/dev/null)"
exec "$@"
