#!/bin/bash
# This script sets up the required project-level environment (e.g., PYTHONPATH)
# and then executes the given command.

# Get the directory of the current script
export PATH="/usr/local/bin:$PATH"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Add the project root to PYTHONPATH to handle imports like `from src...` and `from ithor...`
PROJECT_ROOT=$( realpath "$SCRIPT_DIR/../.." )
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PYTHONPATH"
# Execute the rest of the command
exec "$@"
