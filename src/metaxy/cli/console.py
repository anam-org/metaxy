import os
import sys

from rich.console import Console

# Rich console for logs and status messages (goes to stderr)
console = Console(file=sys.stderr, stderr=True)

# Console for data output (goes to stdout)
# This is used for outputting important data that scripts might want to capture
# When not connected to a TTY (e.g., piped output), use a large width to avoid truncation
# Users can override with COLUMNS environment variable
_data_width = None if sys.stdout.isatty() else int(os.environ.get("COLUMNS", 200))
data_console = Console(file=sys.stdout, highlight=False, width=_data_width)

# Error console (also goes to stderr)
error_console = Console(file=sys.stderr, stderr=True, style="bold red", highlight=False)
