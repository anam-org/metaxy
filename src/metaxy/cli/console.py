import sys

from rich.console import Console

# Rich console for logs and status messages (goes to stderr)
console = Console(stderr=True, highlight=False)

# Console for data output (goes to stdout)
# This is used for outputting important data that scripts might want to capture
data_console = Console(file=sys.stdout, highlight=False)

# Error console (also goes to stderr)
error_console = Console(stderr=True, style="bold red", highlight=False)
