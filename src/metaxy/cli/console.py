import sys

from rich.console import Console

# Rich console for logs and status messages (goes to stderr)
console = Console(file=sys.stderr, stderr=True)

# Console for data output (goes to stdout)
# This is used for outputting important data that scripts might want to capture
# Set a minimum width to prevent Rich tables from truncating content in narrow terminals
data_console = Console(file=sys.stdout, highlight=False, width=120)

# Error console (also goes to stderr)
error_console = Console(file=sys.stderr, stderr=True, style="bold red", highlight=False)
