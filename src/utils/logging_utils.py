# src/utils/logging_utils.py
"""
logging_utils.py
-----------------

Centralized logging tools for the pipeline.

Responsibilities:
- Create timestamped loggers
- Provide decorators for timing
- Configure colored console output

Tools:
- logging
- functools

TODO:
- Add rotating file logs
"""

from rich.console import Console

console = Console()

def info(msg): console.print(f"[bold cyan]{msg}[/]")
def warn(msg): console.print(f"[bold yellow]{msg}[/]")
def error(msg): console.print(f"[bold red]{msg}[/]")
