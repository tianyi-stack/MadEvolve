"""
MadEvolve Branding and Display Utilities
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from madevolve import __version__
from madevolve.logo import LOGO_ASCII


console = Console()


def print_banner():
    """Print the MadEvolve startup banner."""
    logo_text = Text(LOGO_ASCII, style="bold cyan")

    info_text = Text()
    info_text.append(f"\n  Version: ", style="dim")
    info_text.append(f"{__version__}", style="bold green")
    info_text.append(f"\n  LLM-Driven Evolution Framework", style="dim")

    panel = Panel(
        logo_text + info_text,
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(panel)


def print_generation_header(generation: int, total: int):
    """Print a generation header."""
    console.print(
        f"\n[bold blue]━━━ Generation {generation}/{total} ━━━[/bold blue]"
    )


def print_result(program_id: str, score: float, improved: bool):
    """Print an evaluation result."""
    status = "[green]↑[/green]" if improved else "[dim]─[/dim]"
    console.print(f"  {status} Program {program_id[:8]}... Score: {score:.4f}")


def print_summary(stats: dict):
    """Print evolution summary statistics."""
    console.print("\n[bold]Evolution Summary[/bold]")
    for key, value in stats.items():
        console.print(f"  • {key}: {value}")


def print_step(msg: str):
    """Print a major step indicator."""
    console.print(f"[bold]→ {msg}[/bold]")


def print_substep(msg: str):
    """Print a sub-step indicator."""
    console.print(f"[dim]  ├ {msg}[/dim]")


def print_error(msg: str):
    """Print an error message."""
    console.print(f"[bold red]✗ {msg}[/bold red]")
