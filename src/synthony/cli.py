"""
Command-line interface for synthony.

Provides two commands:
- profile: Analyze a dataset and output stress factors
- benchmark: Generate synthetic benchmark datasets
"""

import sys
from pathlib import Path

# CLI is optional dependency - fail gracefully if not installed
try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print(
        "CLI dependencies not installed. Install with: pip install synthony[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

from synthony.benchmark.generators import BenchmarkDatasetGenerator
from synthony.core.analyzer import StochasticDataAnalyzer

app = typer.Typer(help="synthony: Tabular data profiling and stress detection")
console = Console()


@app.command()
def profile(
    input_path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path (prints to stdout if not specified)"
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--no-pretty", help="Pretty print results to console"
    ),
):
    """Profile a dataset and detect stress factors.

    Analyzes the input file and outputs a JSON profile with detected
    stress factors (skewness, cardinality, Zipfian, etc.).

    Examples:

        # Profile and print to console
        synthony-profile data.csv

        # Save profile to JSON
        synthony-profile data.csv --output profile.json

        # Quiet mode (JSON only)
        synthony-profile data.csv --no-pretty
    """
    try:
        console.print(f"[bold blue]Analyzing:[/bold blue] {input_path}")

        # Run analysis
        analyzer = StochasticDataAnalyzer()
        profile = analyzer.analyze_from_file(input_path)

        # Display summary if pretty mode
        if pretty:
            console.print()
            table = Table(title="Dataset Profile Summary", show_header=True)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")

            # Basic info
            table.add_row("Dataset ID", profile.dataset_id)
            table.add_row("Row Count", str(profile.row_count))
            table.add_row("Column Count", str(profile.column_count))

            # Stress factors
            table.add_row("", "")  # Separator
            table.add_row(
                "Severe Skew", "✓ Yes" if profile.stress_factors.severe_skew else "✗ No"
            )
            table.add_row(
                "High Cardinality",
                "✓ Yes" if profile.stress_factors.high_cardinality else "✗ No",
            )
            table.add_row(
                "Zipfian Distribution",
                "✓ Yes" if profile.stress_factors.zipfian_distribution else "✗ No",
            )
            table.add_row(
                "Small Data", "✓ Yes" if profile.stress_factors.small_data else "✗ No"
            )
            table.add_row(
                "Large Data", "✓ Yes" if profile.stress_factors.large_data else "✗ No"
            )
            table.add_row(
                "Higher-Order Correlation",
                "✓ Yes" if profile.stress_factors.higher_order_correlation else "✗ No",
            )

            # Details
            if profile.skewness and profile.skewness.severe_columns:
                table.add_row("", "")  # Separator
                table.add_row(
                    "Severe Skew Columns", ", ".join(profile.skewness.severe_columns)
                )
                table.add_row(
                    "Max Skewness", f"{profile.skewness.max_skewness:.2f}"
                )

            if profile.zipfian and profile.zipfian.detected:
                table.add_row("", "")  # Separator
                table.add_row(
                    "Zipfian Columns", ", ".join(profile.zipfian.affected_columns)
                )
                if profile.zipfian.top_20_percent_ratio:
                    table.add_row(
                        "Zipfian Ratio",
                        f"{profile.zipfian.top_20_percent_ratio:.3f}",
                    )

            console.print(table)
            console.print()

        # Save or print JSON
        if output:
            analyzer.to_json(profile, output)
            console.print(f"[green]✓ Saved profile to:[/green] {output}")
        else:
            # Print JSON to stdout if no output file
            if not pretty:
                # In quiet mode, only print JSON
                print(profile.to_json())
            else:
                console.print("[bold]JSON Output:[/bold]")
                console.print(profile.to_json())

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    output_dir: Path = typer.Option(
        "./benchmarks", "--output-dir", "-o", help="Output directory for datasets"
    ),
):
    """Generate synthetic benchmark datasets.

    Creates three control datasets for validation:
    - Dataset A: "The Long Tail" (severe skewness)
    - Dataset B: "The Needle in Haystack" (Zipfian distribution)
    - Dataset C: "The Small Data Trap" (overfitting risk)

    Example:

        # Generate in default directory
        synthony-benchmark

        # Generate in custom directory
        synthony-benchmark --output-dir ./my_benchmarks
    """
    try:
        console.print(
            f"[bold blue]Generating benchmark datasets in:[/bold blue] {output_dir}"
        )
        console.print()

        # Generate datasets
        BenchmarkDatasetGenerator.save_benchmarks(output_dir)

        console.print(f"[green]✓ All benchmarks saved to:[/green] {output_dir}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", err=True)
        raise typer.Exit(code=1)


# Entry points for console scripts
def profile_command():
    """Entry point for synthony-profile command."""
    app(["profile"] + sys.argv[1:])


def benchmark_command():
    """Entry point for synthony-benchmark command."""
    app(["benchmark"] + sys.argv[1:])


if __name__ == "__main__":
    app()
