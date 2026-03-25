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
err_console = Console(stderr=True)


@app.command()
def profile(
    input_path: Path = typer.Argument(..., help="Path to CSV or Parquet file"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path (prints to stdout if not specified)"
    ),
    pretty: bool = typer.Option(
        True, "--pretty/--no-pretty", help="Pretty print results to console"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed per-column metrics"
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

        # Verbose mode with detailed metrics
        synthony-profile data.csv --verbose

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

            # Verbose mode: show detailed per-column metrics
            if verbose:
                # Skewness details
                if profile.skewness and profile.skewness.column_scores:
                    skew_table = Table(title="Skewness by Column", show_header=True)
                    skew_table.add_column("Column", style="cyan")
                    skew_table.add_column("Skewness", style="magenta", justify="right")
                    skew_table.add_column("Status", style="yellow")
                    
                    threshold = profile.thresholds_used.get("skewness_threshold", 2.0)
                    for col, score in sorted(
                        profile.skewness.column_scores.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    ):
                        status = "[red]SEVERE[/red]" if abs(score) > threshold else "[green]OK[/green]"
                        skew_table.add_row(col, f"{score:.4f}", status)
                    
                    console.print(skew_table)
                    console.print()

                # Cardinality details
                if profile.cardinality and profile.cardinality.column_counts:
                    card_table = Table(title="Cardinality by Column", show_header=True)
                    card_table.add_column("Column", style="cyan")
                    card_table.add_column("Unique Values", style="magenta", justify="right")
                    card_table.add_column("Status", style="yellow")
                    
                    threshold = profile.thresholds_used.get("cardinality_threshold", 500)
                    for col, count in sorted(
                        profile.cardinality.column_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        status = "[red]HIGH[/red]" if count > threshold else "[green]OK[/green]"
                        card_table.add_row(col, str(count), status)
                    
                    console.print(card_table)
                    console.print()

                # Correlation details
                if profile.correlation:
                    corr_table = Table(title="Correlation Analysis", show_header=True)
                    corr_table.add_column("Metric", style="cyan")
                    corr_table.add_column("Value", style="magenta")
                    
                    corr_table.add_row(
                        "Correlation Density", 
                        f"{profile.correlation.correlation_density:.3f}"
                    )
                    corr_table.add_row(
                        "Mean R²", 
                        f"{profile.correlation.mean_r_squared:.3f}"
                    )
                    corr_table.add_row(
                        "Higher-Order Detected",
                        "✓ Yes" if profile.correlation.has_higher_order else "✗ No"
                    )
                    
                    console.print(corr_table)
                    console.print()

                # Column types summary
                if profile.column_types:
                    type_table = Table(title="Column Types", show_header=True)
                    type_table.add_column("Column", style="cyan")
                    type_table.add_column("Type", style="magenta")
                    type_table.add_column("Null %", style="yellow", justify="right")
                    
                    for col, ctype in profile.column_types.items():
                        null_pct = profile.null_percentage.get(col, 0.0) * 100
                        type_table.add_row(col, ctype, f"{null_pct:.1f}%")
                    
                    console.print(type_table)
                    console.print()

                # Thresholds used
                thresh_table = Table(title="Thresholds Used", show_header=True)
                thresh_table.add_column("Parameter", style="cyan")
                thresh_table.add_column("Value", style="magenta", justify="right")
                
                for key, val in profile.thresholds_used.items():
                    thresh_table.add_row(key.replace("_", " ").title(), str(val))
                
                console.print(thresh_table)
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
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

@app.command()
def benchmark(
    original_data: Path = typer.Option(
        ..., "--original", "-r", help="Path to original dataset (CSV or Parquet)"
    ),
    synthetic_data: Path = typer.Option(
        ..., "--synthetic", "-s", help="Path to synthetic dataset (CSV or Parquet)"
    ),
    evaluation_data: Path = typer.Option(
        None, "--evaluation", "-e", 
        help="Path to evaluation (holdout) dataset for differential privacy metrics"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path for benchmark results"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed per-column metrics"
    ),
):
    """Compare original and synthetic datasets using quality metrics.

    Computes data quality metrics including:
    - KL Divergence (distribution similarity)
    - Jensen-Shannon Divergence (symmetric, bounded 0-1)
    - Statistical Fidelity (mean, std, correlation preservation)
    - Utility Score (column statistics correlation)
    - Privacy Score (nearest neighbor distance analysis)
    - Differential Privacy metrics (when --evaluation is provided)

    Examples:

        # Basic comparison
        synthony-benchmark -r original.csv -s synthetic.csv

        # With verbose output
        synthony-benchmark -r original.csv -s synthetic.csv --verbose

        # With differential privacy analysis
        synthony-benchmark -r train.csv -s synthetic.csv -e holdout.csv

        # Save results to JSON
        synthony-benchmark -r original.csv -s synthetic.csv -o results.json
    """
    import json
    import pandas as pd
    from synthony.benchmark.metrics import DataQualityBenchmark, DifferentialPrivacyMetrics
    from synthony.core.loaders import DataLoader
    from synthony.core.analyzer import StochasticDataAnalyzer
    
    try:
        console.print(f"[bold blue]Original:[/bold blue] {original_data}")
        console.print(f"[bold blue]Synthetic:[/bold blue] {synthetic_data}")
        if evaluation_data:
            console.print(f"[bold blue]Evaluation:[/bold blue] {evaluation_data}")
        console.print()

        # Load datasets
        console.print("[dim]Loading datasets...[/dim]")
        orig_df = DataLoader.load(original_data, validate=True)
        synth_df = DataLoader.load(synthetic_data, validate=True)
        
        # Load evaluation dataset if provided
        eval_df = None
        if evaluation_data:
            console.print("[dim]Loading evaluation dataset...[/dim]")
            eval_df = DataLoader.load(evaluation_data, validate=True)
        
        # Run profile analysis on both datasets
        console.print("[dim]Profiling original dataset...[/dim]")
        orig_analyzer = StochasticDataAnalyzer()
        orig_profile = orig_analyzer.analyze(orig_df)
        
        console.print("[dim]Profiling synthetic dataset...[/dim]")
        synth_analyzer = StochasticDataAnalyzer()
        synth_profile = synth_analyzer.analyze(synth_df)
        
        # Run benchmark
        console.print("[dim]Computing metrics...[/dim]")
        benchmark_tool = DataQualityBenchmark()
        result = benchmark_tool.compare(orig_df, synth_df)
        
        # Calculate differential privacy metrics if evaluation dataset provided
        dp_metrics = None
        if eval_df is not None:
            console.print("[dim]Computing differential privacy metrics...[/dim]")
            dp_metrics = benchmark_tool.calculate_differential_privacy(
                training_data=orig_df,
                synthetic_data=synth_df,
                evaluation_data=eval_df,
            )
        
        console.print()
        
        # Summary table
        summary_table = Table(title="Benchmark Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Original Rows", str(result.original_rows))
        summary_table.add_row("Synthetic Rows", str(result.synthetic_rows))
        summary_table.add_row("Columns Compared", str(result.column_count))
        summary_table.add_row("", "")
        summary_table.add_row("Avg KL Divergence", f"{result.avg_kl_divergence:.4f}")
        summary_table.add_row("Avg JS Divergence", f"{result.avg_js_divergence:.4f}")
        summary_table.add_row("", "")
        summary_table.add_row(
            "[bold]Overall Quality Score[/bold]", 
            f"[bold]{result.overall_quality_score:.3f}[/bold] (0-1, higher is better)"
        )
        
        console.print(summary_table)
        console.print()
        
        # Profile Comparison - Stress Factors
        stress_table = Table(title="Stress Factor Comparison", show_header=True)
        stress_table.add_column("Factor", style="cyan")
        stress_table.add_column("Original", style="magenta", justify="center")
        stress_table.add_column("Synthetic", style="magenta", justify="center")
        stress_table.add_column("Match", style="yellow", justify="center")
        
        stress_factors = [
            ("Severe Skew", orig_profile.stress_factors.severe_skew, synth_profile.stress_factors.severe_skew),
            ("High Cardinality", orig_profile.stress_factors.high_cardinality, synth_profile.stress_factors.high_cardinality),
            ("Zipfian Distribution", orig_profile.stress_factors.zipfian_distribution, synth_profile.stress_factors.zipfian_distribution),
            ("Small Data", orig_profile.stress_factors.small_data, synth_profile.stress_factors.small_data),
            ("Large Data", orig_profile.stress_factors.large_data, synth_profile.stress_factors.large_data),
            ("Higher-Order Correlation", orig_profile.stress_factors.higher_order_correlation, synth_profile.stress_factors.higher_order_correlation),
        ]
        
        for name, orig_val, synth_val in stress_factors:
            orig_str = "[green]✓ Yes[/green]" if orig_val else "[dim]✗ No[/dim]"
            synth_str = "[green]✓ Yes[/green]" if synth_val else "[dim]✗ No[/dim]"
            match = "[green]✓[/green]" if orig_val == synth_val else "[red]✗[/red]"
            stress_table.add_row(name, orig_str, synth_str, match)
        
        console.print(stress_table)
        console.print()
        
        # Profile Comparison - Key Metrics
        metrics_table = Table(title="Key Metrics Comparison", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Original", style="magenta", justify="right")
        metrics_table.add_column("Synthetic", style="magenta", justify="right")
        metrics_table.add_column("Difference", style="yellow", justify="right")
        
        # Max skewness
        orig_max_skew = max(orig_profile.skewness.column_scores.values()) if orig_profile.skewness.column_scores else 0
        synth_max_skew = max(synth_profile.skewness.column_scores.values()) if synth_profile.skewness.column_scores else 0
        metrics_table.add_row(
            "Max Skewness",
            f"{orig_max_skew:.2f}",
            f"{synth_max_skew:.2f}",
            f"{abs(orig_max_skew - synth_max_skew):.2f}"
        )
        
        # Max cardinality
        orig_max_card = max(orig_profile.cardinality.column_counts.values()) if orig_profile.cardinality.column_counts else 0
        synth_max_card = max(synth_profile.cardinality.column_counts.values()) if synth_profile.cardinality.column_counts else 0
        metrics_table.add_row(
            "Max Cardinality",
            str(orig_max_card),
            str(synth_max_card),
            str(abs(orig_max_card - synth_max_card))
        )
        
        # Correlation density
        metrics_table.add_row(
            "Correlation Density",
            f"{orig_profile.correlation.correlation_density:.3f}",
            f"{synth_profile.correlation.correlation_density:.3f}",
            f"{abs(orig_profile.correlation.correlation_density - synth_profile.correlation.correlation_density):.3f}"
        )
        
        # Mean R²
        metrics_table.add_row(
            "Mean R²",
            f"{orig_profile.correlation.mean_r_squared:.3f}",
            f"{synth_profile.correlation.mean_r_squared:.3f}",
            f"{abs(orig_profile.correlation.mean_r_squared - synth_profile.correlation.mean_r_squared):.3f}"
        )
        
        console.print(metrics_table)
        console.print()
        
        # Fidelity metrics
        if result.fidelity:
            fidelity_table = Table(title="Fidelity Metrics", show_header=True)
            fidelity_table.add_column("Metric", style="cyan")
            fidelity_table.add_column("Score", style="magenta", justify="right")
            fidelity_table.add_column("Interpretation", style="yellow")
            
            fidelity_table.add_row(
                "Mean Preservation",
                f"{result.fidelity.mean_preservation:.3f}",
                _score_interpretation(result.fidelity.mean_preservation)
            )
            fidelity_table.add_row(
                "Std Preservation",
                f"{result.fidelity.std_preservation:.3f}",
                _score_interpretation(result.fidelity.std_preservation)
            )
            fidelity_table.add_row(
                "Correlation Preservation",
                f"{result.fidelity.correlation_preservation:.3f}",
                _score_interpretation(result.fidelity.correlation_preservation)
            )
            fidelity_table.add_row(
                "[bold]Overall Fidelity[/bold]",
                f"[bold]{result.fidelity.overall_fidelity:.3f}[/bold]",
                _score_interpretation(result.fidelity.overall_fidelity)
            )
            
            console.print(fidelity_table)
            console.print()
        
        # Utility metrics
        if result.utility:
            utility_table = Table(title="Utility Metrics", show_header=True)
            utility_table.add_column("Metric", style="cyan")
            utility_table.add_column("Score", style="magenta", justify="right")
            utility_table.add_column("Interpretation", style="yellow")
            
            utility_table.add_row(
                "Column Statistics Correlation",
                f"{result.utility.column_correlation:.3f}",
                _score_interpretation(result.utility.column_correlation)
            )
            utility_table.add_row(
                "Distribution Similarity",
                f"{result.utility.distribution_similarity:.3f}",
                _score_interpretation(result.utility.distribution_similarity)
            )
            utility_table.add_row(
                "[bold]Overall Utility[/bold]",
                f"[bold]{result.utility.overall_utility:.3f}[/bold]",
                _score_interpretation(result.utility.overall_utility)
            )
            
            console.print(utility_table)
            console.print()
        
        # Privacy metrics
        if result.privacy:
            privacy_table = Table(title="Privacy Metrics", show_header=True)
            privacy_table.add_column("Metric", style="cyan")
            privacy_table.add_column("Value", style="magenta", justify="right")
            privacy_table.add_column("Interpretation", style="yellow")
            
            privacy_table.add_row(
                "Avg Min Distance Ratio",
                f"{result.privacy.min_distance_ratio:.4f}",
                "[green]Higher = More Diverse[/green]" if result.privacy.min_distance_ratio > 0.1 else "[yellow]Low diversity[/yellow]"
            )
            privacy_table.add_row(
                "Near-Duplicate Rate",
                f"{result.privacy.duplicate_rate:.2%}",
                "[green]Low risk[/green]" if result.privacy.duplicate_rate < 0.05 else "[red]Privacy risk![/red]"
            )
            # DCR metrics
            dcr_val = result.privacy.dcr
            dcr_interp = (
                "[green]Low memorization risk[/green]" if dcr_val > 0.05
                else "[yellow]Moderate memorization risk[/yellow]" if dcr_val > 0.01
                else "[red]High memorization risk[/red]"
            )
            privacy_table.add_row(
                "DCR (Avg)",
                f"{dcr_val:.4f}",
                dcr_interp
            )
            dcr_5th = result.privacy.dcr_5th_percentile
            dcr_5th_interp = (
                "[green]Low outlier risk[/green]" if dcr_5th > 0.05
                else "[yellow]Some close records[/yellow]" if dcr_5th > 0.01
                else "[red]Near-identical records exist[/red]"
            )
            privacy_table.add_row(
                "DCR (5th Percentile)",
                f"{dcr_5th:.4f}",
                dcr_5th_interp
            )
            privacy_table.add_row(
                "[bold]Privacy Score[/bold]",
                f"[bold]{result.privacy.privacy_score:.3f}[/bold]",
                _score_interpretation(result.privacy.privacy_score)
            )
            
            console.print(privacy_table)
            console.print()
        
        # Differential Privacy metrics (if evaluation dataset was provided)
        if dp_metrics is not None:
            dp_table = Table(title="Differential Privacy Metrics", show_header=True)
            dp_table.add_column("Metric", style="cyan")
            dp_table.add_column("Value", style="magenta", justify="right")
            dp_table.add_column("Interpretation", style="yellow")
            
            # DCR metrics
            dp_table.add_row(
                "DCR (Synth→Train)",
                f"{dp_metrics.dcr_train:.4f}",
                "[dim]Distance to training records[/dim]"
            )
            dp_table.add_row(
                "DCR (Synth→Eval)",
                f"{dp_metrics.dcr_eval:.4f}",
                "[dim]Distance to evaluation records[/dim]"
            )
            
            # DCR ratio interpretation
            ratio_interp = "[green]Good privacy[/green]" if 0.8 <= dp_metrics.dcr_ratio <= 1.2 else (
                "[yellow]Train bias[/yellow]" if dp_metrics.dcr_ratio > 1.2 else "[red]Potential leakage[/red]"
            )
            dp_table.add_row(
                "DCR Ratio (Eval/Train)",
                f"{dp_metrics.dcr_ratio:.3f}",
                ratio_interp
            )
            
            dp_table.add_row("", "", "")  # Separator
            
            # Membership inference
            mia_interp = "[green]No advantage[/green]" if dp_metrics.membership_advantage < 0.1 else (
                "[yellow]Some advantage[/yellow]" if dp_metrics.membership_advantage < 0.3 else "[red]High risk[/red]"
            )
            dp_table.add_row(
                "Membership Inference Advantage",
                f"{dp_metrics.membership_advantage:.3f}",
                mia_interp
            )
            dp_table.add_row(
                "Membership Inference AUC",
                f"{dp_metrics.membership_auc:.3f}",
                "[green]Near random[/green]" if dp_metrics.membership_auc < 0.6 else "[red]Distinguishable[/red]"
            )
            
            dp_table.add_row("", "", "")  # Separator
            
            # Attribute inference
            attr_interp = "[green]Low risk[/green]" if dp_metrics.attribute_inference_risk < 0.3 else (
                "[yellow]Moderate risk[/yellow]" if dp_metrics.attribute_inference_risk < 0.5 else "[red]High risk[/red]"
            )
            dp_table.add_row(
                "Attribute Inference Risk",
                f"{dp_metrics.attribute_inference_risk:.3f}",
                attr_interp
            )
            
            dp_table.add_row("", "", "")  # Separator
            
            # Overall scores
            dp_table.add_row(
                "[bold]Empirical DP Score[/bold]",
                f"[bold]{dp_metrics.empirical_dp_score:.3f}[/bold]",
                _score_interpretation(dp_metrics.empirical_dp_score)
            )
            if dp_metrics.estimated_epsilon is not None:
                eps_interp = "[green]Strong[/green]" if dp_metrics.estimated_epsilon < 1.0 else (
                    "[yellow]Moderate[/yellow]" if dp_metrics.estimated_epsilon < 5.0 else "[red]Weak[/red]"
                )
                dp_table.add_row(
                    "Estimated ε (epsilon)",
                    f"{dp_metrics.estimated_epsilon:.2f}",
                    eps_interp
                )
            
            console.print(dp_table)
            console.print()
        
        # Verbose: per-column metrics
        if verbose and result.column_metrics:
            col_table = Table(title="Per-Column Divergence", show_header=True)
            col_table.add_column("Column", style="cyan")
            col_table.add_column("Type", style="dim")
            col_table.add_column("KL Div", style="magenta", justify="right")
            col_table.add_column("JS Div", style="magenta", justify="right")
            col_table.add_column("Status", style="yellow")
            
            for col_name, metrics in sorted(
                result.column_metrics.items(),
                key=lambda x: x[1].js_divergence,
                reverse=True
            ):
                js = metrics.js_divergence
                if js < 0.05:
                    status = "[green]Excellent[/green]"
                elif js < 0.15:
                    status = "[green]Good[/green]"
                elif js < 0.3:
                    status = "[yellow]Fair[/yellow]"
                else:
                    status = "[red]Poor[/red]"
                
                col_table.add_row(
                    col_name,
                    metrics.column_type[:3],
                    f"{metrics.kl_divergence:.4f}",
                    f"{metrics.js_divergence:.4f}",
                    status
                )
            
            console.print(col_table)
            console.print()
        
        # Save results
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Build complete result with profile comparison
            full_result = result.to_dict()
            full_result["profile_comparison"] = {
                "stress_factors": {
                    "severe_skew": {"original": orig_profile.stress_factors.severe_skew, "synthetic": synth_profile.stress_factors.severe_skew},
                    "high_cardinality": {"original": orig_profile.stress_factors.high_cardinality, "synthetic": synth_profile.stress_factors.high_cardinality},
                    "zipfian_distribution": {"original": orig_profile.stress_factors.zipfian_distribution, "synthetic": synth_profile.stress_factors.zipfian_distribution},
                    "small_data": {"original": orig_profile.stress_factors.small_data, "synthetic": synth_profile.stress_factors.small_data},
                    "large_data": {"original": orig_profile.stress_factors.large_data, "synthetic": synth_profile.stress_factors.large_data},
                    "higher_order_correlation": {"original": orig_profile.stress_factors.higher_order_correlation, "synthetic": synth_profile.stress_factors.higher_order_correlation},
                },
                "skewness": {
                    "original": orig_profile.skewness.column_scores,
                    "synthetic": synth_profile.skewness.column_scores,
                },
                "cardinality": {
                    "original": orig_profile.cardinality.column_counts,
                    "synthetic": synth_profile.cardinality.column_counts,
                },
                "correlation": {
                    "original": {"density": orig_profile.correlation.correlation_density, "mean_r_squared": orig_profile.correlation.mean_r_squared},
                    "synthetic": {"density": synth_profile.correlation.correlation_density, "mean_r_squared": synth_profile.correlation.mean_r_squared},
                }
            }
            
            # Add differential privacy metrics if computed
            if dp_metrics is not None:
                full_result["differential_privacy"] = {
                    "dcr_train": dp_metrics.dcr_train,
                    "dcr_eval": dp_metrics.dcr_eval,
                    "dcr_ratio": dp_metrics.dcr_ratio,
                    "membership_advantage": dp_metrics.membership_advantage,
                    "membership_auc": dp_metrics.membership_auc,
                    "attribute_inference_risk": dp_metrics.attribute_inference_risk,
                    "empirical_dp_score": dp_metrics.empirical_dp_score,
                    "estimated_epsilon": dp_metrics.estimated_epsilon,
                }
            
            with open(output, 'w') as f:
                json.dump(full_result, f, indent=2)
            console.print(f"[green]✓ Results saved to:[/green] {output}")

    except FileNotFoundError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def _score_interpretation(score: float) -> str:
    """Return colored interpretation of a 0-1 score."""
    if score >= 0.9:
        return "[green]Excellent[/green]"
    elif score >= 0.7:
        return "[green]Good[/green]"
    elif score >= 0.5:
        return "[yellow]Fair[/yellow]"
    else:
        return "[red]Poor[/red]"


# Entry points for console scripts
def profile_command():
    """Entry point for synthony-profile command."""
    app(["profile"] + sys.argv[1:])


def benchmark_command():
    """Entry point for synthony-benchmark command."""
    app(["benchmark"] + sys.argv[1:])


def recommender_command():
    """Entry point for synthony-recommender command."""
    app(["recommend"] + sys.argv[1:])


@app.command()
def recommend(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Path to input dataset (CSV or Parquet)"
    ),
    method: str = typer.Option(
        "rulebased", "--method", "-m", 
        help="Recommendation method: llm, rulebased, or hybrid"
    ),
    profile_path: Path = typer.Option(
        None, "--profile", "-p", help="Path to existing profile JSON (optional, skips profiling)"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output JSON path for recommendations"
    ),
    top_n: int = typer.Option(
        3, "--top", "-n", help="Number of top recommendations to return"
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Exclude GPU-only models"
    ),
    strict_dp: bool = typer.Option(
        False, "--strict-dp", help="Only differential privacy models"
    ),
    skew_sf: float = typer.Option(
        1.0, "--skew-sf", help="Scale factor for skewness capability (0=default, >1=emphasize, <1=de-emphasize)"
    ),
    cardinality_sf: float = typer.Option(
        1.0, "--cardinality-sf", help="Scale factor for cardinality capability"
    ),
    zipfian_sf: float = typer.Option(
        1.0, "--zipfian-sf", help="Scale factor for Zipfian distribution capability"
    ),
    small_data_sf: float = typer.Option(
        1.0, "--small-data-sf", help="Scale factor for small data handling capability"
    ),
    correlation_sf: float = typer.Option(
        1.0, "--correlation-sf", help="Scale factor for correlation handling capability"
    ),
    privacy_dp_sf: float = typer.Option(
        1.0, "--privacy-dp-sf", help="Scale factor for privacy/differential privacy capability"
    ),
    scale_config: Path = typer.Option(
        None, "--scale-config", "-sc", help="Path to JSON file with scale factors (overrides individual --*-sf options)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed recommendation reasoning"
    ),
):
    """Recommend the optimal synthetic data model for a dataset.

    Analyzes dataset characteristics and recommends the best model(s) based on:
    - Data stress factors (skewness, cardinality, Zipfian distribution)
    - Model capabilities and constraints
    - Optional constraint filtering (CPU-only, strict DP)

    Methods:
    - rulebased: Fast, deterministic scoring using model capabilities
    - llm: LLM-based reasoning (requires OPENAI_API_KEY)
    - hybrid: Combined rule-based + LLM approach

    Examples:

        # Basic recommendation (rule-based)
        synthony-recommender -i data.csv

        # LLM-based recommendation
        synthony-recommender -i data.csv --method llm

        # With constraints
        synthony-recommender -i data.csv --cpu-only --strict-dp

        # Use existing profile
        synthony-recommender -i data.csv --profile profile.json
    """
    import json
    import os
    from synthony.core.loaders import DataLoader
    from synthony.core.analyzer import StochasticDataAnalyzer
    from synthony.recommender.engine import ModelRecommendationEngine
    
    try:
        console.print(f"[bold blue]Dataset:[/bold blue] {input_path}")
        console.print(f"[bold blue]Method:[/bold blue] {method}")
        console.print()
        
        # Validate method
        valid_methods = ["llm", "rulebased", "hybrid"]
        method_normalized = method.lower().replace("-", "_").replace("rule_based", "rulebased")
        if method_normalized == "rule_based":
            method_normalized = "rulebased"
        
        # Map to engine's expected format
        method_map = {
            "llm": "llm",
            "rulebased": "rule_based",
            "hybrid": "hybrid",
        }
        
        if method_normalized not in method_map:
            err_console.print(f"[red]Error:[/red] Invalid method '{method}'. Choose from: llm, rulebased, hybrid")
            raise typer.Exit(code=1)
        
        engine_method = method_map[method_normalized]
        
        # Load or generate profile
        if profile_path and profile_path.exists():
            console.print(f"[dim]Loading profile from: {profile_path}[/dim]")
            from synthony.core.schemas import DatasetProfile
            with open(profile_path) as f:
                profile_data = json.load(f)
            profile = DatasetProfile.model_validate(profile_data)
        else:
            console.print("[dim]Profiling dataset...[/dim]")
            df = DataLoader.load(input_path, validate=True)
            analyzer = StochasticDataAnalyzer()
            profile = analyzer.analyze(df)
        
        # Build constraints with scale factors
        # Start with CLI-provided scale factors
        sf_skew = skew_sf
        sf_cardinality = cardinality_sf
        sf_zipfian = zipfian_sf
        sf_small_data = small_data_sf
        sf_correlation = correlation_sf
        sf_privacy_dp = privacy_dp_sf
        
        # Override from JSON config if provided
        if scale_config and scale_config.exists():
            console.print(f"[dim]Loading scale factors from: {scale_config}[/dim]")
            with open(scale_config) as f:
                sf_json = json.load(f)
            # Extract from model_constraints key if present, otherwise use root
            sf_data = sf_json.get("model_constraints", sf_json)
            # Map JSON field names to internal names (handle both naming conventions)
            sf_skew = sf_data.get("skewness_sf", sf_data.get("skew_sf", sf_skew))
            sf_cardinality = sf_data.get("cardinality_sf", sf_cardinality)
            sf_zipfian = sf_data.get("zipfian_sf", sf_zipfian)
            sf_small_data = sf_data.get("small_data_sf", sf_small_data)
            sf_correlation = sf_data.get("correlation_sf", sf_correlation)
            sf_privacy_dp = sf_data.get("privacy_dp_sf", sf_privacy_dp)
        
        constraints = {
            "cpu_only": cpu_only,
            "strict_dp": strict_dp,
            "skew_sf": sf_skew,
            "cardinality_sf": sf_cardinality,
            "zipfian_sf": sf_zipfian,
            "small_data_sf": sf_small_data,
            "correlation_sf": sf_correlation,
            "privacy_dp_sf": sf_privacy_dp,
        }
        
        # Validate scale factors (must be >= 0)
        sf_values = {
            "skew_sf": sf_skew,
            "cardinality_sf": sf_cardinality,
            "zipfian_sf": sf_zipfian,
            "small_data_sf": sf_small_data,
            "correlation_sf": sf_correlation,
            "privacy_dp_sf": sf_privacy_dp,
        }
        SF_MIN, SF_MAX = 0.0, 10.0
        for sf_name, sf_val in sf_values.items():
            if sf_val < SF_MIN or sf_val > SF_MAX:
                err_console.print(f"[red]Error:[/red] Scale factor {sf_name} must be between {SF_MIN} and {SF_MAX} (got {sf_val})")
                raise typer.Exit(code=1)
        
        # Show scale factors if verbose or any non-default
        non_default_sfs = {k: v for k, v in sf_values.items() if v != 1.0}
        if verbose or non_default_sfs:
            sf_table = Table(title="Scale Factors", show_header=True)
            sf_table.add_column("Capability", style="cyan")
            sf_table.add_column("Scale Factor", style="magenta", justify="right")
            sf_table.add_column("Effect", style="yellow")
            
            for sf_name, sf_val in sf_values.items():
                if sf_val == 0.0:
                    effect = "[red]Ignore (0x)[/red]"
                elif sf_val == 1.0:
                    effect = "[dim]No Scaling[/dim]"
                elif sf_val > 1.0:
                    effect = f"[green]Emphasize ({sf_val}x)[/green]"
                else:
                    effect = f"[yellow]De-emphasize ({sf_val}x)[/yellow]"
                
                cap_name = sf_name.replace("_sf", "").replace("_", " ").title()
                sf_table.add_row(cap_name, f"{sf_val:.1f}", effect)
            
            console.print(sf_table)
            console.print()
        
        # Initialize recommendation engine
        console.print("[dim]Generating recommendations...[/dim]")
        
        # Check for LLM requirements
        if engine_method in ["llm", "hybrid"]:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                console.print("[yellow]⚠ OPENAI_API_KEY not set. Falling back to rule_based.[/yellow]")
                engine_method = "rule_based"
        
        engine = ModelRecommendationEngine()
        result = engine.recommend(
            dataset_profile=profile,
            constraints=constraints,
            top_n=top_n,
            method=engine_method,
        )
        
        console.print()
        
        # Summary table
        summary_table = Table(title="Recommendation Summary", show_header=True)
        summary_table.add_column("", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Dataset ID", profile.dataset_id)
        summary_table.add_row("Row Count", str(profile.row_count))
        summary_table.add_row("Column Count", str(profile.column_count))
        summary_table.add_row("Method", engine_method)
        summary_table.add_row("", "")
        summary_table.add_row("[bold]Recommended Model[/bold]", f"[bold green]{result.recommended_model.model_name}[/bold green]")
        summary_table.add_row("Confidence", f"{result.recommended_model.confidence_score:.0%}")
        
        console.print(summary_table)
        console.print()
        
        # Difficulty summary
        if result.difficulty_summary:
            diff_table = Table(title="Dataset Difficulty", show_header=True)
            diff_table.add_column("Factor", style="cyan")
            diff_table.add_column("Status", style="magenta")
            
            is_hard = result.difficulty_summary.get("is_hard_problem", False)
            diff_table.add_row("Hard Problem Detected", "[red]✓ Yes[/red]" if is_hard else "[green]✗ No[/green]")
            
            factors = result.difficulty_summary.get("factors", {})
            if factors:
                for factor, active in factors.items():
                    status = "[yellow]✓ Yes[/yellow]" if active else "[dim]✗ No[/dim]"
                    diff_table.add_row(factor.replace("_", " ").title(), status)
            
            console.print(diff_table)
            console.print()
        
        # Primary recommendation details
        prim = result.recommended_model
        rec_table = Table(title="Primary Recommendation", show_header=True)
        rec_table.add_column("Property", style="cyan")
        rec_table.add_column("Value", style="magenta")
        
        rec_table.add_row("Model", f"[bold]{prim.model_name}[/bold]")
        rec_table.add_row("Confidence", f"{prim.confidence_score:.0%}")
        
        if prim.reasoning:
            # reasoning is a list of strings
            reasoning_text = "; ".join(prim.reasoning[:2])
            if len(reasoning_text) > 100:
                reasoning_text = reasoning_text[:100] + "..."
            rec_table.add_row("Reasoning", reasoning_text)
        
        if prim.warnings:
            rec_table.add_row("Warnings", ", ".join(prim.warnings))
        
        console.print(rec_table)
        console.print()
        
        # Alternatives
        if result.alternative_models:
            alt_table = Table(title="Alternative Models", show_header=True)
            alt_table.add_column("#", style="dim")
            alt_table.add_column("Model", style="cyan")
            alt_table.add_column("Confidence", style="magenta", justify="right")
            alt_table.add_column("Reasoning", style="yellow")
            
            for i, alt in enumerate(result.alternative_models, 1):
                # reasoning is a list of strings
                reasoning_text = alt.reasoning[0][:50] + "..." if alt.reasoning and len(alt.reasoning[0]) > 50 else (alt.reasoning[0] if alt.reasoning else "-")
                alt_table.add_row(str(i), alt.model_name, f"{alt.confidence_score:.0%}", reasoning_text)
            
            console.print(alt_table)
            console.print()
        
        # Verbose: excluded models
        if verbose and result.excluded_models:
            excl_table = Table(title="Excluded Models", show_header=True)
            excl_table.add_column("Model", style="cyan")
            excl_table.add_column("Reason", style="yellow")
            
            for model, reason in result.excluded_models.items():
                excl_table.add_row(model, reason)
            
            console.print(excl_table)
            console.print()
        
        # Verbose: model capabilities for top recommendation
        if verbose and prim.model_info:
            info_table = Table(title=f"Model Capabilities: {prim.model_name}", show_header=True)
            info_table.add_column("Capability", style="cyan")
            info_table.add_column("Score", style="magenta", justify="right")
            
            caps = prim.model_info.get("capabilities", {})
            for cap, score in caps.items():
                info_table.add_row(cap.replace("_", " ").title(), str(score))
            
            console.print(info_table)
            console.print()
        
        # Score breakdown table (shown when non-default SFs used or verbose)
        if verbose or non_default_sfs:
            breakdown_table = Table(title=f"Score Breakdown: {prim.model_name}", show_header=True)
            breakdown_table.add_column("Capability", style="cyan")
            breakdown_table.add_column("Model", style="magenta", justify="center")
            breakdown_table.add_column("Match", style="yellow", justify="right")
            breakdown_table.add_column("SF", style="blue", justify="right")
            breakdown_table.add_column("Contribution", style="green", justify="right")
            
            # Map SF names to capability names
            sf_to_cap = {
                "skew_sf": "skew_handling",
                "cardinality_sf": "cardinality_handling",
                "zipfian_sf": "zipfian_handling",
                "small_data_sf": "small_data",
                "correlation_sf": "correlation_handling",
                "privacy_dp_sf": "privacy_dp",
            }
            
            caps = prim.model_info.get("capabilities", {})
            total_contrib = 0.0
            
            for sf_name, cap_name in sf_to_cap.items():
                model_score = caps.get(cap_name, 0)
                sf_val = sf_values.get(sf_name, 1.0)
                
                # Calculate match score (simplified - assumes required)
                match_score = 1.0 if model_score >= 3 else (0.7 if model_score >= 2 else 0.4)
                
                # Calculate contribution
                contribution = match_score * sf_val
                total_contrib += contribution
                
                cap_display = cap_name.replace("_", " ").title()
                contrib_str = f"{contribution:.2f}" if sf_val > 0 else "[dim]0.00[/dim]"
                
                breakdown_table.add_row(
                    cap_display,
                    str(model_score),
                    f"{match_score:.1f}",
                    f"{sf_val:.1f}",
                    contrib_str
                )
            
            # Add total row
            breakdown_table.add_row("", "", "", "[bold]Total[/bold]", f"[bold]{total_contrib:.2f}[/bold]")
            
            console.print(breakdown_table)
            console.print()
        
        # Set default output path if not provided
        if output is None:
            # Default: {project_path}/output/rec/rec__{input_filename}__{method}.json
            input_stem = input_path.stem  # filename without extension
            output = Path("output") / "rec" / f"rec__{input_stem}__{method_normalized}.json"
        
        # Save results
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            "dataset_id": result.dataset_id,
            "method": engine_method,
            "constraints": constraints,
            "primary_recommendation": {
                "model_name": prim.model_name,
                "confidence_score": prim.confidence_score,
                "reasoning": prim.reasoning,
                "warnings": prim.warnings,
            },
            "alternatives": [
                {
                    "model_name": alt.model_name,
                    "confidence_score": alt.confidence_score,
                    "reasoning": alt.reasoning,
                }
                for alt in result.alternative_models
            ],
            "difficulty_summary": result.difficulty_summary,
            "excluded_models": result.excluded_models,
        }
        
        with open(output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        console.print(f"[green]✓ Recommendations saved to:[/green] {output}")

    except FileNotFoundError as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        err_console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
