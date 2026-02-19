#!/usr/bin/env python3
"""
Report Generator CLI

Generate comprehensive evolution reports from experiment results.

Note: This CLI requires adapters to be registered before use.
Register adapters in your application code using:

    from madevolve.analyzer import register_adapter
    from your_project.adapters import YourAdapter
    register_adapter('your_scenario', YourAdapter)

Usage:
    python -m madevolve.analyzer.generate_report [RESULTS_DIR] [OPTIONS]

Examples:
    # Generate report (scenario auto-detected from registered adapters)
    python -m madevolve.analyzer.generate_report /path/to/results

    # Generate with specific scenario
    python -m madevolve.analyzer.generate_report /path/to/results --scenario your_scenario

    # Use a specific generation as the "best" algorithm
    python -m madevolve.analyzer.generate_report /path/to/results --generation 5

    # Quick report without LLM
    python -m madevolve.analyzer.generate_report /path/to/results --quick

    # Specify model and output
    python -m madevolve.analyzer.generate_report /path/to/results --model gpt-4o --output report.md

    # Export to PDF
    python -m madevolve.analyzer.generate_report /path/to/results --pdf

    # List registered scenarios
    python -m madevolve.analyzer.generate_report --list-scenarios
"""

import argparse
import sys
from pathlib import Path

from . import (
    DataExtractor,
    ReportGenerator,
    ScenarioAdapter,
    LITELLM_AVAILABLE,
    DEFAULT_MODEL,
    export_markdown_to_pdf,
    get_adapter,
    list_adapters,
    register_adapter,
)


def print_summary(data):
    """Print a quick summary of the evolution data."""
    print("\n" + "=" * 60)
    print("EVOLUTION DATA SUMMARY")
    print("=" * 60)

    print(f"\nScenario: {data.scenario}")
    print(f"Results Directory: {data.results_dir}")
    print(f"Duration: {data.duration_hours:.2f} hours")
    print(f"Start: {data.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {data.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nTotal Generations: {data.history.total_generations}")
    print(f"Total Programs: {data.history.total_programs}")
    print(f"Successful Programs: {data.history.successful_programs}")

    if data.baseline.metrics:
        print(f"\nBaseline Score: {data.baseline.metrics.combined_score:.4f}")
    if data.best.metrics:
        print(f"Best Score: {data.best.metrics.combined_score:.4f}")
        if data.baseline.metrics:
            improvement = data.best.metrics.combined_score - data.baseline.metrics.combined_score
            baseline_score = data.baseline.metrics.combined_score
            if baseline_score != 0:
                pct = 100 * improvement / abs(baseline_score)
                print(f"Improvement: +{improvement:.4f} ({pct:.1f}%)")

    print(f"\nBest Algorithm Generation: {data.best.generation}")

    print("\n" + "=" * 60)


def auto_detect_scenario(results_dir: Path) -> str:
    """
    Try to auto-detect the scenario from the results directory.

    This function checks registered adapters and uses their detection logic if available.

    Returns:
        Detected scenario name, or None if no registered adapter matched
    """
    adapters = list_adapters()

    if not adapters:
        return None

    # Check each registered adapter for a match
    for name, adapter_class in adapters.items():
        adapter = adapter_class()
        # Check if adapter has a detect method
        if hasattr(adapter, 'detect_scenario'):
            if adapter.detect_scenario(results_dir):
                return name

    # Return the first registered adapter as fallback
    return list(adapters.keys())[0] if adapters else None


def main():
    parser = argparse.ArgumentParser(
        description="Generate evolution reports from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_report.py /path/to/results                  # Auto-detect scenario
    python generate_report.py /path/to/results -s my_scenario   # Specific scenario
    python generate_report.py /path/to/results -g 5             # Use generation 5 as best
    python generate_report.py /path/to/results --quick          # Without LLM
    python generate_report.py /path/to/results -m gpt-4o        # Use specific model
    python generate_report.py --list-scenarios                  # List registered adapters

Note: Adapters must be registered before use. See module docstring for details.
        """
    )

    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to results directory"
    )

    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=None,
        help="Scenario type (must be registered; use --list-scenarios to see available)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: evolution_report.md in results directory)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Generate quick report without LLM analysis"
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary to console, don't generate report"
    )

    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Don't include algorithm code in the report appendix"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export the report to PDF format"
    )

    parser.add_argument(
        "--pdf-only",
        type=str,
        default=None,
        metavar="MARKDOWN_FILE",
        help="Only convert an existing markdown file to PDF (skip report generation)"
    )

    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )

    parser.add_argument(
        "--generation", "-g",
        type=int,
        default=None,
        metavar="N",
        help="Use generation N as the 'best' algorithm instead of auto-selecting highest score"
    )

    args = parser.parse_args()

    # List scenarios
    if args.list_scenarios:
        adapters = list_adapters()
        if not adapters:
            print("No adapters registered.")
            print("Use register_adapter() to register your custom adapter before using this CLI.")
        else:
            print("Available scenarios:")
            for name, adapter_cls in adapters.items():
                adapter = adapter_cls()
                print(f"  {name}: {adapter.metrics_adapter.scenario_description}")
        sys.exit(0)

    # Handle --pdf-only: convert existing markdown to PDF and exit
    if args.pdf_only:
        md_path = Path(args.pdf_only)
        if not md_path.exists():
            print(f"Error: Markdown file not found: {md_path}")
            sys.exit(1)

        pdf_path = str(md_path.with_suffix('.pdf'))
        print(f"Converting {md_path} to PDF...")

        result = export_markdown_to_pdf(
            str(md_path),
            pdf_path,
            base_dir=str(md_path.parent),
        )

        if result:
            print(f"PDF saved to: {result}")
            print("Done!")
            sys.exit(0)
        else:
            print("Error: Failed to export PDF. Make sure playwright is installed:")
            print("  pip install playwright && playwright install chromium")
            sys.exit(1)

    # Validate results directory
    if not args.results_dir:
        parser.print_help()
        print("\nError: results_dir is required")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if not (results_dir / "evolution_db.sqlite").exists():
        print(f"Error: evolution_db.sqlite not found in {results_dir}")
        sys.exit(1)

    # Auto-detect or use specified scenario
    if args.scenario:
        scenario = args.scenario
    else:
        scenario = auto_detect_scenario(results_dir)
        if scenario:
            print(f"Auto-detected scenario: {scenario}")
        else:
            print("Error: No adapters registered and no scenario specified.")
            print("Use register_adapter() to register your custom adapter, or specify --scenario.")
            sys.exit(1)

    # Create adapter
    try:
        adapter = get_adapter(scenario)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Extract data
    print(f"Loading evolution data from: {results_dir}")
    if args.generation is not None:
        print(f"Using generation {args.generation} as best algorithm")
    try:
        extractor = DataExtractor(adapter)
        data = extractor.extract_evolution_data(
            str(results_dir),
            best_generation=args.generation,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print summary
    print_summary(data)

    if args.summary_only:
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(results_dir / "evolution_report.md")

    # Create report generator
    generator = ReportGenerator(adapter, model=args.model)

    # Generate report
    if args.quick or not LITELLM_AVAILABLE:
        if not LITELLM_AVAILABLE and not args.quick:
            print("\nWarning: litellm not installed. Generating quick report instead.")
            print("Install litellm for full LLM-powered analysis: pip install litellm")
        report = generator.generate_quick_report(data, output_path)
    else:
        print(f"\nGenerating full report using model: {args.model}")
        try:
            report = generator.generate_full_report(
                data,
                output_path=output_path,
                include_code=not args.no_code,
            )
        except Exception as e:
            print(f"Error generating full report: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            print("Falling back to quick report...")
            report = generator.generate_quick_report(data, output_path)

    print(f"\nReport saved to: {output_path}")

    # Export to PDF if requested
    if args.pdf:
        pdf_path = output_path.replace('.md', '.pdf')
        if not pdf_path.endswith('.pdf'):
            pdf_path = output_path + '.pdf'

        print(f"\nExporting to PDF...")
        result = export_markdown_to_pdf(
            output_path,
            pdf_path,
            base_dir=str(results_dir),
        )

        if result:
            print(f"PDF saved to: {result}")
        else:
            print("Warning: Failed to export PDF. Make sure playwright is installed:")
            print("  pip install playwright && playwright install chromium")

    print("Done!")


if __name__ == "__main__":
    main()
