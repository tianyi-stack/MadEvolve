"""
MadEvolve CLI Entry Point
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from madevolve.branding import print_banner
from madevolve.engine.configuration import EvolutionConfig
from madevolve.engine.orchestrator import EvolutionOrchestrator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def build_config_from_dict(data: Dict[str, Any]) -> EvolutionConfig:
    """Recursively build EvolutionConfig from dictionary."""
    from dataclasses import fields, is_dataclass
    from madevolve.engine import configuration as cfg_module

    def _build(cls, d):
        if not is_dataclass(cls) or not isinstance(d, dict):
            return d

        kwargs = {}
        for field in fields(cls):
            if field.name in d:
                field_type = field.type
                # Handle nested dataclasses
                if hasattr(cfg_module, field_type.__name__ if hasattr(field_type, '__name__') else ''):
                    kwargs[field.name] = _build(field_type, d[field.name])
                else:
                    kwargs[field.name] = d[field.name]
            elif field.default is not field.default_factory:
                pass  # Use default
        return cls(**kwargs)

    return _build(EvolutionConfig, data)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MadEvolve: LLM-Driven Evolution Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evolution")
    run_parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress banner output",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging output",
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from madevolve import __version__
        print(f"MadEvolve v{__version__}")
        return 0

    if args.command == "run":
        if not args.quiet:
            print_banner()

        # Configure logging before anything else
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s | %(name)s | %(message)s",
        )

        try:
            config_dict = load_config(args.config)
            config = build_config_from_dict(config_dict)

            orchestrator = EvolutionOrchestrator(
                config=config,
                results_dir=args.output,
                checkpoint_path=args.resume,
            )

            orchestrator.run()
            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
