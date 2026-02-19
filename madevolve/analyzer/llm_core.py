"""
LLM-based report generation core module.

This module provides:
- LLM calling utilities with retry logic
- Report generation framework
- Integration with scenario adapters

Supported models via litellm:
- Anthropic: claude-3-5-haiku, claude-3-5-sonnet, claude-3-opus, etc.
- OpenAI: gpt-4o, gpt-4o-mini, o1, o3-mini, etc.
- DeepSeek: deepseek-chat, deepseek-reasoner
- Gemini: gemini-2.5-pro, gemini-2.5-flash, etc.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    AlgorithmInfo,
    EvolutionData,
    ScenarioAdapter,
)

logger = logging.getLogger(__name__)


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. If None, searches common locations.
    """
    search_paths = []

    if env_path:
        search_paths.append(Path(env_path))
    else:
        # Search common locations
        search_paths.extend([
            Path(__file__).parent.parent.parent / ".env",  # Project root
            Path(__file__).parent.parent / ".env",
            Path.cwd() / ".env",
        ])

    for path in search_paths:
        if path.exists():
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Don't override existing environment variables
                        if key not in os.environ:
                            os.environ[key] = value
            logger.debug(f"Loaded environment from: {path}")
            return

    logger.debug("No .env file found")


# Try to load .env on module import
load_env_file()


# Try to import litellm
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("litellm not installed. LLM features will be disabled.")


# Default model
DEFAULT_MODEL = "gemini/gemini-3-pro-preview"

# Models that need special handling (disable thinking mode)
GEMINI_THINKING_MODELS = {
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite-preview-06-17",
    "gemini/gemini-3-pro-preview",
}


class LLMClient:
    """
    LLM client wrapper with retry logic and model-specific handling.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the LLM client.

        Args:
            model: The LLM model to use
        """
        self.model = model
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if LLM features are available."""
        if not LITELLM_AVAILABLE:
            logger.warning(
                "LLM features disabled. Install litellm: pip install litellm"
            )

    @property
    def is_available(self) -> bool:
        """Check if LLM is available for use."""
        return LITELLM_AVAILABLE

    def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The user prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt

        Returns:
            The LLM response text, or None if unavailable/error
        """
        if not LITELLM_AVAILABLE:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Workaround for Gemini 2.5 thinking models
            if self.model in GEMINI_THINKING_MODELS:
                kwargs["thinking"] = {"type": "disabled"}

            response = litellm.completion(**kwargs)
            content = response.choices[0].message.content

            if content is None:
                logger.error("LLM returned None content")
                return None

            return content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None


class ReportGenerator:
    """
    Main report generator that uses scenario adapters for customization.
    """

    def __init__(
        self,
        adapter: ScenarioAdapter,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the report generator.

        Args:
            adapter: ScenarioAdapter for the target domain
            model: LLM model to use
        """
        self.adapter = adapter
        self.llm = LLMClient(model)
        self.model = model

    def generate_algorithm_analysis(
        self,
        algorithm: AlgorithmInfo,
        is_baseline: bool,
    ) -> str:
        """
        Generate LLM analysis of an algorithm.

        Args:
            algorithm: The algorithm to analyze
            is_baseline: Whether this is the baseline algorithm

        Returns:
            Markdown-formatted analysis
        """
        if not self.llm.is_available:
            return "_LLM analysis not available._"

        # Extract relevant code block
        code = self.adapter.extract_code_block(algorithm.code)

        # Get metrics info
        metrics_info = ""
        if algorithm.metrics:
            metrics_info = self.adapter.metrics_adapter.get_key_metrics_for_llm(
                algorithm.metrics
            )

        # Build the prompt
        prompt = self.adapter.prompt_adapter.get_algorithm_analysis_prompt(
            AlgorithmInfo(
                code=code,
                generation=algorithm.generation,
                metrics=algorithm.metrics,
                timestamp=algorithm.timestamp,
            ),
            is_baseline=is_baseline,
        )

        response = self.llm.complete(prompt, temperature=0.3, max_tokens=8000)
        return response or "_Analysis generation failed._"

    def generate_improvement_analysis(
        self,
        baseline: AlgorithmInfo,
        best: AlgorithmInfo,
    ) -> str:
        """
        Generate LLM analysis of improvements.

        Args:
            baseline: The baseline algorithm
            best: The best evolved algorithm

        Returns:
            Markdown-formatted improvement analysis
        """
        if not self.llm.is_available:
            return "_LLM analysis not available._"

        # Generate metrics comparison
        metrics_comparison = ""
        if baseline.metrics and best.metrics:
            metrics_comparison = self.adapter.metrics_adapter.get_metrics_comparison_table(
                baseline.metrics, best.metrics
            )

        prompt = self.adapter.prompt_adapter.get_improvement_analysis_prompt(
            AlgorithmInfo(
                code=self.adapter.extract_code_block(baseline.code),
                generation=baseline.generation,
                metrics=baseline.metrics,
            ),
            AlgorithmInfo(
                code=self.adapter.extract_code_block(best.code),
                generation=best.generation,
                metrics=best.metrics,
            ),
            metrics_comparison,
        )

        response = self.llm.complete(prompt, temperature=0.3, max_tokens=8000)
        return response or "_Improvement analysis generation failed._"

    def generate_executive_summary(
        self,
        data: EvolutionData,
    ) -> str:
        """
        Generate an executive summary of the evolution.

        Args:
            data: Complete evolution data

        Returns:
            Markdown-formatted executive summary
        """
        if not self.llm.is_available:
            return "_LLM analysis not available._"

        metrics_comparison = ""
        if data.baseline.metrics and data.best.metrics:
            metrics_comparison = self.adapter.metrics_adapter.get_metrics_comparison_table(
                data.baseline.metrics, data.best.metrics
            )

        prompt = self.adapter.prompt_adapter.get_executive_summary_prompt(
            data, metrics_comparison
        )

        response = self.llm.complete(prompt, temperature=0.3, max_tokens=4000)
        return response or "_Summary generation failed._"

    def generate_full_report(
        self,
        data: EvolutionData,
        output_path: Optional[str] = None,
        include_code: bool = True,
    ) -> str:
        """
        Generate a complete evolution report.

        Args:
            data: The extracted evolution data
            output_path: Optional path to save the report
            include_code: Whether to include algorithm code in appendix

        Returns:
            The complete report as markdown text
        """
        print(f"Generating report for scenario: {data.scenario}")
        print(f"Using model: {self.model}")

        # Generate all components
        print("  - Generating metrics comparison...")
        metrics_table = ""
        if data.baseline.metrics and data.best.metrics:
            metrics_table = self.adapter.metrics_adapter.get_metrics_comparison_table(
                data.baseline.metrics, data.best.metrics
            )

        print("  - Generating baseline analysis...")
        baseline_analysis = self.generate_algorithm_analysis(
            data.baseline, is_baseline=True
        )

        print("  - Generating best algorithm analysis...")
        best_analysis = self.generate_algorithm_analysis(
            data.best, is_baseline=False
        )

        print("  - Generating improvement analysis...")
        improvement_analysis = self.generate_improvement_analysis(
            data.baseline, data.best
        )

        print("  - Generating executive summary...")
        executive_summary = self.generate_executive_summary(data)

        # Assemble the report
        report = self.adapter.template_adapter.format_final_report(
            data=data,
            metrics_table=metrics_table,
            baseline_analysis=baseline_analysis,
            best_analysis=best_analysis,
            improvement_analysis=improvement_analysis,
            executive_summary=executive_summary,
        )

        # Add code appendix if requested
        if include_code:
            report += self._generate_code_appendix(data)

        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")

        return report

    def generate_quick_report(
        self,
        data: EvolutionData,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a quick report without LLM analysis.

        Args:
            data: Evolution data
            output_path: Optional path to save the report

        Returns:
            Markdown report without LLM analysis
        """
        print("Generating quick report (no LLM)...")

        # Metrics comparison
        metrics_table = ""
        if data.baseline.metrics and data.best.metrics:
            metrics_table = self.adapter.metrics_adapter.get_metrics_comparison_table(
                data.baseline.metrics, data.best.metrics
            )

        # Evolution stats
        history = data.history
        if history.best_scores:
            initial_score = history.best_scores[0]
            final_score = max(history.best_scores)
            if initial_score != 0:
                improvement_pct = 100 * (final_score - initial_score) / abs(initial_score)
            else:
                improvement_pct = 0
        else:
            initial_score = final_score = improvement_pct = 0

        title = self.adapter.template_adapter.get_report_title()
        headers = self.adapter.template_adapter.get_section_headers()

        report = f"""# {title} (Quick Summary)

**Scenario**: {data.scenario}

---

## Table of Contents

1. [Summary Statistics](#summary-statistics)
2. [{headers.get('metrics', 'Metrics Comparison')}](#{headers.get('metrics', 'Metrics Comparison').lower().replace(' ', '-')})
3. [Configuration](#configuration)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Duration | {data.duration_hours:.2f} hours |
| Total Generations | {history.total_generations} |
| Total Programs | {history.total_programs} |
| Successful Programs | {history.successful_programs} ({100*history.successful_programs/max(1,history.total_programs):.1f}%) |
| Best Generation | {data.best.generation} |
| Initial Score | {initial_score:.4f} |
| Final Best Score | {final_score:.4f} |
| Relative Improvement | {improvement_pct:.1f}% |

---

## {headers.get('metrics', 'Metrics Comparison')}

{metrics_table}

---

## Configuration

| Setting | Value |
|---------|-------|
| Number of Islands | {data.config.num_islands} |
| LLM Models | {', '.join(data.config.llm_models) if data.config.llm_models else 'N/A'} |

---

*Quick report generated without LLM analysis. Use full report for detailed analysis.*
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Quick report saved to: {output_path}")

        return report

    def _generate_code_appendix(self, data: EvolutionData) -> str:
        """Generate the code appendix section."""
        baseline_code = self.adapter.extract_code_block(data.baseline.code)
        best_code = self.adapter.extract_code_block(data.best.code)

        return f"""

---

## Appendix: Algorithm Code

### Baseline Algorithm

```python
{baseline_code}
```

### Best Evolved Algorithm

```python
{best_code}
```

---

*This report was automatically generated using LLM-assisted analysis.*
"""


def export_markdown_to_pdf(
    markdown_path: str,
    pdf_path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Export a markdown file to PDF with embedded images.

    Uses playwright to render markdown as HTML and export to PDF.

    Args:
        markdown_path: Path to the markdown file
        pdf_path: Output PDF path (default: same name with .pdf extension)
        base_dir: Base directory for resolving relative image paths
                  (default: directory containing the markdown file)

    Returns:
        Path to the generated PDF, or None if export failed
    """
    import os
    from pathlib import Path

    md_file = Path(markdown_path)
    if not md_file.exists():
        print(f"Error: Markdown file not found: {markdown_path}")
        return None

    # Default PDF path
    if pdf_path is None:
        pdf_path = str(md_file.with_suffix('.pdf'))

    # Default base directory
    if base_dir is None:
        base_dir = str(md_file.parent)

    try:
        import markdown
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        print(f"Error: Required library not installed: {e}")
        print("Install with: pip install markdown playwright")
        return None

    # Read markdown content
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    md_converter = markdown.Markdown(
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    html_body = md_converter.convert(md_content)

    # Create full HTML with styling
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Evolution Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            font-size: 14px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 28px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 8px;
            margin-top: 30px;
            font-size: 22px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
            font-size: 18px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
        }}
        pre {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 11px;
            line-height: 1.4;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f8f9fa;
            color: #666;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #eee;
            margin: 30px 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        .codehilite {{
            background-color: #2d2d2d;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
        }}
        @media print {{
            body {{
                padding: 20px;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    # Write temporary HTML file (needed for relative image paths)
    temp_html_path = md_file.with_suffix('.temp.html')
    try:
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Use playwright to generate PDF
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the HTML file (file:// protocol for local images)
            page.goto(f'file://{os.path.abspath(temp_html_path)}')

            # Wait for images to load
            page.wait_for_load_state('networkidle')
            page.wait_for_timeout(500)

            # Generate PDF
            page.pdf(
                path=pdf_path,
                format='A4',
                margin={
                    'top': '20mm',
                    'bottom': '20mm',
                    'left': '15mm',
                    'right': '15mm'
                },
                print_background=True,
            )

            browser.close()

        print(f"PDF exported to: {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"Error exporting to PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Clean up temp HTML file
        if temp_html_path.exists():
            temp_html_path.unlink()
