"""
Global constants for MadEvolve.
"""

# LLM Defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0

# Code Block Markers
EVOLVE_BLOCK_START = "# EVOLVE-BLOCK-START"
EVOLVE_BLOCK_END = "# EVOLVE-BLOCK-END"

# Regex patterns that match both "# EVOLVE-BLOCK-START" and "# === EVOLVE-BLOCK-START ==="
EVOLVE_BLOCK_START_PATTERN = r"#\s*(?:===\s*)?EVOLVE-BLOCK-START(?:\s*===)?"
EVOLVE_BLOCK_END_PATTERN = r"#\s*(?:===\s*)?EVOLVE-BLOCK-END(?:\s*===)?"

# Patch Syntax
SEARCH_MARKER = "<<<<<<< SEARCH"
REPLACE_MARKER = "======="
END_MARKER = ">>>>>>> REPLACE"

# Population Defaults
DEFAULT_ARCHIVE_SIZE = 50
DEFAULT_ELITE_COUNT = 10
DEFAULT_PARTITION_COUNT = 4

# Embedding Defaults
DEFAULT_EMBEDDING_DIM = 1536

# File Extensions
SUPPORTED_LANGUAGES = {
    ".py": "python",
    ".cpp": "cpp",
    ".c": "c",
    ".cu": "cuda",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# Parameter Optimization
TUNABLE_PATTERN = r"#\s*TUNABLE:\s*(\w+)\s*=\s*([^,]+)(?:,\s*bounds\s*=\s*\(([^)]+)\))?"
DEFAULT_OPT_BUDGET = 10

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
