#!/usr/bin/env python3
"""Entry point for the independent graph-free structured method."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from trace_structured.cli import main


if __name__ == "__main__":
    main()
