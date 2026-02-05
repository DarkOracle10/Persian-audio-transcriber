"""
Entry point for running Persian Transcriber as a module.

This allows the package to be run with:
    python -m persian_transcriber

Example:
    python -m persian_transcriber audio.mp3 -m large-v3
    python -m persian_transcriber ./recordings/ --recursive
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
