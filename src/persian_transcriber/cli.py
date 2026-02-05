"""
Command-line interface for Persian Transcriber.

This module provides the CLI entry point for the Persian Transcriber tool,
allowing users to transcribe audio and video files from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import TranscriberConfig, EngineConfig, OutputConfig, DeviceType
from .engines.base import EngineType
from .output import OutputFormat
from .transcriber import PersianAudioTranscriber, SUPPORTED_EXTENSIONS
from .utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="persian-transcriber",
        description="Persian Audio & Video Transcription Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  persian-transcriber audio.mp3

  # Transcribe with large model and GPU
  persian-transcriber audio.mp3 -m large-v3 -d cuda

  # Transcribe entire directory
  persian-transcriber ./recordings/ --recursive

  # Output as SRT subtitles
  persian-transcriber video.mp4 -f srt

  # Use OpenAI API
  persian-transcriber audio.mp3 -e openai_api --api-key sk-...

For more information, visit: https://github.com/DarkOracle10/persian-audio-transcriber
        """,
    )

    # Version
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Input
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Path to audio/video file or directory to transcribe",
    )

    # Engine options
    engine_group = parser.add_argument_group("Engine Options")

    engine_group.add_argument(
        "-e",
        "--engine",
        type=str,
        choices=[e.value for e in EngineType],
        default="faster_whisper",
        help="Transcription engine to use (default: faster_whisper)",
    )

    engine_group.add_argument(
        "-m",
        "--model",
        type=str,
        default="medium",
        help="Model size for Whisper engines (tiny, base, small, medium, large-v3)",
    )

    engine_group.add_argument(
        "-d",
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Computation device (default: auto)",
    )

    engine_group.add_argument(
        "--compute-type",
        type=str,
        choices=["float16", "int8", "int8_float16", "float32"],
        help="Compute type for Faster Whisper (default: auto-detected)",
    )

    # Language options
    lang_group = parser.add_argument_group("Language Options")

    lang_group.add_argument(
        "-l",
        "--language",
        type=str,
        default="fa",
        help="Language code for transcription (default: fa)",
    )

    lang_group.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable Persian text normalization",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")

    output_group.add_argument(
        "-f",
        "--format",
        type=str,
        choices=[f.value for f in OutputFormat],
        default="txt",
        help="Output format (default: txt)",
    )

    output_group.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (default: same directory as input)",
    )

    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch processing",
    )

    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output file, only print to stdout",
    )

    output_group.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in text output",
    )

    # Batch processing options
    batch_group = parser.add_argument_group("Batch Processing")

    batch_group.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search directories for media files",
    )

    batch_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already have transcription output (default: True)",
    )

    batch_group.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files, overwrite instead",
    )

    # API options
    api_group = parser.add_argument_group("API Options")

    api_group.add_argument(
        "--api-key",
        type=str,
        help="API key for OpenAI (can also use OPENAI_API_KEY env var)",
    )

    # General options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser


def validate_input(path: str) -> Path:
    """Validate the input file or directory."""
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {p.suffix}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return p


def print_progress(current: int, total: int, filename: str) -> None:
    """Print progress during batch processing."""
    print(f"[{current}/{total}] Processing: {filename}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Setup logging
    if args.quiet:
        setup_logging(level="ERROR")
    elif args.verbose:
        setup_logging(verbose=True)
    else:
        setup_logging()

    # Check for input
    if not args.input:
        parser.print_help()
        return 1

    try:
        # Validate input
        input_path = validate_input(args.input)

        # Build configuration
        config = TranscriberConfig(
            language=args.language,
            engine=EngineConfig(
                type=args.engine,
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
            ),
            output=OutputConfig(
                format=args.format,
                directory=Path(args.output_dir) if args.output_dir else None,
                include_timestamps=args.timestamps,
            ),
            openai_api_key=args.api_key,
            verbose=args.verbose,
        )

        # Disable normalization if requested
        if args.no_normalize:
            config.normalizer.enabled = False

        # Create transcriber
        transcriber = PersianAudioTranscriber(config=config)

        # Process
        if input_path.is_file():
            # Single file
            result = transcriber.transcribe_file(
                input_path,
                output_path=args.output,
                save_output=not args.no_save,
            )

            if args.no_save:
                # Print result to stdout
                print(result["text"])
            else:
                print(f"\n✓ Transcription saved to: {result.get('output_path', 'N/A')}")
                print(f"  Duration: {result.get('duration', 0):.2f}s")
                print(f"  Processing time: {result.get('processing_time', 0):.2f}s")

        else:
            # Directory (batch)
            skip_existing = not args.no_skip

            results = transcriber.scan_and_transcribe(
                input_path,
                recursive=args.recursive,
                skip_existing=skip_existing,
                output_directory=args.output_dir,
                progress_callback=None if args.quiet else print_progress,
            )

            # Summary
            successful = sum(1 for r in results if "error" not in r)
            print(f"\n✓ Batch processing complete")
            print(f"  Successful: {successful}/{len(results)}")

            if successful < len(results):
                failed = [r for r in results if "error" in r]
                print(f"  Failed files:")
                for r in failed:
                    print(f"    - {r.get('file', 'Unknown')}: {r.get('error', 'Unknown error')}")

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    except ValueError as e:
        logger.error(str(e))
        return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
