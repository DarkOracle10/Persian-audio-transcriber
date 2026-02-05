#!/usr/bin/env python3
"""
Batch Processing Example - Persian Audio Transcriber

This example demonstrates batch processing of multiple audio files,
parallel processing, progress tracking, and error handling.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from persian_transcriber import PersianAudioTranscriber
from persian_transcriber.utils.exceptions import TranscriberError


def example_1_transcribe_directory() -> None:
    """Example 1: Transcribe all files in a directory."""
    print("\n" + "=" * 60)
    print("Example 1: Transcribe Directory")
    print("=" * 60)

    transcriber = PersianAudioTranscriber(
        model_size="medium", device="cuda", verbose=True  # Use "cpu" if no GPU
    )

    # Directory containing audio files
    audio_dir = Path("./recordings")

    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        print("Creating example directory...")
        audio_dir.mkdir(exist_ok=True)
        return

    # Get all audio files
    audio_files = []
    for ext in [".mp3", ".wav", ".flac", ".m4a"]:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    # Process each file
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        try:
            result = transcriber.transcribe_file(str(audio_file))
            results.append(
                {
                    "file": audio_file.name,
                    "text": result["text"],
                    "duration": result.get("duration", 0),
                    "success": True,
                }
            )
            print(f"✓ Success: {result['text'][:50]}...")
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append(
                {
                    "file": audio_file.name,
                    "error": str(e),
                    "success": False,
                }
            )

    # Save results
    output_file = "batch_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_file}")


def example_2_parallel_processing() -> None:
    """Example 2: Parallel processing for faster batch transcription."""
    print("\n" + "=" * 60)
    print("Example 2: Parallel Processing")
    print("=" * 60)

    audio_dir = Path("./recordings")
    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    # Get audio files
    audio_files = list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("No MP3 files found")
        return

    print(f"Processing {len(audio_files)} files in parallel...")

    def transcribe_one(audio_path: Path) -> Dict[str, Any]:
        """Transcribe a single file."""
        # Each worker gets its own transcriber instance
        transcriber = PersianAudioTranscriber(
            model_size="small",
            device="cpu",  # Use CPU for parallel processing
        )

        try:
            start_time = time.time()
            result = transcriber.transcribe_file(str(audio_path))
            elapsed = time.time() - start_time

            return {
                "file": audio_path.name,
                "text": result["text"],
                "duration": result.get("duration", 0),
                "processing_time": elapsed,
                "success": True,
            }
        except Exception as e:
            return {
                "file": audio_path.name,
                "error": str(e),
                "success": False,
            }

    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(transcribe_one, audio_file): audio_file for audio_file in audio_files
        }

        # Process completed tasks
        for future in as_completed(future_to_file):
            audio_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)

                if result["success"]:
                    print(f"✓ {result['file']}: {result['text'][:40]}...")
                else:
                    print(f"✗ {result['file']}: {result['error']}")
            except Exception as e:
                print(f"✗ {audio_file.name}: Unexpected error: {e}")

    # Calculate statistics
    successful = sum(1 for r in results if r["success"])
    total_duration = sum(r.get("duration", 0) for r in results if r["success"])
    total_processing = sum(r.get("processing_time", 0) for r in results if r["success"])

    print(f"\n{'=' * 60}")
    print(f"Completed: {successful}/{len(results)} files")
    print(f"Total audio duration: {total_duration:.2f}s")
    print(f"Total processing time: {total_processing:.2f}s")
    print(f"Speed: {total_duration / total_processing:.2f}x realtime")


def example_3_progress_callback() -> None:
    """Example 3: Custom progress tracking with callback."""
    print("\n" + "=" * 60)
    print("Example 3: Progress Tracking")
    print("=" * 60)

    from tqdm import tqdm

    audio_dir = Path("./recordings")
    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    audio_files = list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("No MP3 files found")
        return

    transcriber = PersianAudioTranscriber(model_size="small")

    # Use tqdm for progress bar
    results = []
    with tqdm(total=len(audio_files), desc="Transcribing") as pbar:
        for audio_file in audio_files:
            try:
                result = transcriber.transcribe_file(str(audio_file))
                results.append(
                    {
                        "file": audio_file.name,
                        "text": result["text"],
                        "success": True,
                    }
                )
                pbar.set_postfix(file=audio_file.name[:20])
            except Exception as e:
                results.append(
                    {
                        "file": audio_file.name,
                        "error": str(e),
                        "success": False,
                    }
                )
            pbar.update(1)

    print(f"\nProcessed {len(results)} files")


def example_4_incremental_output() -> None:
    """Example 4: Save results incrementally during batch processing."""
    print("\n" + "=" * 60)
    print("Example 4: Incremental Output")
    print("=" * 60)

    audio_dir = Path("./recordings")
    output_dir = Path("./transcriptions")
    output_dir.mkdir(exist_ok=True)

    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    audio_files = list(audio_dir.glob("*.mp3"))
    if not audio_files:
        print("No MP3 files found")
        return

    transcriber = PersianAudioTranscriber(model_size="medium", output_format="txt")

    print(f"Processing {len(audio_files)} files...")
    print(f"Output directory: {output_dir}")

    for i, audio_file in enumerate(audio_files, 1):
        output_file = output_dir / f"{audio_file.stem}.txt"

        # Skip if already processed
        if output_file.exists():
            print(f"[{i}/{len(audio_files)}] ⊙ Skipping (exists): {audio_file.name}")
            continue

        print(f"[{i}/{len(audio_files)}] ⟳ Processing: {audio_file.name}")
        try:
            result = transcriber.transcribe_file(str(audio_file), output_path=str(output_file))
            print(f"  ✓ Saved to: {output_file.name}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nAll files processed. Transcriptions saved to: {output_dir}")


def example_5_filter_and_validate() -> None:
    """Example 5: Filter files and validate results."""
    print("\n" + "=" * 60)
    print("Example 5: Filter and Validate")
    print("=" * 60)

    audio_dir = Path("./recordings")
    if not audio_dir.exists():
        print(f"Directory not found: {audio_dir}")
        return

    # Get all audio files with size info
    audio_files = []
    for ext in [".mp3", ".wav", ".flac"]:
        for audio_file in audio_dir.glob(f"*{ext}"):
            size_mb = audio_file.stat().st_size / (1024 * 1024)
            audio_files.append((audio_file, size_mb))

    # Filter: only process files under 50MB
    MAX_SIZE_MB = 50
    filtered_files = [(f, s) for f, s in audio_files if s <= MAX_SIZE_MB]

    print(f"Found {len(audio_files)} audio files")
    print(f"Filtered to {len(filtered_files)} files under {MAX_SIZE_MB}MB")

    transcriber = PersianAudioTranscriber()

    for audio_file, size_mb in filtered_files:
        print(f"\nProcessing: {audio_file.name} ({size_mb:.2f}MB)")
        try:
            result = transcriber.transcribe_file(str(audio_file))

            # Validate result
            text = result["text"].strip()
            if not text:
                print(f"  ⚠ Warning: Empty transcription")
            elif len(text) < 10:
                print(f"  ⚠ Warning: Very short transcription ({len(text)} chars)")
            else:
                print(f"  ✓ Success: {len(text)} characters")

        except Exception as e:
            print(f"  ✗ Failed: {e}")


def main() -> None:
    """Run batch processing examples."""
    print("Persian Audio Transcriber - Batch Processing Examples")
    print("=" * 60)
    print("\nThese examples demonstrate batch processing capabilities.")
    print("Create a './recordings' directory with audio files to test.\n")

    # Create example directory if it doesn't exist
    recordings_dir = Path("./recordings")
    if not recordings_dir.exists():
        print(f"Creating directory: {recordings_dir}")
        recordings_dir.mkdir(exist_ok=True)
        print("Add audio files to this directory and run again.")
        return

    # Run examples
    example_1_transcribe_directory()
    # example_2_parallel_processing()  # Uncomment to test parallel processing
    # example_3_progress_callback()    # Requires tqdm: pip install tqdm
    # example_4_incremental_output()
    # example_5_filter_and_validate()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
