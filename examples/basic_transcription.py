#!/usr/bin/env python3
"""
Basic Transcription Example - Persian Audio Transcriber

This example demonstrates simple single-file transcription using the
PersianAudioTranscriber Python API.
"""

from pathlib import Path
from persian_transcriber import PersianAudioTranscriber
from persian_transcriber.config import TranscriberConfig, EngineConfig, OutputConfig
from persian_transcriber.engines import EngineType
from persian_transcriber.output import OutputFormat


def example_1_simple_transcription() -> None:
    """Example 1: Basic transcription with defaults."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Transcription")
    print("=" * 60)

    # Create transcriber with default settings
    transcriber = PersianAudioTranscriber()

    # Transcribe a file
    audio_file = "audio.mp3"  # Replace with your audio file
    if Path(audio_file).exists():
        result = transcriber.transcribe_file(audio_file)

        print(f"\nTranscribed text:")
        print(result["text"])
        print(f"\nLanguage: {result.get('language', 'unknown')}")
        print(f"Duration: {result.get('duration', 0):.2f}s")
    else:
        print(f"Audio file not found: {audio_file}")


def example_2_custom_model() -> None:
    """Example 2: Transcription with custom model and device."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Model Configuration")
    print("=" * 60)

    # Create transcriber with specific model and GPU
    transcriber = PersianAudioTranscriber(
        model_size="large-v3",  # Use largest, most accurate model
        device="cuda",  # Use GPU acceleration
        language="fa",  # Specify Persian/Farsi
        verbose=True,  # Show detailed progress
    )

    audio_file = "audio.mp3"
    if Path(audio_file).exists():
        result = transcriber.transcribe_file(audio_file)
        print(f"\nTranscribed text: {result['text']}")


def example_3_with_config_object() -> None:
    """Example 3: Using configuration objects for advanced control."""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Configuration")
    print("=" * 60)

    # Create configuration object
    config = TranscriberConfig(
        engine=EngineConfig(
            type=EngineType.FASTER_WHISPER,  # Use faster CTranslate2 backend
            model_size="medium",
            device="cuda",
            compute_type="float16",  # Mixed precision for speed
        ),
        output=OutputConfig(
            format=OutputFormat.JSON,  # Get detailed JSON output
            include_timestamps=True,
            include_word_timestamps=True,
        ),
    )

    # Create transcriber with config
    transcriber = PersianAudioTranscriber(config=config)

    audio_file = "audio.mp3"
    if Path(audio_file).exists():
        result = transcriber.transcribe_file(audio_file)

        # JSON output includes detailed segment information
        print(f"\nFull text: {result['text']}")
        if "segments" in result:
            print(f"\nNumber of segments: {len(result['segments'])}")
            print("\nFirst 3 segments:")
            for i, segment in enumerate(result["segments"][:3], 1):
                print(f"\n  Segment {i}:")
                print(f"    Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
                print(f"    Text: {segment['text']}")


def example_4_save_to_file() -> None:
    """Example 4: Transcribe and save to different output formats."""
    print("\n" + "=" * 60)
    print("Example 4: Save to File")
    print("=" * 60)

    audio_file = "audio.mp3"
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        return

    # Save as plain text
    transcriber_txt = PersianAudioTranscriber(output_format="txt")
    result = transcriber_txt.transcribe_file(audio_file, output_path="output.txt")
    print(f"Saved plain text to: output.txt")

    # Save as SRT subtitles
    transcriber_srt = PersianAudioTranscriber(output_format="srt")
    result = transcriber_srt.transcribe_file(audio_file, output_path="output.srt")
    print(f"Saved SRT subtitles to: output.srt")

    # Save as JSON with full details
    transcriber_json = PersianAudioTranscriber(output_format="json")
    result = transcriber_json.transcribe_file(audio_file, output_path="output.json")
    print(f"Saved JSON data to: output.json")


def example_5_error_handling() -> None:
    """Example 5: Proper error handling."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    from persian_transcriber.utils.exceptions import (
        TranscriberError,
        AudioProcessingError,
        EngineError,
    )

    transcriber = PersianAudioTranscriber()

    audio_files = ["audio1.mp3", "audio2.wav", "nonexistent.mp3"]

    for audio_file in audio_files:
        try:
            print(f"\nProcessing: {audio_file}")
            result = transcriber.transcribe_file(audio_file)
            print(f"✓ Success: {result['text'][:50]}...")

        except FileNotFoundError as e:
            print(f"✗ File not found: {e}")

        except AudioProcessingError as e:
            print(f"✗ Audio processing failed: {e}")

        except EngineError as e:
            print(f"✗ Transcription engine error: {e}")

        except TranscriberError as e:
            print(f"✗ General transcription error: {e}")

        except Exception as e:
            print(f"✗ Unexpected error: {e}")


def example_6_openai_api() -> None:
    """Example 6: Using OpenAI API for transcription."""
    print("\n" + "=" * 60)
    print("Example 6: OpenAI API Transcription")
    print("=" * 60)

    # Option 1: Pass API key directly
    transcriber = PersianAudioTranscriber(
        engine="openai_api",
        openai_api_key="sk-proj-YOUR_API_KEY_HERE",  # Replace with your key
    )

    # Option 2: Use environment variable (recommended)
    # Set OPENAI_API_KEY environment variable
    # transcriber = PersianAudioTranscriber(engine="openai_api")

    audio_file = "audio.mp3"
    if Path(audio_file).exists():
        print("Note: This will use OpenAI API credits")
        # result = transcriber.transcribe_file(audio_file)
        # print(f"\nTranscribed: {result['text']}")
    else:
        print(f"Audio file not found: {audio_file}")

    print("\nTo use OpenAI API:")
    print("1. Get API key from https://platform.openai.com/account/api-keys")
    print("2. Set environment variable: export OPENAI_API_KEY='sk-...'")
    print("3. Or pass key to transcriber: openai_api_key='sk-...'")


def main() -> None:
    """Run all examples."""
    print("Persian Audio Transcriber - Basic Examples")
    print("=" * 60)
    print("\nNote: These examples assume you have an 'audio.mp3' file")
    print("in the current directory. Replace with your own audio file.")

    # Run examples
    example_1_simple_transcription()
    example_2_custom_model()
    example_3_with_config_object()
    example_4_save_to_file()
    example_5_error_handling()
    example_6_openai_api()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
