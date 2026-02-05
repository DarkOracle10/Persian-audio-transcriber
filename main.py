import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.persian_transcriber.utils.cuda_setup import setup_cuda_dll_paths
from src.persian_transcriber.normalizers.basic import BasicNormalizer

# Configure CUDA library paths before importing GPU-dependent modules.
try:
    setup_cuda_dll_paths()
except (RuntimeError, ValueError) as cuda_error:
    print(f"[WARNING] CUDA path configuration skipped: {cuda_error}")

# Persian text processing
HAZM_AVAILABLE = False
try:
    # Try to import hazm (may fail due to fasttext dependency on Python 3.13)
    from hazm import Normalizer as PersianNormalizer

    HAZM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    # Check if it's a fasttext or other dependency error
    error_msg = str(e).lower()
    if "fasttext" in error_msg or "module" in error_msg:
        HAZM_AVAILABLE = False
        # Don't print warning here - will be handled by BasicPersianNormalizer
    else:
        # Re-raise if it's an unexpected error
        raise
except Exception as e:
    # Catch any other errors (like AttributeError from pkgutil)
    HAZM_AVAILABLE = False
# Use hazm if available, otherwise use basic normalizer
if not HAZM_AVAILABLE:
    PersianNormalizer = BasicNormalizer


# Persian initial prompts to help guide transcription output
# These help the model produce proper Persian script instead of Arabic
PERSIAN_INITIAL_PROMPTS = {
    "fa": "این یک متن فارسی است.",  # "This is a Persian text."
    "fa-lecture": "بسم الله الرحمن الرحیم. این متن فارسی در مورد موضوعات آموزشی است.",
    "fa-conversation": "سلام، خوبی؟ این یک مکالمه فارسی است.",
}

# Recommended minimum model sizes for different languages
# Persian/Arabic script languages need larger models for accuracy
MINIMUM_MODEL_SIZES = {
    "fa": "medium",  # Persian needs at least medium
    "ar": "medium",  # Arabic needs at least medium
    "zh": "medium",  # Chinese needs at least medium
    "ja": "medium",  # Japanese needs at least medium
    "ko": "medium",  # Korean needs at least medium
    "default": "small",
}

# Model size ranking for comparison
MODEL_SIZE_RANK = {
    "tiny": 1,
    "tiny.en": 1,
    "base": 2,
    "base.en": 2,
    "small": 3,
    "small.en": 3,
    "medium": 4,
    "medium.en": 4,
    "large": 5,
    "large-v1": 5,
    "large-v2": 5,
    "large-v3": 6,
    "turbo": 5,
    "distil-large-v2": 5,
    "distil-large-v3": 5,
}


class PersianAudioTranscriber:
    """
    Audio transcription system with Persian language support
    Optimized for Farsi speech recognition with text normalization
    """

    def __init__(
        self,
        engine="whisper",
        model_size="medium",
        api_key=None,
        language="fa",
        normalize_persian=True,
        initial_prompt=None,
    ):
        """
        Initialize transcriber with Persian language support

        Parameters:
        engine (str): 'whisper', 'google', 'openai_api', or 'faster_whisper'
        model_size (str): For whisper - 'tiny', 'base', 'small', 'medium', 'large', 'large-v3'
                         For Persian, 'medium' or larger recommended for best accuracy
        api_key (str): For openai_api
        language (str): Language code - 'fa' for Persian/Farsi
        normalize_persian (bool): Apply Persian text normalization using hazm
        """
        self.engine = engine
        self.api_key = api_key
        self.language = language
        self.normalize_persian = (
            normalize_persian  # Always allow normalization (uses fallback if hazm unavailable)
        )
        self.model = None

        # Validate and potentially upgrade model size for the target language
        self.model_size = self._validate_model_size(model_size, language)

        # Set initial prompt for better language-specific transcription
        if initial_prompt is None:
            self.initial_prompt = PERSIAN_INITIAL_PROMPTS.get(
                language, PERSIAN_INITIAL_PROMPTS.get("fa", "")
            )
        else:
            self.initial_prompt = initial_prompt

        # Initialize Persian normalizer (uses hazm if available, otherwise basic normalizer)
        if self.normalize_persian:
            self.persian_normalizer = PersianNormalizer()
            normalizer_type = "hazm" if HAZM_AVAILABLE else "basic"
            print(f"[OK] Persian text normalizer initialized ({normalizer_type})")

        # Supported formats
        self.supported_formats = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}

        # Initialize engine
        self._initialize_engine()

    def _validate_model_size(self, requested_size: str, language: str) -> str:
        """
        Validate and potentially upgrade model size based on language requirements.

        Persian, Arabic, Chinese, Japanese, and Korean languages require larger models
        for acceptable transcription quality. Tiny/base models produce garbage output
        for these languages.

        Args:
            requested_size: The model size requested by the user
            language: The target language code (e.g., 'fa' for Persian)

        Returns:
            The validated (possibly upgraded) model size
        """
        minimum_size = MINIMUM_MODEL_SIZES.get(language, MINIMUM_MODEL_SIZES["default"])

        requested_rank = MODEL_SIZE_RANK.get(requested_size.lower(), 0)
        minimum_rank = MODEL_SIZE_RANK.get(minimum_size, 0)

        if requested_rank < minimum_rank:
            print(f"[WARNING] Model '{requested_size}' is too small for {language} transcription.")
            print(f"[INFO] Upgrading to '{minimum_size}' model for better accuracy.")
            print(f"[INFO] For best Persian results, use 'large-v3' model.")
            return minimum_size

        return requested_size

    def _check_cuda_available(self):
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                print(f"[INFO] CUDA detected: {device_name}")
                return True
            else:
                print("[INFO] CUDA not available, using CPU")
                return False
        except ImportError:
            print("[INFO] PyTorch not available for CUDA check")
            return False

    def _add_nvidia_dll_paths(self) -> bool:
        """Ensure CUDA runtime libraries are visible to the current process."""

        try:
            configured = setup_cuda_dll_paths()
        except (RuntimeError, ValueError) as exc:
            print(f"[WARNING] CUDA library path configuration failed: {exc}")
            return False

        if configured:
            print("[INFO] CUDA library paths configured.")
        else:
            print("[INFO] No CUDA library paths configured; continuing with CPU fallback.")
        return configured

    def _initialize_engine(self):
        """Initialize the selected transcription engine"""

        # DLL paths already added in __init__ before any imports

        if self.engine == "whisper":
            try:
                import whisper

                print(f"Loading Whisper {self.model_size} model for Persian...")

                # Check for CUDA/GPU availability
                cuda_available = self._check_cuda_available()

                # Load model (Whisper automatically uses GPU if available and PyTorch is installed)
                self.model = whisper.load_model(self.model_size)

                # Check which device is actually being used
                if hasattr(self.model, "device"):
                    device_str = str(self.model.device)
                    if "cuda" in device_str:
                        print(f"[OK] Whisper model loaded on GPU: {device_str}")
                    else:
                        print(f"[OK] Whisper model loaded on CPU: {device_str}")
                else:
                    print("[OK] Whisper model loaded")

                print(f"[OK] Persian (Farsi) language support enabled (code: {self.language})")
            except ImportError:
                raise ImportError("Whisper not installed. Run: pip install openai-whisper")

        elif self.engine == "faster_whisper":
            try:
                from faster_whisper import WhisperModel

                print(f"Loading Faster-Whisper {self.model_size} model for Persian...")

                # Try CUDA first, fallback to CPU if not available
                device = "cuda"
                compute_type = "float16"

                try:
                    # Attempt to initialize with CUDA and FP16
                    self.model = WhisperModel(
                        self.model_size, device=device, compute_type=compute_type
                    )
                    print(f"[OK] Faster-Whisper model loaded on GPU (CUDA) with FP16 precision")
                    print(f"[OK] Using device: cuda, compute_type: float16")
                except Exception as e:
                    # Check if it's a cuDNN/CUDA error
                    error_msg = str(e).lower()
                    if "cudnn" in error_msg or "dll" in error_msg or "cuda" in error_msg:
                        print(f"[WARNING] GPU initialization failed - cuDNN/CUDA issue: {e}")
                        print("[INFO] cuDNN DLLs not found. Falling back to CPU...")
                        print(
                            "[INFO] For GPU acceleration, install cuDNN (see CUDNN_FIX_GUIDE.txt)"
                        )
                    else:
                        print(f"[WARNING] GPU initialization failed: {e}")
                        print("[INFO] Falling back to CPU...")

                    # Fallback to CPU
                    device = "cpu"
                    compute_type = "int8"  # int8 is faster on CPU
                    try:
                        self.model = WhisperModel(
                            self.model_size, device=device, compute_type=compute_type
                        )
                        print(f"[OK] Faster-Whisper model loaded on CPU")
                        print(f"[OK] Using device: cpu, compute_type: int8")
                        print("[INFO] Transcription will continue but may be slower without GPU")
                    except Exception as cpu_error:
                        print(f"[ERROR] CPU initialization also failed: {cpu_error}")
                        raise

            except ImportError:
                raise ImportError("Faster-Whisper not installed. Run: pip install faster-whisper")

        elif self.engine == "google":
            try:
                import speech_recognition as sr

                self.recognizer = sr.Recognizer()
                print("[OK] Google Speech Recognition initialized")
                print("Note: Google may have limited Persian support compared to Whisper")
            except ImportError:
                raise ImportError(
                    "SpeechRecognition not installed. Run: pip install SpeechRecognition"
                )

        elif self.engine == "openai_api":
            if not self.api_key:
                raise ValueError("API key required for OpenAI API")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
                print("[OK] OpenAI API client initialized for Persian")
            except ImportError:
                raise ImportError("OpenAI not installed. Run: pip install openai")

    def _remove_repetitions(self, text: str, min_repeat_len: int = 5, max_repeats: int = 2) -> str:
        """
        Post-process text to remove excessive repetitions (hallucinations).

        This handles cases where Whisper gets stuck in a loop repeating phrases.

        Args:
            text: The transcribed text
            min_repeat_len: Minimum length of phrase to check for repetition
            max_repeats: Maximum allowed repetitions before removal

        Returns:
            Text with excessive repetitions removed
        """
        import re

        if not text or len(text) < min_repeat_len * 2:
            return text

        # Split into words
        words = text.split()
        if len(words) < 3:
            return text

        # Method 1: Remove consecutive duplicate phrases
        # Look for patterns like "word word word word word word" (same word repeated)
        cleaned_words = []
        i = 0
        while i < len(words):
            word = words[i]
            # Count consecutive repeats
            repeat_count = 1
            while i + repeat_count < len(words) and words[i + repeat_count] == word:
                repeat_count += 1

            # Keep only up to max_repeats
            for _ in range(min(repeat_count, max_repeats)):
                cleaned_words.append(word)
            i += repeat_count

        # Method 2: Remove repeated multi-word phrases
        text = " ".join(cleaned_words)

        # Pattern to find repeated phrases (3+ words repeated 3+ times)
        # This regex finds phrases that are repeated consecutively
        for phrase_len in range(8, 2, -1):  # Check longer phrases first
            pattern = r"((?:\S+\s+){" + str(phrase_len - 1) + r"}\S+)\s+(\1\s*){2,}"
            text = re.sub(pattern, r"\1 ", text)

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _format_persian_text(self, text: str) -> str:
        """
        Format Persian text with proper punctuation and spacing.

        Args:
            text: Raw transcribed text

        Returns:
            Formatted text with proper Persian punctuation
        """
        import re

        if not text:
            return text

        # Persian punctuation marks
        # Add space after punctuation if missing
        text = re.sub(r"([،؟!:.])([^\s])", r"\1 \2", text)

        # Fix spacing around Persian-specific punctuation
        text = re.sub(r"\s+([،؟!:.])", r"\1", text)  # No space before punctuation

        # Add proper sentence breaks for readability
        # Split on Persian question mark and exclamation
        text = re.sub(r"([؟!.])\s*", r"\1\n", text)

        # Clean up multiple newlines
        text = re.sub(r"\n\s*\n", "\n", text)
        text = re.sub(r"\n+", "\n", text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)

        return text

    def _normalize_persian_text(self, text: str) -> str:
        """
        Normalize Persian text using hazm library
        Handles Arabic characters, spacing, and Persian-specific corrections
        """
        if not self.normalize_persian or not text:
            return text

        try:
            # Apply Persian normalization
            normalized = self.persian_normalizer.normalize(text)

            # Additional Persian-specific corrections
            # Convert Arabic characters to Persian equivalents
            persian_mappings = {
                "ك": "ک",  # Arabic kaf to Persian kaf
                "ي": "ی",  # Arabic yeh to Persian yeh
                "ى": "ی",  # Arabic alef maksura to Persian yeh
                "ؤ": "و",  # Arabic waw with hamza
                "أ": "ا",  # Arabic alef with hamza above
                "ئ": "ی",  # Arabic yeh with hamza
                "ة": "ه",  # Arabic teh marbuta
            }

            for arabic_char, persian_char in persian_mappings.items():
                normalized = normalized.replace(arabic_char, persian_char)

            return normalized
        except Exception as e:
            print(f"Warning: Persian normalization failed: {e}")
            return text

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Transcribe a single audio file with Persian support

        Parameters:
        audio_path (str): Path to audio file
        language (str): Override language code (default uses instance language)

        Returns:
        dict: Contains 'text', 'language', 'segments', etc.
        """

        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}

        file_path = Path(audio_path)

        if file_path.suffix.lower() not in self.supported_formats:
            return {"error": f"Unsupported format: {file_path.suffix}"}

        # Use specified language or default to Persian
        target_language = language or self.language

        print(f"\nTranscribing: {file_path.name}")
        print(f"Language: {target_language} (Persian/Farsi)")

        try:
            if self.engine == "whisper":
                # Handle MP4 files by extracting audio first (more reliable on Windows)
                processing_path = str(audio_path)
                temp_audio_file = None

                if file_path.suffix.lower() == ".mp4":
                    try:
                        from pydub import AudioSegment

                        print("Extracting audio from MP4 file...")
                        # Extract audio from MP4 using pydub (which uses ffmpeg)
                        audio = AudioSegment.from_file(str(audio_path))
                        # Create temporary WAV file
                        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        audio.export(temp_audio_file.name, format="wav")
                        processing_path = temp_audio_file.name
                        print("Audio extracted successfully")
                    except Exception as e:
                        print(f"Warning: Could not extract audio from MP4: {e}")
                        print("Trying direct MP4 processing...")
                        # Fall back to direct processing
                        processing_path = str(Path(audio_path).resolve())

                # Transcribe with language parameter
                result = self.model.transcribe(
                    processing_path,
                    language=target_language,
                    task="transcribe",  # Use 'translate' for English translation
                    verbose=False,
                )

                # Clean up temporary file if created
                if temp_audio_file and os.path.exists(temp_audio_file.name):
                    try:
                        os.unlink(temp_audio_file.name)
                    except:
                        pass

                # Normalize Persian text
                normalized_text = self._normalize_persian_text(result["text"])

                # Normalize segments
                normalized_segments = []
                for segment in result.get("segments", []):
                    normalized_segment = segment.copy()
                    normalized_segment["text"] = self._normalize_persian_text(segment["text"])
                    normalized_segments.append(normalized_segment)

                return {
                    "text": normalized_text,
                    "text_raw": result["text"],  # Original before normalization
                    "language": result.get("language", target_language),
                    "segments": normalized_segments,
                }

            elif self.engine == "faster_whisper":
                # Faster-Whisper transcription
                # Convert path to absolute path to avoid any path issues
                absolute_audio_path = str(Path(audio_path).resolve())

                # Transcription parameters optimized for Persian/Farsi
                # with anti-hallucination and anti-repetition settings
                transcribe_params = {
                    "language": target_language,
                    "task": "transcribe",
                    "beam_size": 5,
                    "best_of": 5,
                    "patience": 1.0,
                    "temperature": 0.0,  # Greedy decoding for consistency
                    "vad_filter": True,  # Voice Activity Detection
                    "vad_parameters": {"min_silence_duration_ms": 500},
                    "condition_on_previous_text": True,  # Better context continuity
                    "initial_prompt": self.initial_prompt,  # Critical for Persian!
                    "compression_ratio_threshold": 2.4,
                    "log_prob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    # Anti-repetition settings
                    "repetition_penalty": 1.1,  # Penalize repeated tokens (>1 to penalize)
                    "no_repeat_ngram_size": 3,  # Prevent 3-gram repetitions
                    # Hallucination detection
                    "word_timestamps": True,  # Required for hallucination detection
                    "hallucination_silence_threshold": 2.0,  # Skip silent periods >2s during hallucinations
                }

                try:
                    segments, info = self.model.transcribe(absolute_audio_path, **transcribe_params)
                except Exception as e:
                    # Check if it's a cuDNN/CUDA error
                    error_msg = str(e).lower()
                    if "cudnn" in error_msg or "cuda" in error_msg or "dll" in error_msg:
                        print(f"[WARNING] CUDA/cuDNN error during transcription: {e}")
                        print("[INFO] Reinitializing with CPU fallback...")
                        # Reinitialize with CPU
                        from faster_whisper import WhisperModel

                        self.model = WhisperModel(
                            self.model_size, device="cpu", compute_type="int8"
                        )
                        print("[OK] Using CPU for transcription")
                        # Retry transcription with CPU
                        segments, info = self.model.transcribe(
                            absolute_audio_path, **transcribe_params
                        )
                    else:
                        # Re-raise if it's a different error
                        raise

                # Collect all segments
                all_segments = []
                full_text = []

                for segment in segments:
                    # Normalize and clean the segment text
                    normalized_text = self._normalize_persian_text(segment.text)
                    cleaned_text = self._remove_repetitions(normalized_text)

                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": cleaned_text,
                    }
                    all_segments.append(segment_dict)
                    full_text.append(cleaned_text)

                # Join and post-process full text
                combined_text = " ".join(full_text)
                combined_text = self._remove_repetitions(combined_text)
                combined_text = self._format_persian_text(combined_text)

                return {
                    "text": combined_text,
                    "language": info.language,
                    "segments": all_segments,
                    "duration": info.duration,
                }

            elif self.engine == "google":
                import speech_recognition as sr
                from pydub import AudioSegment

                # Convert to WAV if needed
                if file_path.suffix.lower() != ".wav":
                    audio = AudioSegment.from_file(str(file_path))
                    wav_path = file_path.with_suffix(".wav")
                    audio.export(str(wav_path), format="wav")
                    audio_path = str(wav_path)

                with sr.AudioFile(audio_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)

                # Recognize with Persian language code
                text = self.recognizer.recognize_google(audio_data, language="fa-IR")
                normalized_text = self._normalize_persian_text(text)

                return {"text": normalized_text, "text_raw": text, "language": "fa"}

            elif self.engine == "openai_api":
                with open(audio_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=target_language,
                        response_format="verbose_json",
                    )

                normalized_text = self._normalize_persian_text(transcription.text)

                return {
                    "text": normalized_text,
                    "text_raw": transcription.text,
                    "language": transcription.language,
                    "duration": transcription.duration,
                }

        except Exception as e:
            return {"error": str(e)}

    def scan_and_transcribe(
        self, folder_path: str, output_dir: Optional[str] = None, save_format: str = "txt"
    ) -> List[Dict]:
        """
        Scan folder and transcribe all audio files with Persian support

        Parameters:
        folder_path (str): Path to folder
        output_dir (str): Where to save transcriptions (default: same as audio)
        save_format (str): 'txt', 'json', or 'srt'

        Returns:
        list: Results for each file
        """

        folder = Path(folder_path)

        if not folder.exists():
            print(f"Error: Folder not found: {folder_path}")
            return []

        # Find all audio files
        audio_files = [f for f in folder.rglob("*") if f.suffix.lower() in self.supported_formats]

        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return []

        print(f"\n{'='*60}")
        print(f"Persian Audio Transcription")
        print(f"{'='*60}")
        print(f"Found {len(audio_files)} audio files")
        print(f"Engine: {self.engine}")
        print(f"Language: {self.language} (Persian/Farsi)")
        print(f"Output format: {save_format}")
        print(f"Persian normalization: {'Enabled' if self.normalize_persian else 'Disabled'}")
        print(f"{'='*60}")

        results = []

        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            # Transcribe
            result = self.transcribe_file(str(audio_file))
            result["file"] = str(audio_file)
            results.append(result)

            # Save transcription
            if "error" not in result:
                output_path = self._get_output_path(audio_file, output_dir, save_format)
                self._save_transcription(result, output_path, save_format)
                print(f"[OK] Saved to: {output_path}")

                # Display preview
                preview_text = (
                    result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                )
                print(f"Preview: {preview_text}")
            else:
                print(f"[ERROR] {result['error']}")

        # Save summary
        self._save_summary(results, folder, output_dir)

        return results

    def _get_output_path(
        self, audio_file: Path, output_dir: Optional[str], save_format: str
    ) -> Path:
        """Determine output file path"""

        if output_dir:
            output_folder = Path(output_dir)
            output_folder.mkdir(parents=True, exist_ok=True)
            return output_folder / f"{audio_file.stem}.{save_format}"
        else:
            return audio_file.with_suffix(f".{save_format}")

    def _save_transcription(self, result: Dict, output_path: Path, save_format: str):
        """Save transcription to file with UTF-8 encoding for Persian"""

        if save_format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

        elif save_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        elif save_format == "srt":
            # Generate SRT subtitle format with timestamps
            with open(output_path, "w", encoding="utf-8") as f:
                if "segments" in result:
                    for i, segment in enumerate(result["segments"], 1):
                        start = self._format_timestamp(segment["start"])
                        end = self._format_timestamp(segment["end"])
                        f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
                else:
                    f.write(result["text"])

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _save_summary(self, results: List[Dict], folder: Path, output_dir: Optional[str]):
        """Save summary of all transcriptions"""

        summary_path = Path(output_dir) if output_dir else folder
        summary_file = (
            summary_path
            / f"persian_transcription_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "engine": self.engine,
            "model_size": self.model_size,
            "language": self.language,
            "persian_normalization": self.normalize_persian,
            "total_files": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Summary saved to: {summary_file}")
        print(f"Successful: {summary['successful']}/{summary['total_files']}")
        print(f"{'='*60}")


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Persian Audio Transcription Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe Persian audio with Whisper
  python persian_transcriber.py audio.mp3 --engine whisper --model medium
  
  # Batch transcribe folder
  python persian_transcriber.py ./audio_folder --output ./transcriptions
  
  # Use Faster-Whisper for better performance
  python persian_transcriber.py audio.mp3 --engine faster_whisper --model large-v3
  
  # Create Persian subtitles
  python persian_transcriber.py video.mp4 --format srt
        """,
    )

    parser.add_argument("input", help="Audio file or folder path")
    parser.add_argument(
        "--engine",
        choices=["whisper", "faster_whisper", "google", "openai_api"],
        default="faster_whisper",
        help="Transcription engine (default: faster_whisper for GPU acceleration)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Model size: tiny/base/small/medium/large-v3 (default: large-v3 for best Persian accuracy)",
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--format",
        choices=["txt", "json", "srt"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument("--api-key", help="OpenAI API key (for openai_api engine)")
    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable Persian text normalization"
    )
    parser.add_argument(
        "--language", default="fa", help="Language code (default: fa for Persian/Farsi)"
    )
    parser.add_argument(
        "--initial-prompt",
        help="Initial prompt to guide transcription style (helps distinguish Persian from Arabic)",
    )

    args = parser.parse_args()

    # Show model recommendation for Persian
    if args.language == "fa" and args.model in ["tiny", "base"]:
        print(
            f"\n[IMPORTANT] For Persian transcription, '{args.model}' model produces poor results."
        )
        print(
            "[IMPORTANT] Use 'medium', 'large-v2', or 'large-v3' for accurate Persian transcription.\n"
        )

    # Create transcriber
    transcriber = PersianAudioTranscriber(
        engine=args.engine,
        model_size=args.model,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        language=args.language,
        normalize_persian=not args.no_normalize,
        initial_prompt=args.initial_prompt,
    )

    # Check if input is file or folder
    input_path = Path(args.input)

    if input_path.is_file():
        result = transcriber.transcribe_file(str(input_path))
        if "error" not in result:
            # Display transcription result
            print(f"\n{'='*60}")
            print("Persian Transcription Result:")
            print(f"{'='*60}")
            print(result["text"])
            print(f"\nLanguage detected: {result.get('language', 'unknown')}")

            # Save transcription to file
            output_path = transcriber._get_output_path(input_path, args.output, args.format)
            transcriber._save_transcription(result, output_path, args.format)
            print(f"\n{'='*60}")
            print(f"Transcription saved to: {output_path}")
            print(f"Format: {args.format}")
            print(f"{'='*60}")
        else:
            print(f"Error: {result['error']}")

    elif input_path.is_dir():
        transcriber.scan_and_transcribe(
            str(input_path), output_dir=args.output, save_format=args.format
        )

    else:
        print(f"Error: Invalid path: {args.input}")
