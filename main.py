import os
import sys
import re
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add NVIDIA CUDA DLL paths to PATH BEFORE any GPU-related imports
# This ensures CTranslate2 (used by faster-whisper) can find the DLLs
def _setup_cuda_dll_paths():
    """Add NVIDIA CUDA DLL paths to environment PATH at module load time"""
    import site
    current_path = os.environ.get('PATH', '')
    paths_to_add = []
    
    # Priority 1: System CUDA 12.x bin directories (most reliable)
    cuda_install_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_install_base):
        # Check for CUDA 12.x versions (prioritize 12.5, 12.4, etc.)
        for version in ['12.5', '12.4', '12.3', '12.2', '12.1', '12.0', 'v12.5', 'v12.4', 'v12.3', 'v12.2', 'v12.1', 'v12.0']:
            cuda_bin = os.path.join(cuda_install_base, f'v{version}' if not version.startswith('v') else version, 'bin')
            if os.path.exists(cuda_bin) and os.path.exists(os.path.join(cuda_bin, 'cublas64_12.dll')):
                if cuda_bin not in current_path:
                    paths_to_add.insert(0, cuda_bin)  # Add at beginning (highest priority)
                break  # Use first found CUDA 12 version
    
    # Priority 2: Python package NVIDIA DLLs (installed via pip)
    try:
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(sys.prefix, 'Lib', 'site-packages')
        cublas_bin = os.path.join(site_packages, 'nvidia', 'cublas', 'bin')
        cudnn_bin = os.path.join(site_packages, 'nvidia', 'cudnn', 'bin')
        
        if os.path.exists(cublas_bin) and cublas_bin not in current_path:
            paths_to_add.append(cublas_bin)
        if os.path.exists(cudnn_bin) and cudnn_bin not in current_path:
            paths_to_add.append(cudnn_bin)
    except:
        pass  # Ignore errors during PATH setup
    
    # Add all paths to the beginning of PATH (so they're found first)
    if paths_to_add:
        new_paths = os.pathsep.join(paths_to_add)
        os.environ['PATH'] = new_paths + os.pathsep + current_path

# Set up CUDA DLL paths immediately when module loads
_setup_cuda_dll_paths()

# Persian text processing
HAZM_AVAILABLE = False
try:
    # Try to import hazm (may fail due to fasttext dependency on Python 3.13)
    from hazm import Normalizer as PersianNormalizer
    HAZM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    # Check if it's a fasttext or other dependency error
    error_msg = str(e).lower()
    if 'fasttext' in error_msg or 'module' in error_msg:
        HAZM_AVAILABLE = False
        # Don't print warning here - will be handled by BasicPersianNormalizer
    else:
        # Re-raise if it's an unexpected error
        raise
except Exception as e:
    # Catch any other errors (like AttributeError from pkgutil)
    HAZM_AVAILABLE = False


class BasicPersianNormalizer:
    """
    Basic Persian text normalizer (fallback when hazm is not available)
    Handles Arabic to Persian character conversion and basic text cleanup
    """
    def __init__(self):
        # Arabic to Persian character mappings
        self.persian_mappings = {
            'ك': 'ک',  # Arabic kaf to Persian kaf
            'ي': 'ی',  # Arabic yeh to Persian yeh
            'ى': 'ی',  # Arabic alef maksura to Persian yeh
            'ؤ': 'و',  # Arabic waw with hamza
            'أ': 'ا',  # Arabic alef with hamza above
            'ئ': 'ی',  # Arabic yeh with hamza
            'ة': 'ه',  # Arabic teh marbuta
            'إ': 'ا',  # Arabic alef with hamza below
            'ء': '',   # Arabic hamza (often removed)
            'آ': 'آ',  # Persian alef with maddah (keep as is)
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize Persian text:
        1. Convert Arabic characters to Persian
        2. Normalize whitespace (multiple spaces to single)
        3. Trim extra spaces
        """
        if not text:
            return text
        
        # Convert Arabic characters to Persian
        normalized = text
        for arabic_char, persian_char in self.persian_mappings.items():
            normalized = normalized.replace(arabic_char, persian_char)
        
        # Normalize whitespace (multiple spaces/tabs/newlines to single space)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Trim leading/trailing spaces
        normalized = normalized.strip()
        
        return normalized


# Use hazm if available, otherwise use basic normalizer
if not HAZM_AVAILABLE:
    PersianNormalizer = BasicPersianNormalizer


class PersianAudioTranscriber:
    """
    Audio transcription system with Persian language support
    Optimized for Farsi speech recognition with text normalization
    """
    
    def __init__(self, engine="whisper", model_size="medium", api_key=None, 
                 language="fa", normalize_persian=True):
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
        self.model_size = model_size
        self.api_key = api_key
        self.language = language
        self.normalize_persian = normalize_persian  # Always allow normalization (uses fallback if hazm unavailable)
        self.model = None
        
        # Initialize Persian normalizer (uses hazm if available, otherwise basic normalizer)
        if self.normalize_persian:
            self.persian_normalizer = PersianNormalizer()
            normalizer_type = "hazm" if HAZM_AVAILABLE else "basic"
            print(f"[OK] Persian text normalizer initialized ({normalizer_type})")
        
        # Supported formats
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.mp4'}
        
        # Initialize engine
        self._initialize_engine()
    
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
    
    def _add_nvidia_dll_paths(self):
        """Add NVIDIA CUDA DLL paths to environment PATH"""
        import site
        current_path = os.environ.get('PATH', '')
        paths_to_add = []
        
        # Priority 1: System CUDA 12.x bin directories (most reliable)
        cuda_install_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(cuda_install_base):
            # Check for CUDA 12.x versions (12.0, 12.1, 12.2, 12.3, 12.4, 12.5, etc.)
            for version in ['12.5', '12.4', '12.3', '12.2', '12.1', '12.0', 'v12.5', 'v12.4', 'v12.3', 'v12.2', 'v12.1', 'v12.0']:
                cuda_bin = os.path.join(cuda_install_base, f'v{version}' if not version.startswith('v') else version, 'bin')
                if os.path.exists(cuda_bin) and os.path.exists(os.path.join(cuda_bin, 'cublas64_12.dll')):
                    if cuda_bin not in current_path:
                        paths_to_add.insert(0, cuda_bin)  # Add at beginning (highest priority)
                        print(f"[INFO] Found system CUDA 12 DLLs: {cuda_bin}")
                    break  # Use first found CUDA 12 version
        
        # Priority 2: Python package NVIDIA DLLs (installed via pip)
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else os.path.join(sys.prefix, 'Lib', 'site-packages')
        cublas_bin = os.path.join(site_packages, 'nvidia', 'cublas', 'bin')
        cudnn_bin = os.path.join(site_packages, 'nvidia', 'cudnn', 'bin')
        
        if os.path.exists(cublas_bin) and cublas_bin not in current_path:
            paths_to_add.append(cublas_bin)
        if os.path.exists(cudnn_bin) and cudnn_bin not in current_path:
            paths_to_add.append(cudnn_bin)
        
        # Add all paths to the beginning of PATH (so they're found first)
        if paths_to_add:
            new_paths = os.pathsep.join(paths_to_add)
            os.environ['PATH'] = new_paths + os.pathsep + current_path
    
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
                if hasattr(self.model, 'device'):
                    device_str = str(self.model.device)
                    if 'cuda' in device_str:
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
                    self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
                    print(f"[OK] Faster-Whisper model loaded on GPU (CUDA) with FP16 precision")
                    print(f"[OK] Using device: cuda, compute_type: float16")
                except Exception as e:
                    # Check if it's a cuDNN/CUDA error
                    error_msg = str(e).lower()
                    if 'cudnn' in error_msg or 'dll' in error_msg or 'cuda' in error_msg:
                        print(f"[WARNING] GPU initialization failed - cuDNN/CUDA issue: {e}")
                        print("[INFO] cuDNN DLLs not found. Falling back to CPU...")
                        print("[INFO] For GPU acceleration, install cuDNN (see CUDNN_FIX_GUIDE.txt)")
                    else:
                        print(f"[WARNING] GPU initialization failed: {e}")
                        print("[INFO] Falling back to CPU...")
                    
                    # Fallback to CPU
                    device = "cpu"
                    compute_type = "int8"  # int8 is faster on CPU
                    try:
                        self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
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
                raise ImportError("SpeechRecognition not installed. Run: pip install SpeechRecognition")
        
        elif self.engine == "openai_api":
            if not self.api_key:
                raise ValueError("API key required for OpenAI API")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print("[OK] OpenAI API client initialized for Persian")
            except ImportError:
                raise ImportError("OpenAI not installed. Run: pip install openai")
    
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
                'ك': 'ک',  # Arabic kaf to Persian kaf
                'ي': 'ی',  # Arabic yeh to Persian yeh
                'ى': 'ی',  # Arabic alef maksura to Persian yeh
                'ؤ': 'و',  # Arabic waw with hamza
                'أ': 'ا',  # Arabic alef with hamza above
                'ئ': 'ی',  # Arabic yeh with hamza
                'ة': 'ه',  # Arabic teh marbuta
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
                
                if file_path.suffix.lower() == '.mp4':
                    try:
                        from pydub import AudioSegment
                        print("Extracting audio from MP4 file...")
                        # Extract audio from MP4 using pydub (which uses ffmpeg)
                        audio = AudioSegment.from_file(str(audio_path))
                        # Create temporary WAV file
                        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        audio.export(temp_audio_file.name, format='wav')
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
                    verbose=False
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
                    "segments": normalized_segments
                }
            
            elif self.engine == "faster_whisper":
                # Faster-Whisper transcription
                # Convert path to absolute path to avoid any path issues
                absolute_audio_path = str(Path(audio_path).resolve())
                
                try:
                    segments, info = self.model.transcribe(
                        absolute_audio_path,
                        language=target_language,
                        task="transcribe",
                        beam_size=5,
                        vad_filter=True,  # Voice Activity Detection
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                except Exception as e:
                    # Check if it's a cuDNN/CUDA error
                    error_msg = str(e).lower()
                    if 'cudnn' in error_msg or 'cuda' in error_msg or 'dll' in error_msg:
                        print(f"[WARNING] CUDA/cuDNN error during transcription: {e}")
                        print("[INFO] Reinitializing with CPU fallback...")
                        # Reinitialize with CPU
                        from faster_whisper import WhisperModel
                        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                        print("[OK] Using CPU for transcription")
                        # Retry transcription with CPU
                        segments, info = self.model.transcribe(
                            absolute_audio_path,
                            language=target_language,
                            task="transcribe",
                            beam_size=5,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                    else:
                        # Re-raise if it's a different error
                        raise
                
                # Collect all segments
                all_segments = []
                full_text = []
                
                for segment in segments:
                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": self._normalize_persian_text(segment.text)
                    }
                    all_segments.append(segment_dict)
                    full_text.append(segment_dict["text"])
                
                return {
                    "text": " ".join(full_text),
                    "language": info.language,
                    "segments": all_segments,
                    "duration": info.duration
                }
            
            elif self.engine == "google":
                import speech_recognition as sr
                from pydub import AudioSegment
                
                # Convert to WAV if needed
                if file_path.suffix.lower() != '.wav':
                    audio = AudioSegment.from_file(str(file_path))
                    wav_path = file_path.with_suffix('.wav')
                    audio.export(str(wav_path), format='wav')
                    audio_path = str(wav_path)
                
                with sr.AudioFile(audio_path) as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                
                # Recognize with Persian language code
                text = self.recognizer.recognize_google(audio_data, language="fa-IR")
                normalized_text = self._normalize_persian_text(text)
                
                return {
                    "text": normalized_text,
                    "text_raw": text,
                    "language": "fa"
                }
            
            elif self.engine == "openai_api":
                with open(audio_path, "rb") as audio_file:
                    transcription = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=target_language,
                        response_format="verbose_json"
                    )
                
                normalized_text = self._normalize_persian_text(transcription.text)
                
                return {
                    "text": normalized_text,
                    "text_raw": transcription.text,
                    "language": transcription.language,
                    "duration": transcription.duration
                }
        
        except Exception as e:
            return {"error": str(e)}
    
    def scan_and_transcribe(self, folder_path: str, output_dir: Optional[str] = None, 
                           save_format: str = "txt") -> List[Dict]:
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
        audio_files = [f for f in folder.rglob('*') 
                      if f.suffix.lower() in self.supported_formats]
        
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
            result['file'] = str(audio_file)
            results.append(result)
            
            # Save transcription
            if "error" not in result:
                output_path = self._get_output_path(audio_file, output_dir, save_format)
                self._save_transcription(result, output_path, save_format)
                print(f"[OK] Saved to: {output_path}")
                
                # Display preview
                preview_text = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                print(f"Preview: {preview_text}")
            else:
                print(f"[ERROR] {result['error']}")
        
        # Save summary
        self._save_summary(results, folder, output_dir)
        
        return results
    
    def _get_output_path(self, audio_file: Path, output_dir: Optional[str], 
                        save_format: str) -> Path:
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
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
        
        elif save_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif save_format == "srt":
            # Generate SRT subtitle format with timestamps
            with open(output_path, 'w', encoding='utf-8') as f:
                if 'segments' in result:
                    for i, segment in enumerate(result['segments'], 1):
                        start = self._format_timestamp(segment['start'])
                        end = self._format_timestamp(segment['end'])
                        f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
                else:
                    f.write(result['text'])
    
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
        summary_file = summary_path / f"persian_transcription_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "engine": self.engine,
            "model_size": self.model_size,
            "language": self.language,
            "persian_normalization": self.normalize_persian,
            "total_files": len(results),
            "successful": sum(1 for r in results if "error" not in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
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
        """
    )
    
    parser.add_argument("input", help="Audio file or folder path")
    parser.add_argument("--engine", choices=["whisper", "faster_whisper", "google", "openai_api"], 
                       default="whisper", help="Transcription engine (default: whisper)")
    parser.add_argument("--model", default="medium", 
                       help="Model size for Whisper: tiny/base/small/medium/large/large-v3 (default: medium for Persian)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--format", choices=["txt", "json", "srt"], 
                       default="txt", help="Output format (default: txt)")
    parser.add_argument("--api-key", help="OpenAI API key (for openai_api engine)")
    parser.add_argument("--no-normalize", action="store_true", 
                       help="Disable Persian text normalization")
    parser.add_argument("--language", default="fa", 
                       help="Language code (default: fa for Persian/Farsi)")
    
    args = parser.parse_args()
    
    # Create transcriber
    transcriber = PersianAudioTranscriber(
        engine=args.engine,
        model_size=args.model,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        language=args.language,
        normalize_persian=not args.no_normalize
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
            print(result['text'])
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
            str(input_path),
            output_dir=args.output,
            save_format=args.format
        )
    
    else:
        print(f"Error: Invalid path: {args.input}")
