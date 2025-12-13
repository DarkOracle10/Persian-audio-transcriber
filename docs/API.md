# API Documentation

Python API reference for programmatic use of the Voice Transcription toolkit.

---

## Table of Contents

1. [Transcription Engines](#transcription-engines)
2. [Transcription Manager](#transcription-manager)
3. [Text Normalizer](#text-normalizer)
4. [CUDA Utilities](#cuda-utilities)
5. [CLI Module](#cli-module)
6. [Complete Examples](#complete-examples)
7. [Error Handling](#error-handling)

---

## Transcription Engines

### TranscriptionEngine (Abstract Base)

**Module:** `src.engines.base`

Base class defining the engine interface.

```python
from src.engines.base import TranscriptionEngine
from typing import Iterable, List

class TranscriptionEngine(ABC):
    def __init__(self, language: str = "en") -> None:
        """Initialize engine with default language."""
        
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file."""
        
    def transcribe_batch(self, audio_paths: Iterable[str]) -> List[str]:
        """Transcribe multiple files sequentially."""
        
    def validate_audio_path(self, audio_path: str) -> Path:
        """Validate file exists and format is supported."""
```

**Properties:**
- `supported_formats`: Tuple of supported file extensions (`.wav`, `.mp3`, `.flac`, etc.)

---

### OfflineTranscriptionEngine

**Module:** `src.engines.offline`

Local transcription using torchaudio Wav2Vec2 ASR.

#### Initialization

```python
from src.engines.offline import OfflineTranscriptionEngine

engine = OfflineTranscriptionEngine(
    language="en",
    device="cuda"  # Or "cpu", or None for auto-detect
)
```

#### Parameters

- `language` (str): Language code for decoding (default: `"en"`)
- `device` (Optional[str]): Compute device override. If `None`, auto-selects CUDA if available, else CPU.

#### Methods

##### `transcribe(audio_path: str) -> str`

Transcribe a single audio file.

```python
transcript = engine.transcribe("audio.wav")
print(transcript)
# Output: "hello world this is a test"
```

**Returns:** Transcribed text as string

**Raises:**
- `FileNotFoundError`: If audio file doesn't exist
- `ValueError`: If audio format is unsupported
- `RuntimeError`: If torch/torchaudio are not installed

#### Example with Device Selection

```python
# Auto-detect device
engine_auto = OfflineTranscriptionEngine()

# Force CPU
engine_cpu = OfflineTranscriptionEngine(device="cpu")

# Force CUDA
engine_cuda = OfflineTranscriptionEngine(device="cuda")
```

#### Example Output

```python
engine = OfflineTranscriptionEngine(language="en")
result = engine.transcribe("sample.wav")
# result: "the quick brown fox jumps over the lazy dog"
```

---

### OpenAITranscriptionEngine

**Module:** `src.engines.openai`

Cloud-based transcription via OpenAI Whisper API.

#### Initialization

```python
from src.engines.openai import OpenAITranscriptionEngine
import os

engine = OpenAITranscriptionEngine(
    api_key=os.getenv("OPENAI_API_KEY"),
    language="fa"
)
```

#### Parameters

- `api_key` (Optional[str]): OpenAI API key. Falls back to `OPENAI_API_KEY` environment variable.
- `language` (str): Language code (default: `"en"`)

**Raises:**
- `ValueError`: If no API key is provided (init time)

#### Methods

##### `transcribe(audio_path: str) -> str`

Transcribe with automatic retry and rate limiting.

```python
try:
    transcript = engine.transcribe("audio.mp3")
    print(transcript)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded (retries exhausted)")
```

**Returns:** Transcribed text

**Raises:**
- `FileNotFoundError`: Audio file not found
- `ValueError`: Unsupported audio format
- `AuthenticationError`: Invalid API key
- `RateLimitError`: Quota exceeded (after 3 retries)
- `APIError`: Server error (after 3 retries)

#### Rate Limiting

- **Max 3 requests per second** (rolling window)
- Automatically throttles requests to stay within limit
- Uses `time.monotonic()` for accurate timing

#### Retry Logic

```python
# Automatically retries on RateLimitError or APIError
# - Attempt 1: immediate
# - Attempt 2: 2 second delay
# - Attempt 3: 4 second delay
# After 3 attempts, raises final exception
```

#### Example with Error Handling

```python
from openai import AuthenticationError, RateLimitError

engine = OpenAITranscriptionEngine(api_key="sk-...")

try:
    transcript = engine.transcribe("audio.mp3")
    print(f"Success: {transcript}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Quota exceeded")
```

---

## Transcription Manager

**Module:** `src.transcriber`

High-level orchestration for batch transcription with parallel processing.

### TranscriptionManager

#### Initialization

```python
from src.transcriber import TranscriptionManager
from src.engines.openai import OpenAITranscriptionEngine

engine = OpenAITranscriptionEngine(api_key="sk-...")
manager = TranscriptionManager(
    engine=engine,
    max_workers=4,
    batch_size=10
)
```

#### Parameters

- `engine` (TranscriptionEngine): Engine instance to use
- `max_workers` (int): Maximum parallel workers (default: 4)
- `batch_size` (int): Results to buffer before CSV write (default: 10)

#### Methods

##### `find_audio_files(folder_path: str, extensions: Optional[List[str]]) -> List[Path]`

Discover audio files in a directory.

```python
files = manager.find_audio_files(
    folder_path="./audio",
    extensions=[".wav", ".mp3"]
)
print(f"Found {len(files)} files")
```

**Returns:** Sorted list of Path objects

**Raises:**
- `FileNotFoundError`: If folder doesn't exist

##### `transcribe_file(file_path: Path) -> Dict[str, Any]`

Transcribe a single file (used internally).

```python
result = manager.transcribe_file(Path("audio.mp3"))
print(result)
# {
#     "file_path": "audio.mp3",
#     "status": "success",
#     "transcript": "transcribed text",
#     "error_msg": None
# }
```

**Returns:** Result dictionary

##### `transcribe_batch(folder_path: str, output_csv: str, extensions: Optional[List[str]]) -> None`

Batch transcribe with parallel processing and progress bar.

```python
manager.transcribe_batch(
    folder_path="./audio",
    output_csv="results.csv",
    extensions=[".wav", ".mp3", ".ogg"]
)
```

**Parameters:**
- `folder_path` (str): Directory containing audio files
- `output_csv` (str): Output CSV file path
- `extensions` (Optional[List[str]]): File extensions to include (default: `[".wav", ".mp3", ".ogg"]`)

**CSV Output Format:**
```csv
file_path,status,transcript,error_msg
audio1.wav,success,"hello world",
audio2.mp3,failed,,File not found: audio2.mp3
```

#### Complete Example

```python
from src.transcriber import TranscriptionManager
from src.engines.offline import OfflineTranscriptionEngine

# Setup
engine = OfflineTranscriptionEngine(device="cuda")
manager = TranscriptionManager(
    engine=engine,
    max_workers=4,
    batch_size=10
)

# Run batch transcription
manager.transcribe_batch(
    folder_path="./audio_files",
    output_csv="transcripts.csv"
)

# Output:
# Transcribing files: 100%|████████████| 50/50 [00:25<00:00,  2.00file/s]
# INFO: Transcription complete. Results saved to transcripts.csv
```

---

## Text Normalizer

**Module:** `src.utils.normalizer`

### BasicPersianNormalizer

Persian/Farsi text normalization (Arabic script → Persian script, whitespace cleanup).

#### Initialization

```python
from src.utils.normalizer import BasicPersianNormalizer

normalizer = BasicPersianNormalizer(
    convert_numerals=True,
    lowercase=False
)
```

#### Parameters

- `convert_numerals` (bool): Convert Arabic numerals (٠-٩) to Persian (۰-۹). Default: `True`
- `lowercase` (bool): Lowercase text after normalization. Default: `False`
- `character_map` (Optional[Mapping[str, str]]): Custom character overrides

#### Methods

##### `normalize(text: str) -> str`

Full normalization: character conversion + whitespace cleanup.

```python
text = "سلام   دنيا"  # Arabic yeh, multiple spaces
result = normalizer.normalize(text)
print(result)
# Output: "سلام دنیا"  # Persian yeh, single space
```

**Returns:** Normalized string

##### `normalize_characters_only(text: str) -> str`

Character conversion only (no whitespace changes).

```python
text = "سلام   دنيا"
result = normalizer.normalize_characters_only(text)
print(result)
# Output: "سلام   دنیا"  # Persian yeh, spaces preserved
```

#### Character Mappings

**Arabic → Persian:**
```python
{
    "ك": "ک",  # kaf
    "ي": "ی",  # yeh
    "ى": "ی",  # alef maksura
    "ؤ": "و",  # waw with hamza
    "أ": "ا",  # alef with hamza
    "ئ": "ی",  # yeh with hamza
    "ة": "ه",  # teh marbuta
    "إ": "ا",  # alef with hamza below
    "ء": "",   # hamza (removed)
}
```

**Arabic Numerals → Persian:**
```python
{
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹"
}
```

#### Example: Custom Mappings

```python
normalizer = BasicPersianNormalizer(
    character_map={"@": ""}  # Remove @ symbols
)
result = normalizer.normalize("سلام@دنیا")
# Output: "سلامدنیا"
```

---

## CUDA Utilities

**Module:** `src.utils.cuda_setup`

### setup_cuda_dll_paths()

Configure CUDA library search paths for the current platform.

```python
from src.utils.cuda_setup import setup_cuda_dll_paths

configured = setup_cuda_dll_paths(
    cuda_home="/usr/local/cuda",  # Optional
    config_path=Path("config.yaml")  # Optional
)

if configured:
    print("CUDA libraries configured")
else:
    print("No CUDA libraries found")
```

#### Parameters

- `cuda_home` (Optional[str]): CUDA installation directory. Overrides config file.
- `config_path` (Optional[Path]): Explicit path to configuration file.

**Returns:** `True` if libraries were found, `False` otherwise

**Raises:**
- `ValueError`: If `cuda_home` is provided but doesn't exist
- `RuntimeError`: If platform is unsupported

#### Platform Behavior

**Windows:**
- Modifies `PATH` environment variable
- Searches for `bin/` subdirectory

**Linux:**
- Modifies `LD_LIBRARY_PATH` and `PATH`
- Searches for `lib64/` and `lib/` subdirectories

**macOS:**
- Modifies `DYLD_LIBRARY_PATH` and `PATH`
- Searches for `lib/` subdirectory

#### Example: Auto-Configuration

```python
from src.utils.cuda_setup import setup_cuda_dll_paths

# Automatically finds CUDA via:
# 1. config.yaml (current dir, repo root, ~/.persian_transcriber)
# 2. CUDA_HOME environment variable
configured = setup_cuda_dll_paths()

if configured:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## CLI Module

**Module:** `src.cli`

### Config.load()

Load configuration from YAML file.

```python
from src.cli import Config

config = Config.load("config.yaml")
print(config.openai_api_key)
print(config.offline)
```

**Returns:** `Config` dataclass with:
- `openai_api_key` (Optional[str])
- `offline` (Dict[str, Any])

### main()

CLI entry point (used by `python -m src`).

```python
from src.cli import main

exit_code = main(["transcribe", "--folder", "./audio", "--output", "results.csv"])
# Returns 0 on success, 1 on error
```

---

## Complete Examples

### Example 1: Offline Transcription with Progress Bar

```python
from src.engines.offline import OfflineTranscriptionEngine
from src.transcriber import TranscriptionManager

# Initialize engine
engine = OfflineTranscriptionEngine(language="en", device="cuda")

# Create manager
manager = TranscriptionManager(
    engine=engine,
    max_workers=4,
    batch_size=10
)

# Run batch transcription
manager.transcribe_batch(
    folder_path="./audio",
    output_csv="transcripts.csv",
    extensions=[".wav", ".mp3"]
)
```

### Example 2: OpenAI API with Error Handling

```python
import os
from src.engines.openai import OpenAITranscriptionEngine
from src.transcriber import TranscriptionManager
from openai import AuthenticationError, RateLimitError

# Initialize with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAITranscriptionEngine(api_key=api_key, language="fa")

# Create manager with reduced workers for API rate limits
manager = TranscriptionManager(
    engine=engine,
    max_workers=2,  # Lower to avoid rate limits
    batch_size=5
)

try:
    manager.transcribe_batch(
        folder_path="./persian_audio",
        output_csv="persian_transcripts.csv"
    )
except AuthenticationError:
    print("Error: Invalid OpenAI API key")
except RateLimitError:
    print("Error: OpenAI rate limit exceeded")
```

### Example 3: Persian Text Normalization

```python
from src.utils.normalizer import BasicPersianNormalizer

normalizer = BasicPersianNormalizer()

# Normalize Arabic script to Persian
text = "سلام دنيا، چطوري؟"  # Arabic yeh
normalized = normalizer.normalize(text)
print(normalized)
# Output: "سلام دنیا، چطوری؟"  # Persian yeh

# Normalize numerals
text_with_numbers = "سال ١٣٩٥"  # Arabic numerals
normalized_nums = normalizer.normalize(text_with_numbers)
print(normalized_nums)
# Output: "سال ۱۳۹۵"  # Persian numerals
```

### Example 4: Single File Transcription

```python
from src.engines.offline import OfflineTranscriptionEngine

engine = OfflineTranscriptionEngine()
transcript = engine.transcribe("my_audio.wav")

# Save to file
with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript)

print(f"Saved: {len(transcript)} characters")
```

---

## Error Handling

### Common Exceptions

#### Engine Errors

```python
from src.engines.offline import OfflineTranscriptionEngine

engine = OfflineTranscriptionEngine()

try:
    transcript = engine.transcribe("nonexistent.wav")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Unsupported format: {e}")
except RuntimeError as e:
    print(f"Torch/torchaudio not installed: {e}")
```

#### OpenAI Errors

```python
from src.engines.openai import OpenAITranscriptionEngine
from openai import (
    AuthenticationError,
    RateLimitError,
    APIError
)

engine = OpenAITranscriptionEngine(api_key="sk-...")

try:
    transcript = engine.transcribe("audio.mp3")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded (after 3 retries)")
except APIError as e:
    print(f"OpenAI API error: {e}")
```

#### Manager Errors

```python
from src.transcriber import TranscriptionManager

manager = TranscriptionManager(engine)

try:
    manager.transcribe_batch(
        folder_path="./nonexistent",
        output_csv="results.csv"
    )
except FileNotFoundError as e:
    print(f"Folder not found: {e}")
except OSError as e:
    print(f"CSV write error: {e}")
```

### Error Result Format

When `TranscriptionManager.transcribe_file()` catches an exception, it returns:

```python
{
    "file_path": "audio.mp3",
    "status": "failed",
    "transcript": None,
    "error_msg": "File not found: audio.mp3"
}
```

This allows batch operations to continue despite individual file failures.

---

## Type Hints

All public APIs include complete type hints:

```python
def transcribe(self, audio_path: str) -> str: ...
def transcribe_batch(self, audio_paths: Iterable[str]) -> List[str]: ...
def find_audio_files(self, folder_path: str) -> List[Path]: ...
```

Use with mypy for static type checking:
```bash
mypy src/
```

---

## See Also

- [User Guide](USER_GUIDE.md) - End-user documentation
- [Architecture](ARCHITECTURE.md) - System design
- [Setup Guide](SETUP.md) - Installation instructions
- [Contributing](CONTRIBUTING.md) - Development guide

