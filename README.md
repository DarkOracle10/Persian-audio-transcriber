# Persian Audio Transcriber

[![Tests Passing](https://github.com/DarkOracle10/persian-audio-transcriber/workflows/CI/badge.svg)](https://github.com/DarkOracle10/persian-audio-transcriber/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/DarkOracle10/persian-audio-transcriber)](https://github.com/DarkOracle10/persian-audio-transcriber/stargazers)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)

A fully-featured, GPU-accelerated audio transcription toolkit with Persian/Farsi language support, parallel batch processing, and multiple transcription engines.

## 🎯 Features

### Core Functionality
- **Multiple Transcription Engines**
  - Faster-Whisper (CTranslate2-optimized, recommended)
  - OpenAI Whisper (original implementation)
  - OpenAI Whisper API (cloud-based)
  - Google Speech Recognition (free alternative)
- **GPU Acceleration** - CUDA support for RTX GPUs with automatic CPU fallback
- **Parallel Batch Processing** - Multi-threaded transcription with progress bars
- **Persian/Farsi Support** - Native normalization with Hazm and Arabic→Persian mapping
- **Multiple Output Formats** - TXT, JSON (with timestamps), SRT subtitles
- **Rate Limiting & Retry Logic** - Robust API calls with exponential backoff

### Supported Audio/Video Formats
- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Video**: MP4 (with automatic audio extraction)
- **Conversion**: Automatic format conversion via FFmpeg

### Output Formats
- **TXT** - Plain text transcription
- **JSON** - Detailed metadata with word-level timestamps and confidence scores
- **SRT** - Subtitle format for video players with timing

### Advanced Features
- **Modular Architecture** - Abstract engine base for easy extension with custom engines
- **Configuration Management** - YAML-based settings with environment variable overrides
- **Command-Line Interface** - Comprehensive argparse-based CLI with multiple subcommands
- **Comprehensive Logging** - Colorized console logs with configurable levels and file output
- **Error Handling** - Graceful degradation with detailed error messages and automatic fallbacks
- **Progress Tracking** - Real-time progress bars for batch processing with ETA
- **Model Caching** - Automatic model download and caching for offline use
- **Language Detection** - Automatic language detection for multilingual audio
- **Segment Timestamps** - Word-level and sentence-level timestamp extraction
- **Confidence Scores** - Per-word and per-segment confidence metrics
- **Custom Normalizers** - Extensible text normalization system
- **Retry Logic** - Exponential backoff for API calls with configurable retries
- **Cross-Platform** - Works on Windows, Linux, and macOS with platform-specific optimizations

## 📋 Table of Contents

- [Features Showcase](#features-showcase)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features Showcase

### 🚀 High-Performance Transcription
- **5-10x faster** than real-time with GPU acceleration (RTX 3060)
- **FP16 precision** reduces memory usage by 50% without accuracy loss
- **Batch processing** with parallel execution for multiple files
- **Automatic model selection** based on available hardware

### 🌍 Persian Language Excellence
- **Native Persian support** with Hazm-based normalization
- **Arabic to Persian conversion** for correct character mapping
- **ZWNJ handling** for proper Persian text rendering
- **Number normalization** (Persian/Arabic numerals to consistent format)
- **Diacritic removal** and whitespace cleanup

### 🎯 Multiple Transcription Engines
- **Faster-Whisper**: Optimized CTranslate2 implementation (5-10x faster)
- **OpenAI Whisper**: Original implementation with full features
- **OpenAI API**: Cloud-based transcription with rate limiting
- **Google Speech**: Free alternative with basic Persian support

### 📊 Rich Output Formats
- **Plain Text (.txt)**: Simple transcription output
- **JSON (.json)**: Complete metadata with timestamps, confidence scores, language detection
- **SRT Subtitles (.srt)**: Time-synced subtitles for videos
- **VTT Subtitles (.vtt)**: WebVTT format for web players

### 🔧 Developer-Friendly
- **Python API**: Programmatic access with type hints
- **CLI Tool**: Comprehensive command-line interface
- **Extensible**: Easy to add custom engines and normalizers
- **Well-documented**: Complete API docs, examples, and guides
- **Type-safe**: Full type annotations with mypy support

## 🚀 Installation

### Prerequisites

- **Python**: 3.8+ (3.13 tested)
- **FFmpeg**: For audio/video processing
- **CUDA Toolkit**: 12.0+ (optional, for GPU acceleration)
- **Virtual Environment**: Recommended for dependency isolation

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/DarkOracle10/persian-audio-transcriber.git
cd persian-audio-transcriber
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

#### 3. Install Package

```bash
# Install from source
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"

# Or install specific extras
pip install -e ".[gpu,persian,dev]"
```

#### 4. Install FFmpeg

**Windows (PowerShell):**
```powershell
winget install Gyan.FFmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

#### 5. Configure Environment

Copy the example configuration:
```bash
cp examples/example_config.yaml config.yaml
```

Set environment variables (for OpenAI API):
```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-proj-..."

# Linux/Mac
export OPENAI_API_KEY="sk-proj-..."
```

## 🎬 Quick Start

### Basic Transcription

```bash
# Transcribe a single file
persian-transcriber audio.mp3

# Transcribe with GPU and large model
persian-transcriber audio.mp3 -m large-v3 -d cuda

# Transcribe with OpenAI API
persian-transcriber audio.mp3 -e openai_api --api-key sk-...

# Generate SRT subtitles
persian-transcriber video.mp4 -f srt

# Batch process directory recursively
persian-transcriber ./recordings/ -r --output-dir ./transcriptions
```

### Using Python API

```python
from persian_transcriber import PersianAudioTranscriber

# Initialize transcriber with Faster-Whisper (default)
transcriber = PersianAudioTranscriber(model_size="medium", device="cuda")

# Transcribe a single file
result = transcriber.transcribe_file("audio.mp3")
print(result["text"])

# Batch process a directory
results = transcriber.scan_and_transcribe(
    folder_path="./recordings",
    output_dir="./transcriptions",
    save_format="txt"
)
```

## 📖 Usage

### Command-Line Interface

```bash
persian-transcriber <input_path> [OPTIONS]
```

**Engine Options:**
- `-e, --engine`: Engine type (`faster_whisper`, `whisper`, `openai_api`, `google`)
- `-m, --model`: Model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `-d, --device`: Compute device (`auto`, `cuda`, `cpu`, `mps`)

**Output Options:**
- `-f, --format`: Output format (`txt`, `json`, `srt`, `vtt`)
- `-o, --output`: Output file path
- `--output-dir`: Output directory for batch processing

**Processing Options:**
- `-r, --recursive`: Recursively process subdirectories
- `-l, --language`: Language code (default: `fa` for Persian)
- `--no-normalize`: Disable text normalization
- `--task`: Task type (`transcribe` or `translate`)

**Performance Options:**
- `--batch-size`: Number of concurrent processing threads
- `--beam-size`: Beam search width (affects accuracy vs speed)
- `--compute-type`: Quantization type (`float16`, `float32`, `int8`)

**API Options:**
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--max-retries`: Maximum API retry attempts
- `--timeout`: Request timeout in seconds

**Batch Processing:**
- `-r, --recursive`: Search directories recursively
- `--skip-existing`: Skip files with existing transcriptions

**Examples:**

```bash
# High-quality transcription with GPU
voice-transcriber audio.mp3 -e faster_whisper -m large-v3 -d cuda

# Batch process with JSON output
voice-transcriber ./audio_folder -r -f json --output-dir ./results

# OpenAI API transcription
voice-transcriber audio.mp3 -e openai_api --api-key $OPENAI_API_KEY
```

### Configuration

The tool can be configured via command-line arguments. For Persian text normalization, use the `--no-normalize` flag to disable it:

```bash
# Disable Persian normalization
voice-transcriber audio.mp3 --no-normalize

# Enable verbose logging
voice-transcriber audio.mp3 -v

# Quiet mode (errors only)
voice-transcriber audio.mp3 -q
```

### Environment Variables

Set API keys via environment variables:

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```
## ⚙️ Configuration

### Configuration File

Create a `config.yaml` file for persistent settings:

```yaml
project:
  name: persian-audio-transcriber
  version: \"1.0.0\"

transcription:
  default_engine: faster_whisper
  default_model: medium
  default_device: auto
  default_language: fa
  
  # Engine-specific settings
  faster_whisper:
    compute_type: float16
    beam_size: 5
    best_of: 5
    
  openai_api:
    max_retries: 3
    timeout: 300
    
output:
  default_format: txt
  include_timestamps: true
  include_confidence: true
  
normalization:
  enabled: true
  arabic_to_persian: true
  remove_diacritics: true
  normalize_numbers: true
  
logging:
  level: INFO
  colorize: true
  file: logs/transcription.log
```

### Environment Variables

Override settings with environment variables:

```bash
# API Keys
export OPENAI_API_KEY=\"sk-...\"
export GOOGLE_API_KEY=\"...\"

# Default Settings
export TRANSCRIBER_ENGINE=\"faster_whisper\"
export TRANSCRIBER_MODEL=\"large-v3\"
export TRANSCRIBER_DEVICE=\"cuda\"
export TRANSCRIBER_LANGUAGE=\"fa\"

# Performance
export TRANSCRIBER_COMPUTE_TYPE=\"float16\"
export TRANSCRIBER_BATCH_SIZE=\"8\"

# Output
export TRANSCRIBER_OUTPUT_FORMAT=\"json\"
export TRANSCRIBER_NORMALIZE=\"true\"
```
## 🏗️ Architecture

The toolkit follows a modular architecture with clear separation of concerns:

```
src/
├── engines/              # Transcription engine implementations
│   ├── base.py          # Abstract TranscriptionEngine interface
│   ├── openai_api_engine.py  # OpenAI Whisper API with rate limiting
│   ├── offline.py       # Offline Wav2Vec2 engine
│   └── faster_whisper_engine.py  # Faster-Whisper integration
├── normalizers/         # Text normalization
│   ├── base.py         # Abstract normalizer interface
│   ├── basic.py        # Basic whitespace normalization
│   └── persian.py      # Persian-specific normalization
├── output/             # Output formatters
│   ├── base.py        # Abstract formatter interface
│   ├── txt_formatter.py
│   ├── json_formatter.py
│   └── srt_formatter.py
├── utils/             # Utility modules
│   ├── audio.py       # Audio processing helpers
│   ├── cuda_setup.py  # CUDA configuration
│   ├── exceptions.py  # Custom exceptions
│   └── logging.py     # Logging setup
├── transcriber.py     # TranscriptionManager (batch orchestration)
├── cli.py            # Command-line interface
└── __main__.py       # Module entry point
```

See `docs/ARCHITECTURE.md` for detailed architecture documentation.

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_offline_engine.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── conftest.py                # Shared fixtures
├── test_engines.py           # Engine unit tests
├── test_normalizer.py        # Normalizer tests
├── test_batch_processing.py  # Integration tests
└── fixtures/                 # Test data
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Code Quality Tools

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with Mypy
mypy src/

# Run all checks
black src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest
```

### Adding a New Engine

See `docs/CONTRIBUTING.md` for step-by-step guide on adding new transcription engines.

## 🤝 Contributing

Contributions are welcome! Please see `docs/CONTRIBUTING.md` for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run quality checks: `black`, `ruff`, `mypy`, `pytest`
5. Commit with clear messages: `git commit -m "feat: add new engine"`
6. Push and create Pull Request

### Code Standards

- **Formatting**: Black (line length 100)
- **Linting**: Ruff with strict rules
- **Type Hints**: Full type annotations
- **Testing**: >80% coverage required
- **Documentation**: Docstrings for all public APIs

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art speech recognition
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP toolkit
- [facebook/wav2vec2](https://huggingface.co/facebook/wav2vec2-large-xlsr-53-persian) - Offline Persian ASR

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DarkOracle10/persian-audio-transcriber/issues)
- **Documentation**: See `docs/` directory
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

**Author**: DarkOracle10  
**Email**: darkoracle3860@gmail.com  
**Version**: 1.0.0

