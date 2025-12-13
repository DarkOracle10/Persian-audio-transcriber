# Voice Transcriber

A fully-featured, GPU-accelerated voice transcription toolkit with Persian/Farsi language support, parallel batch processing, and multiple transcription engines.

## 🎯 Features

### Core Functionality
- **Multiple Transcription Engines** - OpenAI Whisper API, Offline Wav2Vec2, Faster-Whisper
- **GPU Acceleration** - CUDA support for RTX GPUs with automatic CPU fallback
- **Parallel Batch Processing** - Multi-threaded transcription with progress bars
- **Persian/Farsi Support** - Native normalization with Arabic→Persian character mapping
- **CSV Output** - Structured batch results with success/failure tracking
- **Rate Limiting & Retry Logic** - Robust API calls with exponential backoff

### Supported Audio/Video Formats
- MP3, MP4, WAV, M4A, FLAC, OGG, AAC, WMA
- Automatic audio extraction from video files

### Advanced Features
- **Modular Architecture** - Abstract engine base for easy extension
- **Configuration Management** - YAML-based settings with environment variables
- **Command-Line Interface** - argparse-based CLI with multiple subcommands
- **Comprehensive Logging** - Colorized logs with configurable levels
- **Error Handling** - Graceful degradation with detailed error messages

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- **Python**: 3.8+ (3.13 tested)
- **FFmpeg**: For audio/video processing
- **CUDA Toolkit**: 12.0+ (optional, for GPU acceleration)
- **Virtual Environment**: Recommended for dependency isolation

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/DarkOracle10/Voice-Transcriber.git
cd Voice-Transcriber
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
voice-transcriber audio.mp3

# Transcribe with GPU and large model
voice-transcriber audio.mp3 -m large-v3 -d cuda

# Transcribe with OpenAI API
voice-transcriber audio.mp3 -e openai_api --api-key sk-...

# Generate SRT subtitles
voice-transcriber video.mp4 -f srt

# Batch process directory recursively
voice-transcriber ./recordings/ -r --output-dir ./transcriptions
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
voice-transcriber <input_path> [OPTIONS]
```

**Engine Options:**
- `-e, --engine`: Engine type (`faster_whisper`, `whisper`, `openai_api`, `google`)
- `-m, --model`: Model size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `-d, --device`: Compute device (`auto`, `cuda`, `cpu`, `mps`)

**Output Options:**
- `-f, --format`: Output format (`txt`, `json`, `srt`, `vtt`)
- `-o, --output`: Output file path
- `--output-dir`: Output directory for batch processing

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

- **Issues**: [GitHub Issues](https://github.com/DarkOracle10/Voice-Transcriber/issues)
- **Documentation**: See `docs/` directory
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

**Author**: DarkOracle10  
**Email**: darkoracle3860@gmail.com  
**Version**: 1.0.0

