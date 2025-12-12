# Persian Audio/Video Transcription Tool

A powerful, GPU-accelerated audio and video transcription tool with native Persian/Farsi language support, featuring advanced text normalization and multiple transcription engines.

## 🎯 Features

### Core Functionality
- **Persian/Farsi Language Support** - Native support with Hazm text normalization
- **GPU Acceleration** - CUDA/FP16 support for RTX GPUs (5-10x faster)
- **Multiple Engines** - Whisper, Faster-Whisper, OpenAI API, Google Speech Recognition
- **Batch Processing** - Process entire folders of audio/video files
- **Multiple Formats** - TXT, JSON, SRT subtitle generation

### Supported Audio/Video Formats
- MP3, MP4, WAV, M4A, FLAC, OGG, AAC, WMA
- Automatic audio extraction from video files (MP4)

### Advanced Features
- **Text Normalization** - Arabic to Persian character conversion, whitespace normalization
- **Subtitle Generation** - SRT format with precise timestamps
- **Detailed Metadata** - JSON output with segments, timestamps, language detection
- **Error Handling** - Automatic CPU fallback if GPU unavailable
- **Windows Optimized** - Automatic CUDA DLL path detection

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Requirements](#requirements)
- [Features Detail](#features-detail)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- **Python**: 3.8+ (3.11 recommended for full Hazm support)
- **CUDA Toolkit**: 12.0+ (for GPU acceleration, optional)
- **cuDNN**: For GPU acceleration (optional, auto-installed via pip)
- **FFmpeg**: For audio/video processing

### Step-by-Step Setup

#### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd Voice-transcription-tool
```

#### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install FFmpeg

**Windows:**
```bash
# Using winget (Windows 10/11)
winget install Gyan.FFmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

#### 5. Install CUDA Libraries (For GPU Acceleration)

**Option A: Automatic (Recommended)**
```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

**Option B: Manual CUDA Installation**
- Download CUDA 12.x from: https://developer.nvidia.com/cuda-downloads
- Download cuDNN from: https://developer.nvidia.com/cudnn
- Follow installation instructions

## 🎬 Quick Start

### Basic Transcription

```bash
# Transcribe a single audio file
python main.py audio.mp3

# Transcribe a video file (MP4)
python main.py video.mp4

# Use GPU acceleration (recommended)
python main.py audio.mp3 --engine faster_whisper --model medium
```

### Output

- **Console**: Displays transcribed text in real-time
- **File**: Automatically saves to `audio.txt` (same location as input)

## 📖 Usage

### Command-Line Interface

```bash
python main.py <input_file_or_folder> [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--engine` | Transcription engine: `whisper`, `faster_whisper`, `google`, `openai_api` | `whisper` |
| `--model` | Model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v3` | `medium` |
| `--format` | Output format: `txt`, `json`, `srt` | `txt` |
| `--output` | Output directory | Same as input |
| `--language` | Language code (default: `fa` for Persian) | `fa` |
| `--no-normalize` | Disable Persian text normalization | `False` |
| `--api-key` | OpenAI API key (for `openai_api` engine) | - |

### Examples

#### Single File Transcription

```bash
# Basic transcription
python main.py recording.mp3

# High-quality with GPU
python main.py recording.mp3 --engine faster_whisper --model large

# Generate subtitles
python main.py video.mp4 --format srt

# JSON output with timestamps
python main.py audio.mp3 --format json
```

#### Batch Processing

```bash
# Process all audio files in a folder
python main.py ./audio_folder --output ./transcriptions

# Batch with JSON output
python main.py ./audio_folder --format json --output ./results
```

#### GPU-Accelerated Transcription

```bash
# Use Faster-Whisper with GPU (recommended)
python main.py audio.mp3 --engine faster_whisper --model medium

# Large model for best accuracy
python main.py audio.mp3 --engine faster_whisper --model large-v3
```

#### Custom Output Location

```bash
python main.py audio.mp3 --output ./transcription_results --format json
```

### Python API Usage

```python
from main import PersianAudioTranscriber

# Initialize transcriber
transcriber = PersianAudioTranscriber(
    engine="faster_whisper",
    model_size="medium",
    language="fa",
    normalize_persian=True
)

# Transcribe a single file
result = transcriber.transcribe_file("audio.mp3")
print(result['text'])

# Batch process folder
results = transcriber.scan_and_transcribe(
    folder_path="./audio_folder",
    output_dir="./output",
    save_format="txt"
)
```

## 🔧 Requirements

### System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: 2GB+ for models (one-time download)
- **GPU**: NVIDIA GPU with CUDA support (optional, but highly recommended)

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Dual-core 2GHz | Quad-core 3GHz+ |
| RAM | 4GB | 16GB+ |
| GPU | - | NVIDIA RTX 3060+ |
| Storage | 2GB free | 10GB+ free |

### Python Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

- `openai-whisper` - OpenAI Whisper transcription
- `faster-whisper` - Optimized Whisper implementation
- `hazm` - Persian text normalization
- `pydub` - Audio processing
- `torch` - Deep learning framework
- `nvidia-cudnn-cu12` - CUDA Deep Neural Network library

## ✨ Features Detail

### Persian Language Support

- **Hazm Integration**: Automatic Persian text normalization
- **Character Conversion**: Arabic to Persian character mapping
- **Text Cleanup**: Whitespace normalization, formatting
- **Fallback Normalizer**: Basic normalization when Hazm unavailable

### GPU Acceleration

- **Automatic Detection**: Detects CUDA and GPU availability
- **FP16 Precision**: 2x faster with minimal accuracy loss
- **CPU Fallback**: Automatically falls back to CPU if GPU unavailable
- **Multi-GPU Support**: Utilizes available GPUs

### Multiple Output Formats

1. **TXT**: Plain text transcription
2. **JSON**: Detailed metadata with segments and timestamps
3. **SRT**: Subtitle files for video players

### Batch Processing

- Process entire folders
- Progress tracking
- Summary reports
- Error handling per file

## 🐛 Troubleshooting

### Common Issues

#### CUDA DLL Not Found

**Problem**: `Library cublas64_12.dll is not found`

**Solution**:
```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```
The code automatically adds DLL paths, but you may need to restart your terminal.

#### FFmpeg Not Found

**Problem**: `ffmpeg is not recognized`

**Solution**:
- Install FFmpeg (see Installation section)
- Add FFmpeg to system PATH
- Restart terminal after installation

#### Model Download Slow

**Problem**: First run downloads large model files

**Solution**: This is normal. Models are cached after first download:
- `tiny`: ~75 MB
- `base`: ~140 MB
- `small`: ~460 MB
- `medium`: ~1.4 GB (recommended for Persian)
- `large-v3`: ~3+ GB

#### GPU Not Used

**Problem**: Transcription uses CPU despite GPU available

**Solutions**:
1. Verify CUDA installation: `nvidia-smi`
2. Install CUDA libraries: `pip install nvidia-cudnn-cu12`
3. Use faster_whisper engine: `--engine faster_whisper`
4. Check GPU driver is up to date

### Performance Tips

- Use `faster_whisper` engine for best performance
- Use `medium` or `large` models for Persian (better accuracy)
- Enable GPU acceleration for 5-10x speedup
- Use `base` or `small` models for faster transcription (less accurate)

## 📊 Performance Benchmarks

Typical transcription speeds on RTX 3060:

| Model | CPU | GPU (FP16) |
|-------|-----|------------|
| tiny | ~2x real-time | ~10x real-time |
| base | ~1x real-time | ~8x real-time |
| small | ~0.5x real-time | ~6x real-time |
| medium | ~0.3x real-time | ~4x real-time |

*Real-time = 1x means 1 minute audio takes 1 minute to transcribe*

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Created for academic and professional portfolio demonstration.

## 🙏 Acknowledgments

- OpenAI for Whisper model
- Faster-Whisper team for optimized implementation
- Hazm team for Persian NLP tools
- CTranslate2 for efficient inference

## 📚 Additional Resources

- [User Guide](docs/USER_GUIDE.md)
- [API Documentation](docs/API.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Performance Benchmarks](docs/BENCHMARKS.md)

---

**Note**: For best results with Persian/Farsi audio, use `medium` or larger models with GPU acceleration.

