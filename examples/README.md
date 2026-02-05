# Persian Audio Transcriber - Usage Examples

This directory contains practical examples of using the Persian Audio Transcriber in various scenarios.

## Examples Overview

- **[basic_transcription.py](basic_transcription.py)** - Simple Python API usage for single file transcription
- **[batch_processing.py](batch_processing.py)** - Batch processing multiple audio files from a directory
- **[cli_usage.sh](cli_usage.sh)** - Common CLI commands and use cases

## Quick Start

### 1. Basic Python Usage

```python
from persian_transcriber import PersianAudioTranscriber

# Simple transcription with defaults
transcriber = PersianAudioTranscriber()
result = transcriber.transcribe_file("audio.mp3")
print(result["text"])
```

### 2. Using the CLI

```bash
# Transcribe a single file
persian-transcriber audio.mp3

# Use GPU acceleration with large model
persian-transcriber audio.mp3 -m large-v3 -d cuda

# Transcribe directory and output as SRT
persian-transcriber ./recordings/ -f srt --recursive
```

## Configuration Options

### Engine Types
- `whisper` - OpenAI Whisper (default)
- `faster_whisper` - CTranslate2 optimized Whisper
- `openai_api` - OpenAI API (requires API key)
- `google` - Google Speech-to-Text

### Model Sizes
- `tiny` - Fastest, lowest accuracy (~1GB RAM)
- `base` - Fast, good for testing (~1GB RAM)
- `small` - Balanced (~2GB RAM)
- `medium` - Better accuracy (~5GB RAM)
- `large` - Best accuracy (~10GB RAM)
- `large-v3` - Latest large model with improvements

### Output Formats
- `txt` - Plain text (default)
- `json` - Detailed JSON with timestamps
- `srt` - SubRip subtitle format
- `vtt` - WebVTT subtitle format

## Environment Setup

### Using Config File

Create `config.yaml` in your project directory:

```yaml
# Engine configuration
offline:
  device: cuda  # or cpu
  
# OpenAI API (optional)
openai:
  api_key: sk-proj-YOUR_API_KEY_HERE

# Transcription settings
transcription:
  max_workers: 4
  batch_size: 10
  extensions:
    - .wav
    - .mp3
    - .flac
    - .m4a

# Persian normalization
normalization:
  convert_numerals: true
  lowercase: false

# Logging
logging:
  level: INFO
```

### Environment Variables

```bash
# OpenAI API key
export OPENAI_API_KEY="sk-proj-YOUR_API_KEY"

# Custom config path
export PERSIAN_TRANSCRIBER_CONFIG="/path/to/config.yaml"

# CUDA path (if needed)
export CUDA_HOME="/usr/local/cuda"
```

## Performance Tips

1. **GPU Acceleration**: Use `-d cuda` with an NVIDIA GPU for 5-10x speedup
2. **Faster Whisper**: Use `-e faster_whisper` for 2-4x speedup with similar accuracy
3. **Batch Processing**: Process multiple files together to amortize startup costs
4. **Model Selection**: Start with `small` or `medium` for development, use `large-v3` for production

## Common Use Cases

### 1. Podcast Transcription
```bash
persian-transcriber podcast.mp3 -m large-v3 -d cuda -f txt
```

### 2. Video Subtitle Generation
```bash
persian-transcriber video.mp4 -f srt -o video.srt
```

### 3. Batch Conference Recording Processing
```bash
persian-transcriber ./recordings/ --recursive -m medium -d cuda
```

### 4. API-based Transcription (for long files)
```bash
persian-transcriber long_audio.mp3 -e openai_api --api-key sk-...
```

## Troubleshooting

- **CUDA errors**: Make sure CUDA toolkit is installed and `nvidia-smi` works
- **Memory errors**: Try a smaller model size or use CPU device
- **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Audio format errors**: FFmpeg must be installed for audio/video processing

For more details, see [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
