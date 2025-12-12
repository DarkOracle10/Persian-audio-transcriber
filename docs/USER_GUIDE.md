# User Guide

Comprehensive guide for using the Persian Audio Transcription Tool.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Output Formats](#output-formats)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

See [README.md](../README.md) for installation instructions.

### Quick Test

```bash
python main.py --help
```

## Basic Usage

### Single File Transcription

```bash
python main.py audio.mp3
```

### Specify Output Format

```bash
# Plain text
python main.py audio.mp3 --format txt

# JSON with metadata
python main.py audio.mp3 --format json

# SRT subtitles
python main.py audio.mp3 --format srt
```

### Batch Processing

```bash
python main.py ./audio_folder --output ./transcriptions
```

## Advanced Usage

### GPU Acceleration

```bash
python main.py audio.mp3 --engine faster_whisper --model medium
```

### Custom Model Size

```bash
# Fast but less accurate
python main.py audio.mp3 --model base

# Slower but more accurate
python main.py audio.mp3 --model large-v3
```

### Disable Normalization

```bash
python main.py audio.mp3 --no-normalize
```

## Output Formats

### TXT Format
Plain text transcription, easy to read and edit.

### JSON Format
Includes:
- Full transcription text
- Original text (before normalization)
- Segments with timestamps
- Language detection
- Duration

### SRT Format
Subtitle file compatible with video players:
```
1
00:00:00,000 --> 00:00:05,500
[Transcribed text segment]
```

## Performance Optimization

1. **Use GPU**: `--engine faster_whisper`
2. **Optimal Model**: `medium` for balance, `large` for accuracy
3. **Batch Processing**: Process multiple files efficiently
4. **Model Caching**: Models are cached after first download

## Troubleshooting

See [README.md](../README.md) Troubleshooting section for common issues.

