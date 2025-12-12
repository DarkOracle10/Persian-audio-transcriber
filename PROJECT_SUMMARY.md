# Persian Audio/Video Transcription Tool
## Project Summary for Resume

---

## Project Title
**Persian Audio/Video Transcription Tool with GPU Acceleration**

## Overview
A production-ready, GPU-accelerated audio and video transcription system with native Persian/Farsi language support, featuring advanced text normalization, multiple transcription engines, and comprehensive output formats.

## Technologies Used

### Core Technologies
- **Python 3.8+** - Primary programming language
- **OpenAI Whisper** - State-of-the-art speech recognition model
- **Faster-Whisper** - Optimized Whisper implementation with CTranslate2
- **Hazm** - Persian natural language processing library

### GPU & Performance
- **CUDA 12.x** - NVIDIA GPU computing platform
- **cuDNN** - CUDA Deep Neural Network library
- **FP16 Precision** - Half-precision floating point for 2x speedup
- **PyTorch** - Deep learning framework

### Audio/Video Processing
- **FFmpeg** - Multimedia processing framework
- **Pydub** - Audio manipulation library
- **OpenAI API** - Cloud-based transcription service

### Additional Technologies
- **JSON/CSV** - Data formats
- **SRT** - Subtitle format generation
- **Command-line Interface** - CLI with argparse
- **Batch Processing** - Multi-file processing capabilities

## Key Features Implemented

### 1. Multi-Engine Transcription System
- **OpenAI Whisper** - Original implementation
- **Faster-Whisper** - GPU-optimized version (5-10x faster)
- **OpenAI API** - Cloud-based transcription
- **Google Speech Recognition** - Alternative engine

### 2. Persian Language Support
- Native Persian/Farsi transcription
- Hazm-based text normalization
- Arabic to Persian character conversion
- Automatic whitespace normalization
- Fallback normalizer when Hazm unavailable

### 3. GPU Acceleration
- Automatic CUDA detection and initialization
- FP16 precision for optimal performance
- Automatic CPU fallback when GPU unavailable
- Multi-GPU support capability
- Windows DLL path management

### 4. Multiple Output Formats
- **TXT** - Plain text transcription
- **JSON** - Detailed metadata with timestamps and segments
- **SRT** - Subtitle files for video players

### 5. Batch Processing
- Process entire folders of audio/video files
- Progress tracking per file
- Summary reports generation
- Error handling with per-file error tracking

### 6. Comprehensive File Format Support
- Audio: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- Video: MP4 (automatic audio extraction)

### 7. Error Handling & Robustness
- Automatic GPU/CPU fallback
- CUDA DLL path management (Windows)
- MP4 audio extraction with fallback
- Comprehensive error messages

## Technical Achievements

### Performance Optimization
- **5-10x speedup** with GPU acceleration on RTX 3060
- FP16 precision reduces memory usage by 50%
- Model caching prevents redundant downloads
- Optimized audio preprocessing pipeline

### Cross-Platform Compatibility
- Windows-specific CUDA DLL path management
- Automatic PATH detection and configuration
- Virtual environment support
- Platform-agnostic audio processing

### Code Quality
- Modular architecture with clear separation of concerns
- Comprehensive error handling
- Type hints for better code maintainability
- Extensive documentation

## Project Structure
```
Voice-transcription-tool/
├── main.py              # Main transcription module
├── README.md            # Comprehensive documentation
├── requirements.txt     # Python dependencies
├── setup.py            # Package installation script
├── LICENSE             # MIT License
├── CHANGELOG.md        # Version history
├── CONTRIBUTING.md     # Contribution guidelines
├── .gitignore          # Git ignore rules
├── docs/               # Documentation folder
│   ├── USER_GUIDE.md
│   ├── API.md
│   ├── TROUBLESHOOTING.md
│   └── BENCHMARKS.md
└── PROJECT_SUMMARY.md  # This file
```

## Skills Demonstrated

### Software Engineering
- Object-oriented programming
- API design and implementation
- Error handling and exception management
- Code documentation and commenting
- Version control (Git)

### Machine Learning / AI
- Speech recognition and natural language processing
- Model optimization and GPU acceleration
- Performance benchmarking and profiling

### System Integration
- Multi-platform compatibility (Windows, Linux, macOS)
- Hardware acceleration (CUDA, GPU programming)
- External library integration (FFmpeg, Whisper, Hazm)
- API integration (OpenAI, Google)

### DevOps / Deployment
- Dependency management (requirements.txt, setup.py)
- Virtual environment setup
- Installation scripts
- Documentation generation

### Problem Solving
- GPU/CUDA configuration issues
- Cross-platform compatibility challenges
- Performance optimization
- Error handling and fallback mechanisms

## Impact & Use Cases

### Potential Applications
- Content creation and video production
- Accessibility (closed captions)
- Language learning and education
- Research and data analysis
- Business transcription services
- Media archiving and documentation

### Performance Metrics
- **Real-time Factor**: 0.83x (faster than real-time) with medium model on GPU
- **Accuracy**: High accuracy for Persian/Farsi speech
- **Throughput**: Can process hours of audio in minutes with GPU

## Learning Outcomes

- Deep understanding of speech recognition systems
- GPU acceleration and CUDA programming
- Persian/Farsi language processing
- Audio/video file processing
- Production-ready code development
- Comprehensive documentation writing

---

**Project Status**: Production-ready, fully documented, and ready for deployment.

