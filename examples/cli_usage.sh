#!/bin/bash
# Persian Audio Transcriber - CLI Usage Examples
#
# Common command-line usage patterns for the Persian Audio Transcriber tool.
# Make this file executable: chmod +x cli_usage.sh

echo "=========================================="
echo "Persian Audio Transcriber - CLI Examples"
echo "=========================================="

# Note: Replace 'audio.mp3', 'video.mp4', etc. with your actual files

# =============================================================================
# BASIC USAGE
# =============================================================================

echo -e "\n## BASIC USAGE ##\n"

# Transcribe a single audio file (uses defaults: whisper, base model, auto-detect device)
# persian-transcriber audio.mp3

# Transcribe and show verbose output
# persian-transcriber audio.mp3 --verbose

# Transcribe and save to specific output file
# persian-transcriber audio.mp3 -o transcript.txt

# =============================================================================
# MODEL AND DEVICE SELECTION
# =============================================================================

echo "## MODEL AND DEVICE SELECTION ##"
echo ""

# Use large model for best accuracy (requires ~10GB VRAM/RAM)
# persian-transcriber audio.mp3 -m large-v3

# Use small model for faster processing
# persian-transcriber audio.mp3 -m small

# Force CPU processing (no GPU)
# persian-transcriber audio.mp3 -d cpu

# Force GPU/CUDA processing
# persian-transcriber audio.mp3 -d cuda

# Use medium model with GPU
# persian-transcriber audio.mp3 -m medium -d cuda

# =============================================================================
# ENGINE SELECTION
# =============================================================================

echo "## ENGINE SELECTION ##"
echo ""

# Use Whisper (default)
# persian-transcriber audio.mp3 -e whisper

# Use Faster-Whisper (CTranslate2 optimized, 2-4x faster)
# persian-transcriber audio.mp3 -e faster_whisper

# Use OpenAI API (requires API key)
# persian-transcriber audio.mp3 -e openai_api --api-key sk-proj-YOUR_KEY

# Use Google Speech-to-Text (requires credentials)
# persian-transcriber audio.mp3 -e google

# =============================================================================
# OUTPUT FORMATS
# =============================================================================

echo "## OUTPUT FORMATS ##"
echo ""

# Output as plain text (default)
# persian-transcriber audio.mp3 -f txt

# Output as JSON with timestamps
# persian-transcriber audio.mp3 -f json -o output.json

# Output as SRT subtitles (for videos)
# persian-transcriber video.mp4 -f srt -o subtitles.srt

# Output as WebVTT subtitles
# persian-transcriber video.mp4 -f vtt -o subtitles.vtt

# =============================================================================
# DIRECTORY/BATCH PROCESSING
# =============================================================================

echo "## DIRECTORY/BATCH PROCESSING ##"
echo ""

# Transcribe all audio files in a directory
# persian-transcriber ./recordings/

# Recursively transcribe all audio files in subdirectories
# persian-transcriber ./recordings/ --recursive

# Process directory and save individual text files
# persian-transcriber ./recordings/ -f txt

# Process directory and save as CSV with all results
# persian-transcriber ./recordings/ --output-csv results.csv

# =============================================================================
# ADVANCED OPTIONS
# =============================================================================

echo "## ADVANCED OPTIONS ##"
echo ""

# Specify language explicitly (fa = Farsi/Persian)
# persian-transcriber audio.mp3 -l fa

# Specify multiple languages to detect
# persian-transcriber audio.mp3 -l fa,en

# Use initial prompt to guide transcription
# persian-transcriber audio.mp3 --initial-prompt "این یک پادکست فارسی است"

# Adjust temperature for more/less creative output (0.0-1.0)
# persian-transcriber audio.mp3 --temperature 0.2

# Disable Persian text normalization
# persian-transcriber audio.mp3 --no-normalize

# Use faster compute type (float16 for GPU, int8 for CPU)
# persian-transcriber audio.mp3 -d cuda --compute-type float16

# =============================================================================
# VIDEO TRANSCRIPTION
# =============================================================================

echo "## VIDEO TRANSCRIPTION ##"
echo ""

# Transcribe video file (extracts audio automatically)
# persian-transcriber video.mp4

# Transcribe video and generate SRT subtitles
# persian-transcriber video.mp4 -f srt -o video.srt

# Transcribe multiple video files
# persian-transcriber ./videos/ -f srt

# =============================================================================
# CONFIGURATION FILE
# =============================================================================

echo "## CONFIGURATION FILE ##"
echo ""

# Use custom config file
# persian-transcriber audio.mp3 --config /path/to/config.yaml

# Config file location priority:
# 1. --config argument
# 2. PERSIAN_TRANSCRIBER_CONFIG environment variable
# 3. ./config.yaml (current directory)
# 4. ~/.persian_transcriber/config.yaml (home directory)

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

echo "## ENVIRONMENT VARIABLES ##"
echo ""

# Set OpenAI API key
# export OPENAI_API_KEY="sk-proj-YOUR_API_KEY_HERE"
# persian-transcriber audio.mp3 -e openai_api

# Set custom config path
# export PERSIAN_TRANSCRIBER_CONFIG="/path/to/config.yaml"
# persian-transcriber audio.mp3

# Set CUDA path (if needed)
# export CUDA_HOME="/usr/local/cuda"
# persian-transcriber audio.mp3 -d cuda

# =============================================================================
# REAL-WORLD EXAMPLES
# =============================================================================

echo "## REAL-WORLD EXAMPLES ##"
echo ""

# Example 1: High-quality podcast transcription
echo "# High-quality podcast transcription"
# persian-transcriber podcast.mp3 -m large-v3 -d cuda -f txt -o podcast_transcript.txt --verbose

# Example 2: Fast batch processing of interviews
echo "# Fast batch processing"
# persian-transcriber ./interviews/ -m small -d cuda -f txt --recursive

# Example 3: Generate subtitles for YouTube video
echo "# Generate subtitles for video"
# persian-transcriber youtube_video.mp4 -m medium -f srt -o subtitles.srt

# Example 4: Transcribe with OpenAI API for long files
echo "# OpenAI API transcription"
# export OPENAI_API_KEY="sk-proj-YOUR_KEY"
# persian-transcriber long_lecture.mp3 -e openai_api -f json -o lecture.json

# Example 5: Quick test with tiny model
echo "# Quick test transcription"
# persian-transcriber test_audio.mp3 -m tiny -d cpu

# Example 6: Production-grade batch processing with error handling
echo "# Production batch processing"
# for file in recordings/*.mp3; do
#     echo "Processing: $file"
#     persian-transcriber "$file" -m large-v3 -d cuda -f txt -o "transcripts/$(basename "$file" .mp3).txt" || echo "Failed: $file"
# done

# =============================================================================
# PERFORMANCE TIPS
# =============================================================================

echo -e "\n## PERFORMANCE TIPS ##"
echo ""
echo "1. GPU Acceleration: Use -d cuda with NVIDIA GPU for 5-10x speedup"
echo "2. Faster Whisper: Use -e faster_whisper for 2-4x speedup"
echo "3. Model Size: Start with 'small' or 'medium', use 'large-v3' for production"
echo "4. Compute Type: Use --compute-type float16 (GPU) or int8 (CPU) for speed"
echo "5. Batch Processing: Process multiple files together to amortize startup costs"
echo ""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

echo "## TROUBLESHOOTING ##"
echo ""
echo "# Check if GPU is available"
# nvidia-smi

echo "# Test with tiny model first"
# persian-transcriber audio.mp3 -m tiny -d cpu --verbose

echo "# Check FFmpeg installation"
# ffmpeg -version

echo "# Verify installation"
# persian-transcriber --version
# pip show persian-audio-transcriber

echo -e "\n=========================================="
echo "For more information:"
echo "  Documentation: https://github.com/DarkOracle10/Persian-audio-transcriber"
echo "  Issues: https://github.com/DarkOracle10/Persian-audio-transcriber/issues"
echo "=========================================="
