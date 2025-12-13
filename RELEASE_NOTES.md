# Voice Transcriber v1.0.0 (2025-12-13)

A fully-featured, GPU-accelerated voice transcription toolkit with native Persian/Farsi support, modular engines, and a polished CLI — ready for your resume.

## Highlights
- Multiple engines: Faster-Whisper (GPU), Whisper, OpenAI API, Google
- Persian normalization (Arabic→Persian chars, whitespace, numerals)
- Parallel batch processing with progress & robust error handling
- CUDA auto-detection with CPU fallback
- Clean packaging with console scripts `voice-transcriber`, `vtranscribe`

## Changes
- Packaging: setup.py and pyproject.toml with accurate entry points and deps
- Version & metadata: aligned to 1.0.0 and author info
- CLI: accurate help, examples, and sane defaults
- Docs: rewritten README, architecture, contributing, setup guides
- Tests: 18/18 passing, mock OpenAI client, faster-whisper stubs
- Cleanup: removed orphaned prototyping files in `src/`

## Install
```bash
pip install -e .
# or with extras
pip install -e ".[gpu,persian]"
```

## Quick Start
```bash
voice-transcriber audio.mp3 -e faster_whisper -m large-v3 -d cuda
voice-transcriber ./recordings -r --output-dir ./transcriptions
```

## Notes
- For OpenAI API engine, set `OPENAI_API_KEY`
- Windows + CUDA may need NVIDIA DLLs in PATH (see `docs/SETUP.md`)

---
All tests passing. Tag `v1.0.0` pushed. Ready to ship.
