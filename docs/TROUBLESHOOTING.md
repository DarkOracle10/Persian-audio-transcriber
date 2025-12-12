# Troubleshooting Guide

Common issues and solutions.

## CUDA/CUDA DLL Issues

### Issue: "cublas64_12.dll not found"

**Solution:**
```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

The code automatically detects DLL paths. Restart terminal if needed.

### Issue: GPU not being used

**Check:**
1. Verify CUDA: `nvidia-smi`
2. Check GPU initialization messages
3. Use `--engine faster_whisper`

## FFmpeg Issues

### Issue: "ffmpeg not recognized"

**Solution:**
1. Install FFmpeg (see README)
2. Add to PATH
3. Restart terminal

## Model Download Issues

### Issue: Slow model download

**Normal:** First run downloads models (1-3 GB depending on model size).

**Solution:** Models are cached after first download.

## Performance Issues

### Issue: Slow transcription

**Solutions:**
1. Use GPU: `--engine faster_whisper`
2. Use smaller model: `--model base`
3. Check CPU/GPU usage

## Persian Text Issues

### Issue: Incorrect Persian characters

**Solution:**
- Text normalization should fix this automatically
- Check if `--no-normalize` was used

