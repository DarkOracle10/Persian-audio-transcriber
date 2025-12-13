# Setup Guide

Complete installation and configuration guide for the Voice Transcription toolkit.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Platform-Specific Setup](#platform-specific-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python**: 3.8 or higher (3.13 supported)
- **pip**: Latest version recommended
- **FFmpeg**: Required for audio/video processing

### Optional (for GPU acceleration)

- **NVIDIA GPU**: CUDA-capable (Compute Capability 3.5+)
- **CUDA Toolkit**: 11.x or 12.x
- **cuDNN**: Compatible with your CUDA version

### Optional (for Persian text processing)

- **Hazm**: Persian NLP library (may have Python 3.13 compatibility issues)

---

## Installation

### Step 1: Clone or Download

```bash
git clone https://github.com/YourUsername/Voice-Transcriber.git
cd Voice-Transcriber
```

### Step 2: Create Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv Transcribe
.\Transcribe\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip and Install Build Tools

```bash
python -m pip install --upgrade pip
python -m pip install setuptools wheel
```

### Step 4: Install Dependencies

#### Option A: Full Installation (Recommended)

```bash
pip install -r requirements.txt
```

#### Option B: Minimal Installation (CPU-only, no GPU)

```bash
pip install -r requirements-minimal.txt  # If available
# Or manually install core dependencies:
pip install faster-whisper openai pyyaml click tqdm
```

#### Option C: Development Installation

```bash
pip install -r requirements-dev.txt
```

This includes testing tools (pytest, black, mypy).

### Step 5: Install FFmpeg

#### Windows

**Option A: Chocolatey**
```powershell
choco install ffmpeg
```

**Option B: Manual**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH:
   ```powershell
   setx PATH "$env:PATH;C:\ffmpeg\bin"
   ```
4. Restart terminal

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Linux (Fedora/RHEL)

```bash
sudo dnf install ffmpeg
```

#### macOS

```bash
brew install ffmpeg
```

### Step 6: Verify FFmpeg

```bash
ffmpeg -version
```

Should display FFmpeg version info.

---

## Platform-Specific Setup

### Windows Setup

#### 1. CUDA Installation (for GPU acceleration)

1. **Download CUDA Toolkit**:
   - Visit https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 or 12 → exe (local)
   - Download and run installer

2. **Verify Installation**:
   ```powershell
   nvcc --version
   ```

3. **Add CUDA to PATH** (if not automatic):
   ```powershell
   setx CUDA_HOME "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
   setx PATH "$env:PATH;$env:CUDA_HOME\bin"
   ```

4. **Create `config.yaml`** (optional, for custom paths):
   ```yaml
   cuda:
     cuda_home_path: "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4"
     library_paths:
       windows:
         - "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
   ```

#### 2. Install PyTorch with CUDA Support

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

(Replace `cu121` with `cu118` for CUDA 11.8, or `cpu` for CPU-only)

#### 3. Set OpenAI API Key

```powershell
# Temporary (current session)
$env:OPENAI_API_KEY="sk-proj-..."

# Permanent (requires restart)
setx OPENAI_API_KEY "sk-proj-..."
```

---

### Linux Setup

#### 1. CUDA Installation (for GPU acceleration)

**Ubuntu/Debian:**

1. **Add NVIDIA Package Repository**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   ```

2. **Install CUDA Toolkit**:
   ```bash
   sudo apt install cuda-toolkit-12-4
   ```

3. **Add to PATH** (add to `~/.bashrc` or `~/.zshrc`):
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **Apply Changes**:
   ```bash
   source ~/.bashrc
   ```

5. **Verify**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

**Fedora/RHEL:**

```bash
sudo dnf install cuda-toolkit
```

#### 2. Install PyTorch with CUDA Support

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Set OpenAI API Key

Add to `~/.bashrc` or `~/.zshrc`:
```bash
export OPENAI_API_KEY="sk-proj-..."
```

Apply:
```bash
source ~/.bashrc
```

---

### macOS Setup

#### 1. Install Dependencies

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and FFmpeg
brew install python@3.13 ffmpeg
```

#### 2. Install PyTorch (Apple Silicon MPS acceleration)

```bash
pip install torch torchaudio
```

PyTorch automatically uses Metal Performance Shaders (MPS) on M1/M2/M3 Macs.

#### 3. Set OpenAI API Key

Add to `~/.zshrc` or `~/.bash_profile`:
```bash
export OPENAI_API_KEY="sk-proj-..."
```

Apply:
```bash
source ~/.zshrc
```

---

## Configuration

### Create `config.yaml`

Create a file named `config.yaml` in the project root:

```yaml
# OpenAI API Configuration
openai:
  api_key: sk-proj-YOUR_API_KEY_HERE  # Optional: can also use env var

# Offline Engine Configuration
offline:
  device: cuda  # Options: "cuda", "cpu", or null for auto-detect

# CUDA Configuration (optional, for custom paths)
cuda:
  cuda_home_path: /usr/local/cuda  # Or C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
  library_paths:
    windows:
      - "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
    linux:
      - /usr/local/cuda/lib64
    darwin:
      - /usr/local/cuda/lib
```

**Security Note:** Never commit `config.yaml` with real API keys to version control. Add it to `.gitignore`.

### Alternative: Environment Variables Only

```bash
# Set these instead of using config.yaml
export OPENAI_API_KEY="sk-proj-..."
export CUDA_HOME="/usr/local/cuda"
```

---

## Verification

### 1. Check CUDA Availability

```bash
python -m src check-cuda
```

Expected output:
```
CUDA available: True
```

If False, check:
- NVIDIA drivers installed: `nvidia-smi`
- PyTorch CUDA version matches system CUDA: `python -c "import torch; print(torch.version.cuda)"`

### 2. Test Normalization

```bash
python -m src normalize --text "سلام دنيا"
```

Expected output:
```
سلام دنیا
```

### 3. Test Transcription (with sample audio)

```bash
# Create a test folder with audio files
mkdir test_audio
# ... add .mp3 or .wav files ...

# Run transcription
python -m src transcribe --folder test_audio --output results.csv --engine openai
```

Check `results.csv` for transcripts.

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch:
```bash
# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `CUDA available: False` (but GPU is present)

**Possible Causes:**
1. **PyTorch CPU version installed**
   - Check: `python -c "import torch; print(torch.__version__)"`
   - Should show `+cu121` or `+cu118`, not `+cpu`
   - Reinstall with CUDA index URL (see above)

2. **CUDA Toolkit not installed or wrong version**
   - Check: `nvcc --version`
   - PyTorch requires CUDA 11.x or 12.x
   - Reinstall CUDA Toolkit if needed

3. **NVIDIA drivers outdated**
   - Check: `nvidia-smi`
   - Update drivers from https://www.nvidia.com/Download/index.aspx

4. **Environment variable issue** (Linux/macOS)
   - Ensure `LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) includes CUDA libraries
   - Add to shell profile (see Linux/macOS setup sections)

### Issue: `ValueError: OpenAI API key required`

**Solution:** Set the API key:
```bash
# Windows
setx OPENAI_API_KEY "sk-proj-..."

# Linux/macOS
export OPENAI_API_KEY="sk-proj-..."
# Add to ~/.bashrc or ~/.zshrc for persistence
```

Or create `config.yaml` with the key.

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Solution:** Install FFmpeg (see Installation Step 5).

Verify with:
```bash
ffmpeg -version
```

If installed but not found, add to PATH:
- **Windows**: `setx PATH "$env:PATH;C:\ffmpeg\bin"`
- **Linux/macOS**: `export PATH=/usr/local/bin:$PATH` (add to shell profile)

### Issue: `OSError: cannot load library 'cudnn_ops'` (Windows)

**Cause:** CuDNN libraries not in PATH.

**Solution:**
1. Download cuDNN from https://developer.nvidia.com/cudnn
2. Extract to `C:\Program Files\NVIDIA\CUDNN\v8.x`
3. Add to PATH:
   ```powershell
   setx PATH "$env:PATH;C:\Program Files\NVIDIA\CUDNN\v8.x\bin"
   ```
4. Restart terminal and Python

### Issue: `ImportError: cannot import name 'Normalizer' from 'hazm'`

**Cause:** Hazm has dependency issues on Python 3.13 (fasttext incompatibility).

**Solution:** The toolkit automatically falls back to `BasicPersianNormalizer`. No action needed.

If you need Hazm:
- Use Python 3.11 or 3.12
- Install with: `pip install hazm`

### Issue: Slow transcription with OpenAI engine

**Cause:** Rate limiting (3 requests/second).

**Solution:** Reduce `--max-workers`:
```bash
python -m src transcribe --folder ./audio --output results.csv --engine openai --max-workers 2
```

### Issue: `429 Client Error: Too Many Requests` (OpenAI)

**Cause:** OpenAI API rate limit exceeded.

**Solution:**
- Wait 60 seconds and retry
- Reduce `--max-workers` to 1 or 2
- Check OpenAI account usage limits

### Issue: Out of memory (GPU)

**Cause:** Model too large for GPU VRAM.

**Solution:**
1. Use smaller model (in `main.py`):
   ```bash
   python main.py audio.mp3 --model medium  # Instead of large-v3
   ```

2. Or use CPU:
   ```yaml
   # config.yaml
   offline:
     device: cpu
   ```

---

## Next Steps

- Read [USER_GUIDE.md](USER_GUIDE.md) for usage examples
- Review [API.md](API.md) for programmatic usage
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to add features
- Explore [ARCHITECTURE.md](ARCHITECTURE.md) for internals

---

## Support

For issues not covered here:
1. Check existing [GitHub Issues](https://github.com/YourUsername/Voice-Transcriber/issues)
2. Run with `--log-level DEBUG` for detailed logs
3. Open a new issue with:
   - Platform (Windows/Linux/macOS)
   - Python version (`python --version`)
   - Full error traceback
   - Steps to reproduce
