# Contributing Guide

Thank you for considering contributing to the Voice Transcription toolkit! This document provides guidelines for contributing code, reporting issues, and proposing new features.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style](#code-style)
5. [Adding a New Transcription Engine](#adding-a-new-transcription-engine)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [Issue Guidelines](#issue-guidelines)

---

## Code of Conduct

### Our Standards

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be inclusive**: Welcome contributors of all skill levels
- **Be patient**: Everyone was new once

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or deliberately derailing discussions
- Publishing others' private information

---

## Getting Started

### Reporting Bugs

1. Check [existing issues](https://github.com/YourUsername/Voice-Transcriber/issues) to avoid duplicates
2. Use the bug report template
3. Include:
   - Platform (Windows/Linux/macOS)
   - Python version
   - Full error traceback
   - Steps to reproduce
   - Expected vs. actual behavior

### Requesting Features

1. Check [existing issues](https://github.com/YourUsername/Voice-Transcriber/issues) for similar requests
2. Use the feature request template
3. Describe:
   - Use case and motivation
   - Proposed API or interface
   - Alternative approaches considered

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YourUsername/Voice-Transcriber.git
cd Voice-Transcriber

# Add upstream remote
git remote add upstream https://github.com/OriginalOwner/Voice-Transcriber.git
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Development Dependencies

```bash
pip install --upgrade pip
pip install setuptools wheel
pip install -r requirements-dev.txt
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `mypy` - Static type checker
- `ruff` - Fast linter

### 4. Install Pre-Commit Hooks (Optional)

```bash
pre-commit install
```

This automatically runs `black`, `ruff`, and `mypy` before each commit.

---

## Code Style

### Formatting with Black

We use [Black](https://black.readthedocs.io/) for code formatting.

**Run formatter:**
```bash
black src/ tests/
```

**Check formatting:**
```bash
black --check src/ tests/
```

**Configuration:**
```python
# pyproject.toml
[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311", "py312"]
```

### Linting with Ruff

[Ruff](https://docs.astral.sh/ruff/) is used for fast linting.

**Run linter:**
```bash
ruff check src/ tests/
```

**Auto-fix issues:**
```bash
ruff check --fix src/ tests/
```

### Type Checking with Mypy

**Run type checker:**
```bash
mypy src/
```

**Configuration:**
```python
# pyproject.toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Code Style Guidelines

#### 1. Type Hints

**Always include type hints** for function signatures:

```python
# Good
def transcribe(self, audio_path: str) -> str:
    ...

# Bad
def transcribe(self, audio_path):
    ...
```

**Use `Optional` for nullable values:**

```python
from typing import Optional

def load_config(path: Optional[str] = None) -> Config:
    ...
```

#### 2. Docstrings

Use **Google-style docstrings** for all public functions/classes:

```python
def transcribe_batch(
    self,
    folder_path: str,
    output_csv: str,
    extensions: Optional[List[str]] = None,
) -> None:
    """Transcribe all audio files in a folder with parallel processing.

    Args:
        folder_path: Directory containing audio files.
        output_csv: Destination CSV file path for results.
        extensions: Optional list of file extensions to include.

    Raises:
        FileNotFoundError: If folder does not exist.
        OSError: If CSV write fails.
    """
```

#### 3. Naming Conventions

- **Classes**: PascalCase (`TranscriptionEngine`, `OpenAIEngine`)
- **Functions/methods**: snake_case (`transcribe_file`, `load_config`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_WORKERS`, `DEFAULT_LANGUAGE`)
- **Private methods**: Prefix with underscore (`_internal_method`)

#### 4. Import Order

Use `isort` or follow this order:
1. Standard library imports
2. Third-party imports
3. Local imports

```python
# Standard library
import logging
from pathlib import Path
from typing import List, Optional

# Third-party
from tqdm import tqdm
import torch

# Local
from .engines.base import TranscriptionEngine
from .utils.normalizer import BasicPersianNormalizer
```

#### 5. Error Handling

**Be specific with exceptions:**

```python
# Good
try:
    config = load_config(path)
except FileNotFoundError:
    logger.error("Config file not found: %s", path)
    raise
except yaml.YAMLError as exc:
    logger.error("Invalid YAML syntax: %s", exc)
    raise ValueError(f"Config parse error: {exc}") from exc

# Bad
try:
    config = load_config(path)
except Exception:
    pass  # Silent failure
```

**Always log errors:**

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = risky_operation()
except ValueError as exc:
    logger.error("Operation failed: %s", exc)
    raise
```

---

## Adding a New Transcription Engine

Follow these steps to add support for a new transcription service.

### Step 1: Create Engine File

Create `src/engines/my_engine.py`:

```python
"""My custom transcription engine implementation."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import TranscriptionEngine

logger = logging.getLogger(__name__)


class MyCustomEngine(TranscriptionEngine):
    """Transcription engine using MyService API."""

    def __init__(self, api_key: str, language: str = "en") -> None:
        """Initialize the engine.

        Args:
            api_key: API key for MyService.
            language: Language code (default: "en").

        Raises:
            ValueError: If API key is missing or invalid.
        """
        super().__init__(language=language)
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using MyService API.

        Args:
            audio_path: Path to audio file.

        Returns:
            Transcribed text.

        Raises:
            FileNotFoundError: If audio file not found.
            ValueError: If audio format unsupported.
        """
        path = self.validate_audio_path(audio_path)
        logger.info("Transcribing with MyService: %s", path)

        # Your implementation here
        # Example:
        # with path.open("rb") as f:
        #     response = my_service_client.transcribe(f)
        # return response.text

        raise NotImplementedError("Implement transcription logic here")
```

### Step 2: Update `__init__.py`

Add export to `src/engines/__init__.py`:

```python
from .offline import OfflineTranscriptionEngine
from .openai import OpenAITranscriptionEngine
from .my_engine import MyCustomEngine  # Add this

__all__ = [
    "OfflineTranscriptionEngine",
    "OpenAITranscriptionEngine",
    "MyCustomEngine",  # Add this
]
```

### Step 3: Add CLI Support

Update `src/cli.py`:

```python
def _create_engine(args: argparse.Namespace, config: Config) -> object:
    """Instantiate the requested transcription engine."""

    if args.engine == "openai":
        api_key = args.api_key or config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAITranscriptionEngine(api_key=api_key, language=args.language)

    if args.engine == "my_custom":  # Add this
        api_key = args.api_key or os.getenv("MY_SERVICE_API_KEY")
        if not api_key:
            raise ValueError("MyService API key is required")
        return MyCustomEngine(api_key=api_key, language=args.language)

    # Offline engine fallback
    offline_settings = config.offline or {}
    device = args.device or offline_settings.get("device")
    return OfflineTranscriptionEngine(language=args.language, device=device)
```

Update argument choices:

```python
transcribe_parser.add_argument(
    "--engine",
    choices=["openai", "offline", "my_custom"],  # Add "my_custom"
    default="openai",
    help="Transcription engine to use (default: openai)",
)
```

### Step 4: Write Tests

Create `tests/test_my_engine.py`:

```python
import pytest
from pathlib import Path
from src.engines.my_engine import MyCustomEngine


def test_init_requires_api_key():
    """Test that API key is required."""
    with pytest.raises(ValueError, match="API key is required"):
        MyCustomEngine(api_key="")


def test_transcribe_validates_path(tmp_path: Path):
    """Test that transcribe validates file path."""
    engine = MyCustomEngine(api_key="test_key")
    missing_file = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        engine.transcribe(str(missing_file))


def test_transcribe_rejects_unsupported_format(tmp_path: Path):
    """Test that unsupported formats are rejected."""
    engine = MyCustomEngine(api_key="test_key")
    bad_file = tmp_path / "audio.txt"
    bad_file.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported audio format"):
        engine.transcribe(str(bad_file))
```

### Step 5: Update Documentation

Add to `docs/API.md`:

```markdown
### MyCustomEngine

**Module:** `src.engines.my_engine`

Custom transcription engine using MyService API.

#### Initialization

\```python
from src.engines.my_engine import MyCustomEngine

engine = MyCustomEngine(
    api_key="your_api_key",
    language="en"
)
\```

#### Parameters

- `api_key` (str): API key for MyService
- `language` (str): Language code (default: `"en"`)

#### Methods

##### `transcribe(audio_path: str) -> str`

Transcribe audio using MyService.

\```python
transcript = engine.transcribe("audio.wav")
\```

**Returns:** Transcribed text

**Raises:**
- `FileNotFoundError`: Audio file not found
- `ValueError`: Unsupported audio format
```

---

## Testing Requirements

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run specific test file:**
```bash
pytest tests/test_my_engine.py
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=term --cov-report=html
```

View HTML report: `open htmlcov/index.html`

### Coverage Goals

- **Core logic**: ≥80% coverage
- **Critical paths** (engine.transcribe, manager.transcribe_batch): ≥90%
- **Utilities**: ≥70%

### Writing Good Tests

#### 1. Use Fixtures

```python
import pytest
from pathlib import Path


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"fake audio data")
    return audio_file


def test_transcribe_with_fixture(sample_audio: Path):
    engine = MyEngine()
    result = engine.transcribe(str(sample_audio))
    assert isinstance(result, str)
```

#### 2. Test Error Cases

```python
def test_transcribe_missing_file():
    engine = MyEngine()
    with pytest.raises(FileNotFoundError):
        engine.transcribe("nonexistent.wav")


def test_transcribe_invalid_format(tmp_path: Path):
    bad_file = tmp_path / "file.txt"
    bad_file.write_text("not audio")

    engine = MyEngine()
    with pytest.raises(ValueError, match="Unsupported format"):
        engine.transcribe(str(bad_file))
```

#### 3. Mock External Dependencies

```python
from unittest.mock import patch, MagicMock


def test_openai_transcribe_with_mock(monkeypatch):
    """Test OpenAI engine with mocked API client."""
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value.text = "mocked transcript"

    with patch("src.engines.openai.OpenAI", return_value=mock_client):
        engine = OpenAITranscriptionEngine(api_key="fake_key")
        result = engine.transcribe("audio.wav")

    assert result == "mocked transcript"
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-new-feature
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring

### 2. Make Changes

- Follow code style guidelines
- Add/update tests
- Update documentation (if needed)
- Commit with clear messages

### 3. Run Pre-Submission Checks

```bash
# Format code
black src/ tests/

# Run linter
ruff check src/ tests/

# Type check
mypy src/

# Run tests with coverage
pytest --cov=src --cov-report=term
```

### 4. Commit with Clear Messages

```bash
git add .
git commit -m "Add MyCustomEngine for new transcription service

- Implement MyCustomEngine class
- Add CLI support for --engine my_custom
- Write unit tests with 85% coverage
- Update API.md with usage examples

Closes #123"
```

**Commit message format:**
- First line: Brief summary (50 chars max)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference related issues with `Closes #123`

### 5. Push and Open Pull Request

```bash
git push origin feature/my-new-feature
```

Then open PR on GitHub with:
- **Title**: Clear, concise summary
- **Description**:
  - What changes were made
  - Why these changes are needed
  - How to test
  - Screenshots (if UI changes)
- **Checklist**:
  - [ ] Tests added/updated
  - [ ] Documentation updated
  - [ ] Code follows style guidelines
  - [ ] All tests passing
  - [ ] Coverage maintained or improved

### 6. Code Review

- Address reviewer feedback
- Make requested changes in new commits (don't force-push)
- Respond to comments politely

### 7. Merge

Once approved and CI passes:
- Maintainer will merge PR
- Delete feature branch after merge

---

## Issue Guidelines

### Bug Reports

Use this template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command `...`
2. See error `...`

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.13.5]
- Package version: [e.g., 2.0.0]

**Logs/Error messages**
```
Paste full error traceback here
```

**Additional context**
Any other relevant information.
```

### Feature Requests

Use this template:

```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. "I'm frustrated when [...]"

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Mockups, examples, or related issues.
```

---

## Development Workflow Example

Complete workflow for adding a new feature:

```bash
# 1. Sync with upstream
git checkout main
git fetch upstream
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/add-azure-engine

# 3. Install dev dependencies
pip install -r requirements-dev.txt

# 4. Implement feature
# - Create src/engines/azure_engine.py
# - Update src/engines/__init__.py
# - Add CLI support in src/cli.py
# - Write tests in tests/test_azure_engine.py

# 5. Run checks
black src/ tests/
ruff check src/ tests/
mypy src/
pytest --cov=src --cov-report=term

# 6. Commit
git add .
git commit -m "Add Azure Speech Service engine

- Implement AzureEngine class with auth
- Add CLI --engine azure option
- Write unit tests (87% coverage)
- Update API.md with examples

Closes #45"

# 7. Push and open PR
git push origin feature/add-azure-engine
# Then open PR on GitHub
```

---

## Questions?

- Check [GitHub Discussions](https://github.com/YourUsername/Voice-Transcriber/discussions)
- Open an issue with tag `question`
- Email: maintainer@example.com

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
