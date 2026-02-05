# Contributing to Persian Audio Transcription Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, GPU model)
   - Error messages/logs

### Suggesting Enhancements

1. Check existing Issues for similar suggestions
2. Create a new issue with:
   - Clear description of the enhancement
   - Use case or motivation
   - Proposed implementation (if applicable)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   - Test with different audio formats
   - Test with and without GPU
   - Test error handling

5. **Commit your changes**
   ```bash
   git commit -m "Add: Description of your changes"
   ```
   - Use clear commit messages
   - Prefix with: `Add:`, `Fix:`, `Update:`, `Remove:`

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Describe your changes clearly
   - Reference related issues
   - Include test results

## Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Comment complex logic

## Testing

- Test on different operating systems if possible
- Test with various audio formats
- Test error cases
- Verify GPU acceleration works

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update CHANGELOG.md

## Development Setup

```bash
# Clone repository
git clone https://github.com/DarkOracle10/Persian-audio-transcriber.git
cd Persian-audio-transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install --upgrade pip
pip install setuptools wheel
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

## Running Tests Locally

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=persian_transcriber --cov-report=term --cov-report=html

# Run specific test file
pytest tests/test_engines.py

# Run tests matching a pattern
pytest -k "test_engine"
```

## Code Style Guidelines

We use several tools to maintain code quality:

### Black - Code Formatting
```bash
# Format all code
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```
Configuration: 100 character line length

### Ruff - Linting
```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

### Mypy - Type Checking
```bash
# Run type checks
mypy src/persian_transcriber
```
We enforce strict type checking. All functions should have type hints.

### Pre-commit Hooks (Recommended)
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# This will automatically run black, ruff, and mypy before each commit
```

## How to Add a New Transcription Engine

Follow these steps to add a new transcription engine:

1. **Create a new engine class** in `src/persian_transcriber/engines/`
   - Inherit from `BaseTranscriptionEngine`
   - Implement required methods: `transcribe()`, `is_available()`

2. **Example structure:**
   ```python
   from .base import BaseTranscriptionEngine
   
   class MyNewEngine(BaseTranscriptionEngine):
       def __init__(self, config: Dict[str, Any]) -> None:
           super().__init__(config)
           # Initialize your engine
       
       def transcribe(self, audio_path: str) -> str:
           # Implement transcription logic
           pass
       
       @classmethod
       def is_available(cls) -> bool:
           # Check if engine dependencies are available
           return True
   ```

3. **Register the engine** in `src/persian_transcriber/engines/__init__.py`

4. **Add tests** in `tests/test_engines.py`
   - Test initialization
   - Test transcription
   - Test error handling

5. **Update documentation:**
   - Add engine to README.md
   - Document configuration options
   - Add usage examples

For detailed architecture, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing! 🎉

