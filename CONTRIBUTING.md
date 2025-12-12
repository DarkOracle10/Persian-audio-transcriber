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
git clone <repository-url>
cd Voice-transcription-tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Setting Up Code Coverage (For Repository Maintainers)

This repository uses [Codecov](https://codecov.io) for tracking code coverage. To enable code coverage reporting in GitHub Actions, the `CODECOV_TOKEN` secret needs to be configured.

### Adding the CODECOV_TOKEN Secret

**Note**: These instructions are for the main repository maintainers. If you have forked this repository, replace the URL with your fork's URL.

1. Go to https://github.com/DarkOracle10/Voice-Transcriber
2. Click **Settings** (top bar of the repo)
3. In the left sidebar, under **Security**, choose **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Set **Name** to `CODECOV_TOKEN`
6. Paste your Codecov token into **Secret**
7. Click **Add secret**

### Getting Your Codecov Token

1. **Sign up or log in** at https://codecov.io
   - You can sign in using your GitHub account

2. **Add your repository to Codecov**:
   - Click on "Add new repository" or the "+" icon
   - Select your GitHub organization/account (DarkOracle10)
   - Find and select the "Voice-Transcriber" repository from the list
   - Click to enable it

3. **Get your upload token**:
   - Once the repository is added, you'll be taken to the repository settings
   - Look for the "Settings" tab in Codecov
   - Find the "Upload Token" or "Repository Upload Token" section
   - Click to reveal or copy the token (it's a long alphanumeric string)
   - Alternatively, it may be shown on the initial setup page

4. **Use this token** in the GitHub secret configuration steps above

Once configured, the CI workflow will automatically upload coverage reports to Codecov after running tests.

**Note on Pull Requests**: For security reasons, GitHub Actions does not provide secrets to workflows triggered by pull requests from forks. The CI workflow is configured to only upload coverage reports for:
- Pushes to the main branch
- Pull requests from branches within the same repository

This prevents potential security issues while still allowing the test suite to run on all pull requests.

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing! 🎉

