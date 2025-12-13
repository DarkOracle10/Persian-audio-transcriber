"""Setup script for Voice Transcription Toolkit."""

from pathlib import Path
from setuptools import setup, find_packages

# Read version from persian_transcriber package
version_file = Path(__file__).parent / "src" / "persian_transcriber" / "__init__.py"
version = "1.0.0"
for line in version_file.read_text(encoding="utf-8").splitlines():
    if line.startswith("__version__"):
        version = line.split("=")[1].strip().strip('"').strip("'")
        break

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="voice-transcriber",
    version=version,
    author="DarkOracle10",
    author_email="darkoracle3860@gmail.com",
    description="GPU-accelerated voice transcription toolkit with Persian/Farsi language support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DarkOracle10/Voice-Transcriber",
    project_urls={
        "Bug Tracker": "https://github.com/DarkOracle10/Voice-Transcriber/issues",
        "Documentation": "https://github.com/DarkOracle10/Voice-Transcriber#readme",
        "Source Code": "https://github.com/DarkOracle10/Voice-Transcriber",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0.2",
        "python-dotenv>=1.1.0",
        "httpx>=0.28.1",
        "certifi>=2025.10.5",
        "click>=8.3.0",
        "colorama>=0.4.6",
        "faster-whisper>=1.2.1",
        "ctranslate2>=4.6.1",
        "openai>=1.86.0",
        "ffmpeg-python>=0.1.17",
        "pydub>=0.25.1",
        "av>=16.0.1",
        "coloredlogs>=15.0.1",
        "tqdm>=4.67.1",
        "typing-extensions>=4.14.0",
    ],
    extras_require={
        "persian": ["hazm>=0.10.0"],
        "gpu": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ],
        "dev": [
            "pytest>=8.3.5",
            "pytest-cov>=6.1.1",
            "pytest-asyncio>=1.0.0",
            "pytest-mock>=3.14.0",
            "ruff>=0.11.12",
            "black>=25.1.0",
            "mypy>=1.16.0",
            "types-pyyaml>=6.0.12",
        ],
        "docs": [
            "mkdocs>=1.6.1",
            "mkdocs-material>=9.5",
            "mkdocstrings>=0.29.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "voice-transcriber=persian_transcriber.cli:main",
            "vtranscribe=persian_transcriber.cli:main",
        ],
    },
    keywords=[
        "transcription",
        "speech-to-text",
        "persian",
        "farsi",
        "whisper",
        "audio",
        "video",
        "gpu",
        "cuda",
        "nlp",
    ],
    include_package_data=True,
    zip_safe=False,
)
