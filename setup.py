"""
Setup script for persian-transcriber package.

This file exists for backwards compatibility with older pip versions
and for editable installs. The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read version from package
version = "2.0.0"
try:
    version_file = this_directory / "src" / "persian_transcriber" / "__init__.py"
    if version_file.exists():
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break
except Exception:
    pass

setup(
    name="persian-transcriber",
    version=version,
    author="Dark Oracle",
    author_email="darkoracle3860@gmail.com",
    description="GPU-accelerated audio/video transcription tool with Persian/Farsi language support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darkoracle/persian-transcriber",
    project_urls={
        "Documentation": "https://github.com/darkoracle/persian-transcriber#readme",
        "Bug Tracker": "https://github.com/darkoracle/persian-transcriber/issues",
        "Changelog": "https://github.com/darkoracle/persian-transcriber/blob/main/CHANGELOG.md",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
        "openai-whisper>=20231117",
        "faster-whisper>=1.0.0",
        "pydub>=0.25.1",
        "numpy>=1.24.0",
    ],
    extras_require={
        "persian": ["hazm>=0.10.0"],
        "gpu": [
            "torch>=2.0.0",
            "nvidia-cudnn-cu12>=9.0.0",
            "nvidia-cublas-cu12>=12.0.0",
        ],
        "openai": ["openai>=1.0.0"],
        "google": ["SpeechRecognition>=3.10.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "persian-transcribe=persian_transcriber.cli:main",
            "ptranscribe=persian_transcriber.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "persian_transcriber": ["py.typed"],
    },
    zip_safe=False,
)

