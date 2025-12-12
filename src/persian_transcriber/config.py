"""
Configuration management for Persian Transcriber.

This module provides configuration dataclasses and utilities for
managing application settings.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .engines.base import EngineType
from .normalizers import NormalizerType
from .output import OutputFormat


class DeviceType(str, Enum):
    """Device types for computation."""
    
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class EngineConfig:
    """
    Configuration for transcription engine.
    
    Attributes:
        type: Type of engine to use.
        model_size: Model size for Whisper-based engines.
        device: Computation device (auto, cuda, cpu, mps).
        compute_type: Precision type (float16, int8, float32).
    """
    
    type: Union[str, EngineType] = EngineType.FASTER_WHISPER
    model_size: str = "medium"
    device: Union[str, DeviceType] = DeviceType.AUTO
    compute_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": str(self.type),
            "model_size": self.model_size,
            "device": str(self.device),
            "compute_type": self.compute_type,
        }


@dataclass
class NormalizerConfig:
    """
    Configuration for text normalization.
    
    Attributes:
        enabled: Whether normalization is enabled.
        type: Type of normalizer to use.
    """
    
    enabled: bool = True
    type: Union[str, NormalizerType] = NormalizerType.PERSIAN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "type": str(self.type),
        }


@dataclass
class OutputConfig:
    """
    Configuration for output formatting.
    
    Attributes:
        format: Output format type.
        directory: Output directory for saved files.
        include_timestamps: Include timestamps in text output.
    """
    
    format: Union[str, OutputFormat] = OutputFormat.TXT
    directory: Optional[Path] = None
    include_timestamps: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": str(self.format),
            "directory": str(self.directory) if self.directory else None,
            "include_timestamps": self.include_timestamps,
        }


@dataclass
class TranscriberConfig:
    """
    Main configuration for PersianAudioTranscriber.
    
    This class holds all configuration options for the transcriber,
    including engine settings, normalization options, and output format.
    
    Attributes:
        language: Default language code for transcription.
        engine: Engine configuration.
        normalizer: Normalizer configuration.
        output: Output configuration.
        openai_api_key: API key for OpenAI (if using OpenAI API engine).
        verbose: Enable verbose logging.
    
    Example:
        >>> config = TranscriberConfig(
        ...     language="fa",
        ...     engine=EngineConfig(model_size="large-v3"),
        ... )
        >>> transcriber = PersianAudioTranscriber(config=config)
    """
    
    language: str = "fa"
    engine: EngineConfig = field(default_factory=EngineConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "language": self.language,
            "engine": self.engine.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "output": self.output.to_dict(),
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriberConfig":
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary.
            
        Returns:
            TranscriberConfig: Configuration instance.
        """
        config = cls()
        
        if "language" in data:
            config.language = data["language"]
        
        if "verbose" in data:
            config.verbose = data["verbose"]
        
        if "openai_api_key" in data:
            config.openai_api_key = data["openai_api_key"]
        
        if "engine" in data:
            engine_data = data["engine"]
            config.engine = EngineConfig(
                type=engine_data.get("type", EngineType.FASTER_WHISPER),
                model_size=engine_data.get("model_size", "medium"),
                device=engine_data.get("device", DeviceType.AUTO),
                compute_type=engine_data.get("compute_type"),
            )
        
        if "normalizer" in data:
            norm_data = data["normalizer"]
            config.normalizer = NormalizerConfig(
                enabled=norm_data.get("enabled", True),
                type=norm_data.get("type", NormalizerType.PERSIAN),
            )
        
        if "output" in data:
            output_data = data["output"]
            config.output = OutputConfig(
                format=output_data.get("format", OutputFormat.TXT),
                directory=Path(output_data["directory"]) if output_data.get("directory") else None,
                include_timestamps=output_data.get("include_timestamps", False),
            )
        
        return config
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "TranscriberConfig":
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            path: Path to configuration file.
            
        Returns:
            TranscriberConfig: Loaded configuration.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        content = path.read_text(encoding="utf-8")
        
        if path.suffix.lower() == ".json":
            import json
            data = json.loads(content)
        elif path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files. Run: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls.from_dict(data)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            path: Path to save the configuration file.
        """
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        path.write_text(content, encoding="utf-8")


# Default configuration
DEFAULT_CONFIG = TranscriberConfig()
