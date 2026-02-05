"""
Cross-platform CUDA/GPU detection and configuration.

This module provides utilities for detecting and configuring GPU
acceleration across different operating systems (Windows, Linux, macOS).

Configuration is loaded from config.yaml if available.

Supports:
- Windows: NVIDIA CUDA Toolkit + pip-installed NVIDIA packages
- Linux: System CUDA installation + LD_LIBRARY_PATH
- macOS: Apple Silicon MPS (Metal Performance Shaders)
"""

import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Global configuration cache
_cuda_config_cache: Optional[Dict[str, Any]] = None


def _load_cuda_config() -> Dict[str, Any]:
    """Load CUDA configuration from config.yaml."""
    global _cuda_config_cache

    if _cuda_config_cache is not None:
        return _cuda_config_cache

    # Default configuration
    default_config: Dict[str, Any] = {
        "cuda_home_path": None,
        "library_paths": {
            "windows": [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
            ],
            "linux": [
                "/usr/local/cuda/lib64",
                "/usr/local/cuda-12/lib64",
                "/usr/local/cuda-11/lib64",
                "/opt/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ],
            "darwin": [],
        },
        "use_fp16": True,
        "compute_type": "float16",
        "fallback_to_cpu": True,
        "device": "auto",
    }

    # Search for config.yaml
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path(__file__).parent.parent.parent.parent / "config.yaml",
        Path(__file__).parent.parent.parent.parent / "config.yml",
        Path.home() / ".persian_transcriber" / "config.yaml",
    ]

    # Also check environment variable
    if os.environ.get("PERSIAN_TRANSCRIBER_CONFIG"):
        search_paths.insert(0, Path(os.environ["PERSIAN_TRANSCRIBER_CONFIG"]))

    config_path = None
    for path in search_paths:
        if path.exists():
            config_path = path
            break

    if config_path is not None:
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config and "cuda" in yaml_config:
                cuda_config = yaml_config["cuda"]
                for key in default_config:
                    if key in cuda_config:
                        # Handle nested library_paths
                        if key == "library_paths" and isinstance(cuda_config[key], dict):
                            for os_key in cuda_config[key]:
                                if os_key in default_config["library_paths"]:
                                    default_config["library_paths"][os_key] = cuda_config[key][
                                        os_key
                                    ]
                        else:
                            default_config[key] = cuda_config[key]

                logger.debug(f"Loaded CUDA config from {config_path}")
        except ImportError:
            logger.debug("PyYAML not installed, using default CUDA config")
        except Exception as e:
            logger.debug(f"Error loading CUDA config: {e}")

    _cuda_config_cache = default_config
    return default_config


@dataclass
class GPUInfo:
    """
    GPU device information container.

    Attributes:
        available: Whether GPU acceleration is available.
        device_name: Name of the GPU device.
        device_type: Type of device ("cuda", "mps", or "cpu").
        cuda_version: CUDA version string if applicable.
        compute_capability: CUDA compute capability (e.g., "8.6").
        memory_total_mb: Total GPU memory in megabytes.
        memory_free_mb: Free GPU memory in megabytes.
        driver_version: GPU driver version.
        platform: Operating system name.
    """

    available: bool = False
    device_name: Optional[str] = None
    device_type: str = "cpu"
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    driver_version: Optional[str] = None
    platform: str = field(default_factory=lambda: platform.system())


def get_platform() -> str:
    """
    Return normalized platform name.

    Returns:
        str: "windows", "linux", or "macos"
    """
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system


def _get_cuda_paths_windows() -> List[Path]:
    """
    Get CUDA library paths on Windows.

    Searches for (in order):
    1. Paths from config.yaml cuda.library_paths.windows
    2. System CUDA Toolkit installation
    3. Pip-installed NVIDIA packages in site-packages

    Returns:
        List[Path]: List of paths containing CUDA libraries.
    """
    paths: List[Path] = []
    config = _load_cuda_config()

    # Priority 0: Custom CUDA home from config
    cuda_home = config.get("cuda_home_path")
    if cuda_home:
        cuda_home_path = Path(cuda_home)
        if cuda_home_path.exists():
            cuda_bin = cuda_home_path / "bin"
            if cuda_bin.exists():
                paths.append(cuda_bin)
                logger.debug(f"Found CUDA from config cuda_home_path: {cuda_bin}")

    # Priority 1: Config-specified library paths
    config_paths = config.get("library_paths", {}).get("windows", [])
    for path_str in config_paths:
        cuda_path = Path(path_str)
        if cuda_path.exists() and cuda_path not in paths:
            # Check for key DLL files
            cublas_dlls = list(cuda_path.glob("cublas*.dll"))
            if cublas_dlls:
                paths.append(cuda_path)
                logger.debug(f"Found CUDA from config: {cuda_path}")
                break

    # Priority 2: System CUDA Toolkit auto-detection (if no config paths found)
    if not paths:
        cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if cuda_base.exists():
            # Find CUDA 12.x versions (prioritize newer versions)
            cuda_versions = [
                "v12.6",
                "v12.5",
                "v12.4",
                "v12.3",
                "v12.2",
                "v12.1",
                "v12.0",
                "v11.8",
                "v11.7",
                "v11.6",
            ]
            for version in cuda_versions:
                cuda_bin = cuda_base / version / "bin"
                if cuda_bin.exists():
                    # Check for key DLL files
                    cublas_dll = cuda_bin / "cublas64_12.dll"
                    cublas_dll_11 = cuda_bin / "cublas64_11.dll"
                    if cublas_dll.exists() or cublas_dll_11.exists():
                        paths.append(cuda_bin)
                        logger.debug(f"Found system CUDA installation: {cuda_bin}")
                        break

    # Priority 3: Pip-installed NVIDIA packages
    try:
        import site

        site_packages_list = site.getsitepackages()
        if not site_packages_list:
            site_packages_list = [os.path.join(sys.prefix, "Lib", "site-packages")]

        for site_packages in site_packages_list:
            site_path = Path(site_packages)
            nvidia_packages = ["cublas", "cudnn", "cuda_runtime", "cufft", "curand"]

            for pkg in nvidia_packages:
                pkg_bin = site_path / "nvidia" / pkg / "bin"
                if pkg_bin.exists() and pkg_bin not in paths:
                    paths.append(pkg_bin)
                    logger.debug(f"Found pip NVIDIA package: {pkg_bin}")
    except Exception as e:
        logger.debug(f"Error scanning pip NVIDIA packages: {e}")

    return paths


def _get_cuda_paths_linux() -> List[Path]:
    """
    Get CUDA library paths on Linux.

    Searches for (in order):
    1. Paths from config.yaml cuda.library_paths.linux
    2. Standard CUDA installation paths (/usr/local/cuda)
    3. Paths in LD_LIBRARY_PATH environment variable

    Returns:
        List[Path]: List of paths containing CUDA libraries.
    """
    paths: List[Path] = []
    config = _load_cuda_config()

    # Priority 0: Custom CUDA home from config
    cuda_home = config.get("cuda_home_path")
    if cuda_home:
        cuda_home_path = Path(cuda_home)
        lib_path = cuda_home_path / "lib64"
        if lib_path.exists():
            paths.append(lib_path)
            logger.debug(f"Found CUDA from config cuda_home_path: {lib_path}")

    # Priority 1: Config-specified library paths
    config_paths = config.get("library_paths", {}).get("linux", [])
    for path_str in config_paths:
        lib_path = Path(path_str)
        if lib_path.exists() and lib_path not in paths:
            # Check for key library files
            libcublas = lib_path / "libcublas.so"
            if libcublas.exists() or any(lib_path.glob("libcublas.so.*")):
                paths.append(lib_path)
                logger.debug(f"Found CUDA from config: {lib_path}")
                break

    # Priority 2: Standard CUDA installation paths (fallback)
    if not paths:
        standard_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-11/lib64",
            "/opt/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
        ]

        for cuda_path in standard_paths:
            lib_path = Path(cuda_path)
            if lib_path.exists():
                # Check for key library files
                libcublas = lib_path / "libcublas.so"
                if libcublas.exists() or any(lib_path.glob("libcublas.so.*")):
                    paths.append(lib_path)
                    logger.debug(f"Found CUDA libraries: {lib_path}")
                    break

    # Priority 3: LD_LIBRARY_PATH entries
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path_str in ld_library_path.split(":"):
        if path_str and "cuda" in path_str.lower():
            path = Path(path_str)
            if path.exists() and path not in paths:
                paths.append(path)
                logger.debug(f"Found CUDA path in LD_LIBRARY_PATH: {path}")

    return paths


def _get_cuda_paths_macos() -> List[Path]:
    """
    Get CUDA paths on macOS.

    Note: CUDA is not supported on modern macOS. Apple Silicon uses
    Metal Performance Shaders (MPS) instead.

    Returns:
        List[Path]: Empty list (CUDA not supported on macOS).
    """
    logger.info("macOS detected: CUDA not available, will use MPS for Apple Silicon or CPU")
    return []


def setup_cuda_paths() -> bool:
    """
    Configure environment for CUDA libraries.

    Detects the platform and adds appropriate CUDA library paths
    to the system PATH (Windows) or LD_LIBRARY_PATH (Linux).

    This function should be called before importing any CUDA-dependent
    libraries like faster-whisper or PyTorch.

    Returns:
        bool: True if CUDA paths were found and configured, False otherwise.

    Example:
        >>> from persian_transcriber.utils.cuda_setup import setup_cuda_paths
        >>> if setup_cuda_paths():
        ...     print("CUDA configured successfully")
        ... else:
        ...     print("CUDA not available, using CPU")
    """
    platform_name = get_platform()

    path_getters = {
        "windows": _get_cuda_paths_windows,
        "linux": _get_cuda_paths_linux,
        "macos": _get_cuda_paths_macos,
    }

    getter = path_getters.get(platform_name)
    if getter is None:
        logger.warning(f"Unknown platform: {platform_name}")
        return False

    paths = getter()

    if not paths:
        logger.info(f"No CUDA paths found on {platform_name}")
        return False

    # Add paths to environment
    if platform_name == "windows":
        current_path = os.environ.get("PATH", "")
        new_paths_str = ";".join(str(p) for p in paths if str(p) not in current_path)
        if new_paths_str:
            os.environ["PATH"] = new_paths_str + ";" + current_path
            logger.info(f"Added CUDA paths to PATH: {new_paths_str}")
    elif platform_name == "linux":
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_paths_str = ":".join(str(p) for p in paths if str(p) not in current_ld)
        if new_paths_str:
            os.environ["LD_LIBRARY_PATH"] = new_paths_str + ":" + current_ld
            logger.info(f"Added CUDA paths to LD_LIBRARY_PATH: {new_paths_str}")

    return True


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for PyTorch.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        logger.debug("PyTorch not installed, cannot check CUDA availability")
        return False
    except Exception as e:
        logger.debug(f"Error checking CUDA availability: {e}")
        return False


def is_mps_available() -> bool:
    """
    Check if Apple Metal Performance Shaders (MPS) is available.

    MPS is Apple's GPU acceleration framework for Apple Silicon Macs.

    Returns:
        bool: True if MPS is available, False otherwise.
    """
    try:
        import torch

        if hasattr(torch.backends, "mps"):
            return torch.backends.mps.is_available()
        return False
    except ImportError:
        logger.debug("PyTorch not installed, cannot check MPS availability")
        return False
    except Exception as e:
        logger.debug(f"Error checking MPS availability: {e}")
        return False


def get_best_device() -> str:
    """
    Get the best available compute device.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        str: Device identifier ("cuda", "mps", or "cpu").

    Example:
        >>> device = get_best_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
    """
    if is_cuda_available():
        return "cuda"
    if is_mps_available():
        return "mps"
    return "cpu"


def get_compute_type(device: str) -> str:
    """
    Get the recommended compute type for a device.

    Args:
        device: Device identifier ("cuda", "mps", or "cpu").

    Returns:
        str: Recommended compute type for the device.
            - "float16" for CUDA (faster, slightly less accurate)
            - "float32" for MPS
            - "int8" for CPU (optimized for CPU inference)
    """
    compute_types = {
        "cuda": "float16",
        "mps": "float32",
        "cpu": "int8",
    }
    return compute_types.get(device, "int8")


def get_device_info() -> GPUInfo:
    """
    Get detailed GPU information.

    Returns:
        GPUInfo: Dataclass containing GPU device details.

    Example:
        >>> info = get_device_info()
        >>> if info.available:
        ...     print(f"GPU: {info.device_name}")
        ...     print(f"Memory: {info.memory_total_mb} MB")
        ... else:
        ...     print("No GPU available")
    """
    info = GPUInfo(platform=get_platform())

    try:
        import torch

        if torch.cuda.is_available():
            info.available = True
            info.device_type = "cuda"
            info.device_name = torch.cuda.get_device_name(0)
            info.cuda_version = torch.version.cuda

            # Get device properties
            props = torch.cuda.get_device_properties(0)
            info.compute_capability = f"{props.major}.{props.minor}"
            info.memory_total_mb = props.total_memory // (1024 * 1024)

            # Get free memory
            try:
                free_memory, total_memory = torch.cuda.mem_get_info(0)
                info.memory_free_mb = free_memory // (1024 * 1024)
            except Exception:
                pass

            # Get driver version
            try:
                import subprocess

                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    info.driver_version = result.stdout.strip().split("\n")[0]
            except Exception:
                pass

            logger.info(
                f"CUDA GPU detected: {info.device_name} "
                f"(CUDA {info.cuda_version}, {info.memory_total_mb} MB)"
            )

        elif is_mps_available():
            info.available = True
            info.device_type = "mps"
            info.device_name = "Apple Silicon (MPS)"
            logger.info("Apple Silicon MPS detected")

        else:
            info.device_type = "cpu"
            logger.info("No GPU detected, using CPU")

    except ImportError:
        logger.warning("PyTorch not installed, cannot detect GPU")
        info.device_type = "cpu"
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")
        info.device_type = "cpu"

    return info


def ensure_cuda_initialized() -> GPUInfo:
    """
    Ensure CUDA is properly initialized and return device info.

    This is a convenience function that:
    1. Sets up CUDA paths
    2. Detects available GPU
    3. Returns device information

    Returns:
        GPUInfo: GPU device information.

    Example:
        >>> info = ensure_cuda_initialized()
        >>> print(f"Using: {info.device_name or 'CPU'}")
    """
    setup_cuda_paths()
    return get_device_info()
