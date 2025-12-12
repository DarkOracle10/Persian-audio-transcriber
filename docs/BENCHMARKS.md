# Performance Benchmarks

Performance metrics for different configurations.

## Test System

- **GPU**: NVIDIA RTX 3060
- **CPU**: Intel Core i7
- **RAM**: 16GB
- **Audio**: 1-minute Persian audio file

## Results

### CPU Performance

| Model | Time | Real-time Factor |
|-------|------|------------------|
| tiny  | 30s  | 0.5x |
| base  | 60s  | 1.0x |
| small | 120s | 2.0x |
| medium| 200s | 3.3x |

### GPU Performance (FP16)

| Model | Time | Real-time Factor | Speedup |
|-------|------|------------------|---------|
| tiny  | 6s   | 0.1x | 5x |
| base  | 7s   | 0.12x | 8.5x |
| small | 20s  | 0.33x | 6x |
| medium| 50s  | 0.83x | 4x |

*Real-time factor < 1 means faster than real-time*

## Recommendations

- **Best Speed**: `tiny` model with GPU
- **Best Balance**: `medium` model with GPU
- **Best Accuracy**: `large-v3` model with GPU
- **No GPU**: Use `base` or `small` model

