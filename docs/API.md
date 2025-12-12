# API Documentation

Python API for programmatic use of the transcription tool.

## PersianAudioTranscriber Class

Main class for audio/video transcription.

### Initialization

```python
from main import PersianAudioTranscriber

transcriber = PersianAudioTranscriber(
    engine="faster_whisper",
    model_size="medium",
    api_key=None,
    language="fa",
    normalize_persian=True
)
```

#### Parameters

- `engine` (str): Transcription engine (`whisper`, `faster_whisper`, `google`, `openai_api`)
- `model_size` (str): Model size (`tiny`, `base`, `small`, `medium`, `large`, `large-v3`)
- `api_key` (str, optional): OpenAI API key (for `openai_api` engine)
- `language` (str): Language code (default: `fa` for Persian)
- `normalize_persian` (bool): Enable Persian text normalization

### Methods

#### transcribe_file()

Transcribe a single audio/video file.

```python
result = transcriber.transcribe_file("audio.mp3")
```

**Returns:**
- `dict` with keys:
  - `text`: Normalized transcription
  - `text_raw`: Original transcription
  - `language`: Detected language
  - `segments`: List of segments with timestamps
  - `error`: Error message (if any)

#### scan_and_transcribe()

Process all audio files in a folder.

```python
results = transcriber.scan_and_transcribe(
    folder_path="./audio_folder",
    output_dir="./output",
    save_format="txt"
)
```

**Parameters:**
- `folder_path` (str): Path to folder with audio files
- `output_dir` (str, optional): Output directory
- `save_format` (str): Output format (`txt`, `json`, `srt`)

**Returns:**
- `list`: List of result dictionaries

### Examples

See [README.md](../README.md) for usage examples.

