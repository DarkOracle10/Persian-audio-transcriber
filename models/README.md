# Models

Use DVC to version model artifacts stored in this directory.

Example workflow:
1. Place trained model under `models/artifacts/` (e.g., `models/artifacts/model-v1/weights.bin`).
2. Track with DVC:
   ```bash
   dvc add models/artifacts
   git add models/artifacts.dvc .gitignore
   git commit -m "Track model artifacts with DVC"
   ```
3. Push to remote storage (configure first):
   ```bash
   dvc remote add -d storage <remote-url>
   dvc push
   ```
