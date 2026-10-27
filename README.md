# Chilli Leaf Disease Screening

Local research project for training a lightweight image classifier on the original and augmented chilli-leaf folders, then predicting individual or batch images through a local web page.

## Runtime

The setup script installs the project dependencies into a local `.venv`. Training uses CUDA when a compatible GPU-enabled PyTorch installation is available and otherwise falls back to CPU. The first training run downloads MobileNetV3 pretrained weights unless `--no-pretrained` is selected. See [PyTorch Start Locally](https://pytorch.org/get-started/locally/) for current platform-specific installation options.

## Setup (PowerShell)

From this project folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_env.ps1
```

The setup script creates `.venv` and installs `requirements.txt` only in that environment.

## Train locally

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py `
  --original-root "C:\path\to\Chili Leaf Disease Original Dataset" `
  --augmented-root "C:\path\to\Chili Leaf Disease Augmented Dataset" `
  --epochs 15 --batch-size 16
```

Training detects CUDA automatically and falls back to CPU if CUDA is unavailable. Validation and test sets contain original images only. Images with the same pre-transformation filename stem or exact SHA-256 hash stay in one partition, so augmented variants and exact copies cannot cross the split. Files with identical hashes under different class labels are excluded from training and listed in the split manifest for review. The script writes the checkpoint to `models/best_model.pt`, metrics to `reports/training_metrics.json`, and the auditable group assignment to `reports/split_manifest.csv`. The model is MobileNetV3-Small. Use `--no-pretrained` to avoid downloading pretrained weights.

## Start the local web app

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The page supports one or multiple image uploads and gives the predicted class, confidence, per-class scores, and a Grad-CAM focus overlay. The focus overlay shows approximate image regions that influenced the classifier; it is not a disease segmentation or a verified bounding box. The classifier was trained with image-level labels, so exact lesion boxes require separately annotated localization data and a detection/segmentation model. An optional farm-context form collects crop stage, symptom spread, affected area, timing, and observed signs to tailor follow-up suggestions; it does not change the image prediction. The form does not ask for a farmer name or address. Each image batch can also be downloaded as CSV.

For CSV batch prediction, prepare a CSV with an `image`, `filename`, or `path` column and a ZIP containing the referenced image files. Keep relative folders in the ZIP when basenames repeat. The downloaded result preserves the CSV columns and adds predicted class, confidence, per-class probabilities, and a row status. The app validates paths and will not extract ZIP entries to the filesystem.

## Dataset audit files

- Original data audit: `reports/dataset_audit/`
- Augmented data audit: `reports/dataset_audit_augmented/`
- Original/augmented comparison and leakage findings: `reports/dataset_audit_comparison.md`
- Reusable read-only audit script: `scripts/audit_dataset.py`

The dataset folders are read in place. The comparison summary is included; detailed inventories and generated training outputs stay local. `.gitignore` excludes model checkpoints, local environments, and generated training manifests.

## Research note

The prediction interface is a research screening aid, not a validated diagnostic. Review class labels and performance with agricultural experts before using predictions outside the project. The current dataset folders use literal labels `Healthy Leaf` and `White spot`, which differ in wording/capitalization from the supplied description; training preserves those literal labels.
