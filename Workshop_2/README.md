# Workshop 2 — Deep Learning (CNNs) for Fashion-MNIST

**Team 3** — Computer vision and object recognition.
Jhojan Stiven Aragón Ramírez (20221020060) · Santiago Reyes Gómez (20221020068) · Juan Andrés Jiménez Palomino (20221020087)

This workshop implements, trains, evaluates and checkpoints deep convolutional neural
networks for Fashion-MNIST image classification, and compares them against the Workshop 1
traditional baselines on the same 70/15/15 split.

## Results

| Model | Test accuracy | Macro F1 | ROC-AUC (OvR) |
|---|---|---|---|
| HistGradientBoosting (W1 best baseline) | 87.48% | 0.8740 | — |
| SmallCNN (from scratch, 30 epochs) | 93.05% | 0.9302 | 0.9961 |
| **ResNet-18 (ImageNet transfer, 20 epochs)** | **94.94%** | **0.9494** | **0.9979** |

The complete comparison (all three W1 baselines versus both deep models) is included in the
Workshop 2 report of the LaTeX repository (`tex-team3-machine-learning` → `Workshop_2/main.pdf`).

## Structure 

```
Workshop_2/
├── src/
│   ├── data.py        # IDX loading, splits (70/15/15, seed 42), transforms, loaders
│   ├── model.py       # SmallCNN (from scratch) + ResNet-18 (transfer learning)
│   ├── train.py       # training loop: loss, optimizer, LR schedule, regularization
│   ├── evaluate.py    # test-set metrics vs Workshop 1 protocol
│   ├── gradcam.py     # Grad-CAM visual analysis of a checkpoint
│   └── test_model.py  # loads a checkpoint and verifies inference
├── notebooks/
│   ├── Training_Pipeline.ipynb   # documented end-to-end training flow (executed)
│   └── GradCAM_Analysis.ipynb    # SmallCNN Grad-CAM analysis (executed)
├── checkpoints/
│   ├── model_best.pt             # SmallCNN best (93.05% test)
│   └── resnet18/model_best.pt    # ResNet-18 best (94.94% test)
├── runs/
│   ├── train_log.csv, test_metrics.json, learning_curves.png, ...
│   └── resnet18/                 # ResNet-18 logs, metrics, curves, Grad-CAM
└── pyproject.toml     # dependencies
```

## Setup

```bash
cd Workshop_2
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0, torchvision ≥ 0.15.

### Optional GPU on Windows (AMD DirectML)

Without an NVIDIA GPU, a separate environment with `torch-directml` trains the ResNet-18 on
any DirectX 12 GPU (it pins `torch==2.4.1`):

```powershell
uv venv .venv-gpu --python 3.11
uv pip install --python .venv-gpu\Scripts\python.exe torch-directml -e .
.\.venv-gpu\Scripts\python.exe -m src.train --model resnet18 --epochs 20 --freeze-epochs 5 --lr 3e-4 --batch-size 32
```

`pick_device` selects the accelerator automatically (CUDA → MPS → DirectML → CPU).

## Quickstart

```bash
# 1. Train the from-scratch CNN (default: 30 epochs, Adam, cosine annealing)
python -m src.train --model smallcnn --epochs 30

# 2. Train the ResNet-18 transfer variant (two-stage fine-tuning; batch 32 fits a 4 GB GPU)
python -m src.train --model resnet18 --epochs 20 --freeze-epochs 5 --lr 3e-4 --batch-size 32 \
    --checkpoint-dir checkpoints/resnet18 --runs-dir runs/resnet18

# 3. Evaluate a checkpoint on the test split
python -m src.evaluate --checkpoint checkpoints/model_best.pt --plot
python -m src.evaluate --checkpoint checkpoints/resnet18/model_best.pt --runs-dir runs/resnet18 --plot

# 4. Verify a checkpoint loads and produces predictions
python -m src.test_model --checkpoint checkpoints/model_best.pt
```

> Use `--checkpoint-dir` / `--runs-dir` subfolders per model so runs do not overwrite each
> other. `checkpoints/model_best.pt` is reserved for the final selected model (currently the
> SmallCNN; the ResNet-18 wins on test accuracy).

## Protocol

- **Data.** Original Fashion-MNIST IDX files, read from `../Workshop_1/code/data` when
  present (no re-download); otherwise downloaded via torchvision into `data/raw`.
  Stratified **70 / 15 / 15** split over the 70,000 images (seed 42) — the same protocol
  as the Workshop 1 team implementation, so metrics remain comparable.
- **Normalization.** Global Fashion-MNIST statistics `mean = 0.2860`, `std = 0.3530`
  (from the Workshop 1 EDA: 72.94/255 and 90.02/255); the ResNet-18 uses ImageNet statistics
  after replicating the grayscale channel to RGB and resizing to 224×224.
- **Augmentation (train only).** Random horizontal flip (p = 0.5) + random translation
  (±10%), mirroring Workshop 1; optional mixup (`--mixup 0.2`).
- **Model.**
  - `smallcnn`: 3 conv blocks (32→64→128) with BatchNorm, max-pooling, global average
    pooling and dropout 0.5 — ≈ 140k parameters, Kaiming initialization.
  - `resnet18`: ImageNet-pretrained torchvision ResNet-18, two-stage fine-tuning
    (`--freeze-epochs`), as suggested in the workshop guide; batch 32 on a 4 GB GPU.
- **Training.** Cross-entropy loss (optional label smoothing), Adam or SGD + momentum,
  cosine annealing schedule, weight decay, early stopping on validation macro-F1.
- **Checkpointing.** `model_best.pt` per run = best validation macro-F1, storing
  `state_dict`, class names, normalization stats and the training configuration.
- **Logging.** `train_log.csv` (every epoch) + TensorBoard in `tensorboard/`; the
  `Training_Pipeline.ipynb` notebook walks through the same flow end to end.

## Reproducibility

All runs are seeded (`--seed 42`), splits are cached in `data/splits.json`, and the full
training configuration is stored inside every checkpoint. Reported metrics can be
regenerated with:

```bash
python -m src.train --model smallcnn --epochs 30 --seed 42
python -m src.train --model resnet18 --epochs 20 --freeze-epochs 5 --lr 3e-4 --batch-size 32 --seed 42 \
    --checkpoint-dir checkpoints/resnet18 --runs-dir runs/resnet18
python -m src.evaluate --checkpoint checkpoints/model_best.pt
```

## Deliverables (workshop mapping)

| Requirement | Where |
|---|---|
| Architecture definition + parameter count | `src/model.py` (printed by `python -m src.model`) |
| Training script (loss, optimizer, schedule, regularization) | `src/train.py` |
| Test-set evaluation + metrics | `src/evaluate.py`, `runs/test_metrics.json` |
| Checkpoint + inference verification | `checkpoints/`, `src/test_model.py` |
| Training logs & learning curves | `runs/`, `runs/*/tensorboard/` |
| Qualitative analysis | `src/gradcam.py`, `notebooks/GradCAM_Analysis.ipynb` |
| Documented training flow | `notebooks/Training_Pipeline.ipynb` |
| Report (PDF) |  `Workshop2.pdf` |
