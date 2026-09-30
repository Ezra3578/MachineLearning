# Workshop 2 — Deep Learning (CNNs) for Fashion-MNIST

**Team 3** — Computer vision and object recognition.

This workshop implements, trains, evaluates and checkpoints a deep convolutional neural
network for Fashion-MNIST image classification, comparing against the Workshop 1
traditional baselines (comparison table reserved for future work).

## Structure (professor's required layout)

```
Workshop_2/
├── src/
│   ├── data.py        # IDX loading, splits (70/15/15, seed 42), transforms, loaders
│   ├── model.py       # SmallCNN (from scratch) + ResNet-18 (transfer learning)
│   ├── train.py       # training loop: loss, optimizer, LR schedule, regularization
│   ├── evaluate.py    # test-set metrics vs Workshop 1 protocol
│   └── test_model.py  # loads checkpoints/model_best.pt and verifies inference
├── notebooks/         # experimentation notebooks
├── checkpoints/       # saved model weights (model_best.pt)
├── runs/              # training logs (CSV + TensorBoard)
└── pyproject.toml     # dependencies
```

## Setup

```bash
cd Workshop_2
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.0, torchvision ≥ 0.15.

## Quickstart

```bash
# 1. Train the from-scratch CNN (default: 30 epochs, Adam, cosine annealing)
python -m src.train --model smallcnn --epochs 30

# 2. Train the ResNet-18 transfer-learning variant (two-stage fine-tuning)
python -m src.train --model resnet18 --epochs 20 --freeze-epochs 5 --lr 3e-4

# 3. Evaluate the best checkpoint on the test split
python -m src.evaluate --checkpoint checkpoints/model_best.pt

# 4. Verify the checkpoint loads and produces predictions
python -m src.test_model --checkpoint checkpoints/model_best.pt
```

## Protocol

- **Data.** Original Fashion-MNIST IDX files, read from `../Workshop_1/code/data` when
  present (no re-download); otherwise downloaded via torchvision into `data/raw`.
  Stratified **70 / 15 / 15** split over the 70,000 images (seed 42) — the same protocol
  as the Workshop 1 team implementation, so metrics remain comparable.
- **Normalization.** Global Fashion-MNIST statistics `mean = 0.2860`, `std = 0.3530`
  (from the Workshop 1 EDA: 72.94/255 and 90.02/255).
- **Augmentation (train only).** Random horizontal flip (p = 0.5) + random translation
  (±10%), mirroring Workshop 1; optional mixup (`--mixup 0.2`).
- **Model.**
  - `smallcnn`: 3 conv blocks (32→64→128) with BatchNorm, max-pooling, global average
    pooling and dropout 0.5 — ≈ 140k parameters, Kaiming initialization.
  - `resnet18`: ImageNet-pretrained torchvision ResNet-18, input adapted to 224×224 RGB;
    two-stage fine-tuning (`--freeze-epochs`), as suggested in the workshop guide.
- **Training.** Cross-entropy loss (optional label smoothing), Adam or SGD + momentum,
  cosine annealing schedule, weight decay, early stopping on validation macro-F1.
- **Checkpointing.** `checkpoints/model_best.pt` = best validation macro-F1, storing
  `state_dict`, class names, normalization stats and the training configuration.
- **Logging.** `runs/train_log.csv` (every epoch) + TensorBoard in `runs/tensorboard/`;
  optional learning-curve notebook in `notebooks/`.

## Reproducibility

All runs are seeded (`--seed 42`), splits are cached in `data/splits.json`, and the full
training configuration is stored inside every checkpoint. Reported metrics can be
regenerated with:

```bash
python -m src.train --model smallcnn --epochs 30 --seed 42
python -m src.evaluate --checkpoint checkpoints/model_best.pt
```

## Deliverables (workshop mapping)

| Requirement | Where |
|---|---|
| Architecture definition + parameter count | `src/model.py` (printed by `python -m src.model`) |
| Training script (loss, optimizer, schedule, regularization) | `src/train.py` |
| Test-set evaluation + metrics | `src/evaluate.py`, `runs/test_metrics.json` |
| Checkpoint + inference verification | `checkpoints/model_best.pt`, `src/test_model.py` |
| Training logs & learning curves | `runs/train_log.csv`, `runs/tensorboard/` |
| Report (PDF) | `Documentation/` (to be added) |
