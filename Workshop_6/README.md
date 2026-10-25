# Workshop 6 — Medical Domain Adaptation: Fashion-MNIST CNNs to Chest X-rays (PneumoniaMNIST)

**Team 3** — Computer vision and object recognition.
Jhojan Stiven Aragón Ramírez (20221020060) · Santiago Reyes Gómez (20221020068) · Juan Andrés Jiménez Palomino (20221020087)

Final workshop: the SmallCNN trained on Fashion-MNIST in Workshop 2 is adapted to
pediatric pneumonia detection on chest radiographs (PneumoniaMNIST), with clinical
metrics (sensitivity/specificity/PPV/NPV at several thresholds), Grad-CAM
interpretability, an ablation study of the transfer components, and a clinical
deployment assessment.

## Medical dataset

**PneumoniaMNIST** (MedMNIST v2, Yang et al., Scientific Data 2021): 5,856
pediatric chest radiographs from Guangzhou Women and Children's Medical Center,
28×28 grayscale, binary labels (0 = normal, 1 = pneumonia).

- Project page / loader: https://medmnist.com
- Data + official benchmark numbers: https://zenodo.org/records/6496656
- Original X-ray source (Kermany et al., Cell 2018):
  https://data.mendeley.com/datasets/rscbjbr9sj/3

The loader downloads the data automatically into `data/` on first use.

## Transfer strategy

| Mode | What transfers | What is trained |
|---|---|---|
| `scratch` | nothing (ablation baseline) | everything |
| `frozen` | conv backbone from `../Workshop_2/checkpoints/model_best.pt` | binary head only |
| `finetune` | conv backbone (initialized from W2) | everything |

Medical adaptations under study: CLAHE contrast enhancement, class weighting for
the 26/74 normal/pneumonia imbalance, and modality-appropriate augmentation
(small rotations/translations, brightness jitter; no horizontal flips).

## Structure

```
Workshop_6/
├── src/
│   ├── medical_preprocessing.py  # CLAHE, normalization stats, imbalance report
│   ├── transfer.py               # SmallCNN + frozen/finetune/scratch transfer modes
│   ├── medical_metrics.py        # sens/spec/PPV/NPV, threshold sweep, Youden J
│   ├── train_medical.py          # training loop (BCE, Adam, cosine, early stop on val AUC)
│   ├── interpretability.py       # Grad-CAM figures + case reports
│   └── evaluate_medical.py       # test evaluation, ROC figure, ablation table
├── checkpoints/                  # <run>_best.pt per configuration
├── runs/workshop6/               # train logs, test metrics, figures, reports
├── docs/                         # group3_medical_final_report.tex / .pdf
└── data/                         # PneumoniaMNIST (auto-downloaded, gitignored)
```

## Setup

```bash
cd Workshop_6
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install medmnist scikit-learn opencv-python-headless matplotlib tqdm
```

## Reproduction

```bash
# 0. Data balance + normalization stats (sanity check)
python -m src.medical_preprocessing

# 1. Ablation runs (each trains with early stopping on val AUC)
python -m src.train_medical --name scratch           --mode scratch
python -m src.train_medical --name frozen            --mode frozen
python -m src.train_medical --name finetune          --mode finetune
python -m src.train_medical --name finetune_clahe    --mode finetune --preprocess clahe
python -m src.train_medical --name finetune_weights  --mode finetune --class-weights balanced
python -m src.train_medical --name full_recipe       --mode finetune --preprocess clahe \
    --class-weights balanced --augment medical

# 2. Clinical evaluation of every run (threshold tables, ROC, ablation table)
python -m src.evaluate_medical

# 3. Grad-CAM interpretability on the best checkpoint
python -m src.interpretability --checkpoint checkpoints/full_recipe_best.pt
```

All runs use seed 42; results land in `runs/workshop6/`
(`ablation_results.json`, `test_metrics_<run>.json`, `roc_curves.png`,
`gradcam_analysis.png`, `interpretability_report.json`).

## Report

`docs/group3_medical_final_report.pdf` (from `docs/group3_medical_final_report.tex`,
`pdflatex` twice from `docs/`).
