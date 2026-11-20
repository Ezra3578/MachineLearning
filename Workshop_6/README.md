# Workshop 6 — Medical Domain Adaptation: Fashion-MNIST CNNs to Chest X-rays (PneumoniaMNIST)

**Team 3** — Computer vision and object recognition.
Jhojan Stiven Aragón Ramírez (20221020060) · Santiago Reyes Gómez (20221020068) · Juan Andrés Jiménez Palomino (20221020087)

We adapt the Workshop 2 SmallCNN from Fashion-MNIST to pediatric pneumonia
detection on chest radiographs. The report compares transfer strategies,
measures sensitivity, specificity, PPV and NPV at several thresholds, and
examines where Grad-CAM explanations fail.

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
In the frozen run, batch-normalization running statistics still update on the
medical images.

## Structure

```
Workshop_6/
├── src/
│   ├── medical_preprocessing.py  # CLAHE, normalization stats, imbalance report
│   ├── transfer.py               # SmallCNN + frozen/finetune/scratch transfer modes
│   ├── medical_metrics.py        # sens/spec/PPV/NPV, threshold sweep, Youden J
│   ├── train_medical.py          # training loop (BCE, Adam, cosine, early stop on val AUC)
│   ├── interpretability.py       # Grad-CAM figures, case reports, regional attribution
│   ├── evaluate_medical.py       # test evaluation, ROC figure, ablation table
│   ├── bootstrap_ci.py           # bootstrap AUC intervals and paired contrasts
│   ├── figures.py                # dataset overview, training curves, confusion matrices
│   ├── plotting.py               # shared RGB figure saving (pdfTeX-safe PNGs)
│   └── device.py                 # auto device selection (CUDA -> MPS -> DirectML -> CPU)
├── checkpoints/                  # <run>_best.pt per configuration
├── runs/workshop6/               # train logs, test metrics, figures, reports
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
python -m src.train_medical --name finetune_clahe_weights --mode finetune --preprocess clahe --class-weights balanced
python -m src.train_medical --name full_recipe       --mode finetune --preprocess clahe \
    --class-weights balanced --augment medical

# 2. Clinical evaluation of every run (threshold tables, ROC, ablation table)
python -m src.evaluate_medical

# 3. Grad-CAM interpretability on the best checkpoint (+ regional attribution)
python -m src.interpretability --checkpoint checkpoints/finetune_weights_best.pt

# 4. Dataset overview, training curves and confusion matrices
python -m src.figures

# 5. Confidence intervals and paired ablation contrasts
python -m src.bootstrap_ci
```

All runs use seed 42 and `--device auto` (CUDA, then Apple MPS, then AMD
DirectML, then CPU); results land in `runs/workshop6/`
(`ablation_results.json`, `test_metrics_<run>.json`, `roc_curves.png`,
`gradcam_analysis.png`, `interpretability_report.json`, `dataset_overview.png`,
`training_curves.png`, `confusion_matrices.png`).

The evaluation selects the Youden threshold on validation and reports its
metrics on the test split alongside the fixed thresholds. To evaluate the
submitted models, start at step 2 with the existing checkpoints and logs.
Use `--device cpu` to match the report's execution device.

## Report

[Final medical adaptation report](group3_medical_final_report.pdf) ·
[`v1.0-medical-final` on GitHub](https://github.com/Ezra3578/MachineLearning/tree/v1.0-medical-final)
