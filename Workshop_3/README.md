# Workshop 3 — Reinforcement Learning (DQN) for Fashion-MNIST

**Team 3** — Computer vision and object recognition.
Jhojan Stiven Aragón Ramírez (20221020060) · Santiago Reyes Gómez (20221020068) · Juan Andrés Jiménez Palomino (20221020087)

RL alternative to the supervised models of Workshop 1/2: a DQN agent learns to
classify Fashion-MNIST images formulated as a sequential decision process
(multi-step episodes of `N=100` i.i.d. images, action = predicted class,
reward `+1/-1`). Same 70/15/15 split (seed 42) and normalization as Workshop 2,
so metrics stay comparable.

> El documento de formulacion MDP vive en `docs/mdp_spec.md`; el reporte
> `group3_workshop3.pdf` se genera desde `docs/group3_workshop3.tex`
> (`pdflatex` dos pasadas, ejecutado desde `docs/`).

## Results (placeholder — updated after final runs + W1/W2 closure)

| Model | Test accuracy | Macro F1 | Notes |
|---|---|---|---|
| HistGradientBoosting (W1 best) | 87.48% | 0.8740 | TODO(W1): confirmar + resto de modelos |
| SmallCNN (W2) | 93.05% | 0.9302 | a confirmar con `runs/test_metrics.json` |
| ResNet-18 (W2) | 94.94% | 0.9494 | a confirmar con `runs/resnet18/test_metrics.json` |
| **DQN (W3, this work)** | TODO | TODO | return-derived; ver `runs/workshop3/test_metrics.json` |

Los unicos valores a completar despues estan centralizados en
`src/compare_placeholders.py` (constantes + TODOs).

## Structure

```
Workshop_3/
├── src/
│   ├── data.py                 # wrapper del pipeline W2 (splits 70/15/15, stats)
│   ├── env.py                  # FashionMNISTClassifyEnv (gymnasium) + validate_random_policy
│   ├── agent.py                # QNetwork + ReplayBuffer + DQNAgent (DQN)
│   ├── baselines.py            # random, majority, supervised-heuristic (ckpt W2)
│   ├── train.py                # entrenamiento + busqueda de 3 hiperparametros
│   ├── evaluate.py             # test, tabla comparativa, trayectorias, curvas
│   └── compare_placeholders.py # TODOs W1/W2 — unico archivo a completar despues
├── docs/mdp_spec.md            # formulacion MDP (6 puntos de la guia)
├── checkpoints/agent_best.pt   # mejor checkpoint (mejor val-return suavizado)
└── runs/workshop3/             # train_log.csv, hparam_results.json, test_metrics.json, PNGs, tensorboard/
```

## Setup

```bash
cd Workshop_3
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

> Sin GPU NVIDIA (ej. AMD/Intel/CPU): instala primero el torch de CPU para
> evitar la descarga pesada con CUDA, y luego el resto sin dependencias:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install gymnasium scikit-learn matplotlib tqdm tensorboard
> pip install -e . --no-deps
> ```

Requiere Python ≥ 3.10, PyTorch ≥ 2.0. Los datos se reutilizan desde
`../Workshop_1/code/data` (via Workshop 2) o se descargan con torchvision en `data/raw`.

## Quickstart

```bash
# 1. Validar el entorno ANTES de entrenar (requisito de la guia: 100 episodios random)
python -m src.train --validate-env

# 2. Smoke test de plumbing (5 episodios, CPU)
python -m src.train --config b --smoke-test

# 3. Entrenar una configuracion (default: b, 300 episodios)
python -m src.train --config b --episodes 300

# 4. Busqueda completa de hiperparametros (a, b, c)
python -m src.train --all-configs --episodes 300

# 5. Evaluar el mejor checkpoint en test + baselines + tabla W1/W2
python -m src.evaluate --checkpoint checkpoints/agent_best.pt --plot
```

## Protocol

- **Episodio:** `max_steps=100` imagenes i.i.d.; `truncated=True` al final.
- **Reward:** `+1` acierto / `-1` fallo (`--reward-mode zero_one|sparse` alternos).
- **DQN:** replay 50k, warmup 1000 pasos, batch 64, target-sync 500 updates,
  Huber loss, grad-clip 10, double-DQN, Adam + weight decay, epsilon 1.0→0.05.
- **Configs:** `a {lr 1e-4, γ 0.90, decay 0.990, wd 0}` ·
  `b {lr 5e-4, γ 0.95, decay 0.995, wd 1e-5}` ·
  `c {lr 1e-3, γ 0.99, decay 0.999, wd 1e-4}`.
- **Seleccion:** mejor retorno de validacion suavizado (media movil de 3) sin
  exploracion, cada 10 episodios sobre 20 episodios de `val`.
- **Reproducibilidad:** `--seed 42`; splits cacheados; config guardada en cada
  checkpoint y en `runs/workshop3/*/summary.json`.

## Deliverables (mapeo con la guia)

| Requisito | Donde |
|---|---|
| MDP spec | `docs/mdp_spec.md` |
| `src/env.py` + diagnostico 100 episodios | `src/env.py`, `python -m src.train --validate-env` |
| `src/agent.py` (DQN) | `src/agent.py` |
| `src/baselines.py` | `src/baselines.py` |
| `src/train.py` + 3 configs | `src/train.py`, `runs/workshop3/hparam_results.json` |
| `src/evaluate.py` + test | `src/evaluate.py`, `runs/workshop3/test_metrics.json` |
| `checkpoints/agent_best.pt` | `checkpoints/` |
| `runs/workshop3/` logs | CSV + TensorBoard + PNGs |
| Comparacion W1/W2 (pendiente cierre) | `src/compare_placeholders.py` |
| Reporte PDF | `docs/group3_workshop3.tex` -> `group3_workshop3.pdf` |
