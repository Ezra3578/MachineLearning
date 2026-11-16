# Workshop 3. Reinforcement learning for Fashion-MNIST

Team 3 (computer vision and object recognition)

Jhojan Stiven Aragón Ramírez (20221020060), Santiago Reyes Gómez (20221020098), and Juan Andrés Jiménez Palomino (20221020087)

The Double DQN agent predicts the class of one Fashion-MNIST image at each step. An episode contains `N=100` independently sampled images. The agent receives `+1/-1` for a correct or incorrect prediction. The environment reuses the Workshop 2 normalization and 70/15/15 split (seed 42).

See the [MDP specification](docs/mdp_spec.md) for the environment and reward definitions.

## Results

| Model | Test accuracy | Macro F1 | Notes |
|---|---|---|---|
| HistGradientBoosting (W1 best) | 87.48% | 0.8740 | best W1 baseline |
| SmallCNN (W2) | 93.05% | 0.9302 | `../Workshop_2/runs/test_metrics.json` |
| ResNet-18 (W2) | 94.94% | 0.9494 | `../Workshop_2/runs/resnet18/test_metrics.json` |
| Double DQN (W3) | 85.20% | n/a | Accuracy derived from rewards over 200 test episodes in `runs/workshop3/test_metrics.json` |

## Structure

```
Workshop_3/
├── src/
│   ├── data.py                 # wrapper of the W2 pipeline (70/15/15 splits, stats)
│   ├── env.py                  # FashionMNISTClassifyEnv (gymnasium) + validate_random_policy
│   ├── agent.py                # QNetwork + ReplayBuffer + DQNAgent (DQN)
│   ├── baselines.py            # random, majority, supervised-heuristic (W2 ckpt)
│   ├── train.py                # training + 3-hyperparameter search
│   ├── evaluate.py             # test, comparison table, trajectories, curves
│   └── compare_placeholders.py # W1/W2 values used by the comparison table
├── docs/mdp_spec.md            # MDP formulation (6 items of the guide)
├── checkpoints/agent_best.pt   # best checkpoint (best smoothed val-return)
├── group3_workshop3.pdf        # compiled report
└── runs/workshop3/             # train_log.csv, hparam_results.json, test_metrics.json, PNGs, tensorboard/
```

## Setup

```bash
cd Workshop_3
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

> For a CPU-only setup, install the CPU builds of PyTorch and torchvision first.
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install gymnasium scikit-learn matplotlib tqdm tensorboard
> pip install -e . --no-deps
> ```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. The Workshop 2 loader reads the dataset from `../Workshop_1/code/data` when available. Otherwise, torchvision downloads it into `data/raw`.

## Quickstart

```bash
# 1. Validate the environment BEFORE training (guide requirement: 100 random episodes)
python -m src.train --validate-env

# 2. Plumbing smoke test (5 episodes, CPU)
python -m src.train --config b --smoke-test

# 3. Train one configuration (default: b, 300 episodes)
python -m src.train --config b --episodes 300

# 4. Full hyperparameter search (a, b, c)
python -m src.train --all-configs --episodes 300

# 5. Evaluate the best checkpoint on test + baselines + W1/W2 table
python -m src.evaluate --checkpoint checkpoints/agent_best.pt --plot
```

## Protocol

- **Episode:** `max_steps=100` i.i.d. images; `truncated=True` at the end.
- **Reward:** `+1` hit / `-1` miss (`--reward-mode zero_one|sparse` alternatives).
- **DQN:** replay 50k, warmup 1000 steps, batch 64, target-sync 500 updates,
  Huber loss, grad-clip 10, double-DQN, Adam + weight decay, epsilon 1.0→0.05.
- **Configs:** `a {lr 1e-4, γ 0.90, decay 0.990, wd 0}` ·
  `b {lr 5e-4, γ 0.95, decay 0.995, wd 1e-5}` ·
  `c {lr 1e-3, γ 0.99, decay 0.999, wd 1e-4}`.
- **Selection:** best smoothed validation return (moving average of 3) without
  exploration, every 10 episodes over 20 episodes of `val`.
- **Reproducibility:** `--seed 42`; cached splits; config stored in every
  checkpoint and in `runs/workshop3/*/summary.json`.

## Deliverables (mapped to the guide)

| Requirement | Where |
|---|---|
| MDP spec | `docs/mdp_spec.md` |
| `src/env.py` + 100-episode diagnostics | `src/env.py`, `python -m src.train --validate-env` |
| `src/agent.py` (DQN) | `src/agent.py` |
| `src/baselines.py` | `src/baselines.py` |
| `src/train.py` + 3 configs | `src/train.py`, `runs/workshop3/hparam_results.json` |
| `src/evaluate.py` + test | `src/evaluate.py`, `runs/workshop3/test_metrics.json` |
| `checkpoints/agent_best.pt` | `checkpoints/` |
| `runs/workshop3/` logs | CSV + TensorBoard + PNGs |
| W1/W2 comparison | `src/compare_placeholders.py`, `runs/workshop3/test_metrics.json` |
| Report (PDF) | [Workshop 3 report](group3_workshop3.pdf) |
