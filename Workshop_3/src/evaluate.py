"""Test-set evaluation for Workshop 3 (guide: Evaluation and Comparison).

- Loads checkpoints/agent_best.pt and evaluates greedily on the test split.
- Scores baselines (random, majority, supervised heuristic) on the same env.
- Prints/saves the comparison table vs Workshop 1/2 (see
  src/compare_placeholders.py for the TODO slots to fill when W1/W2 close).
- Saves trajectory visualizations (3-5 episodes: state, action, reward) and
  learning curves (return vs env steps) for the report.
- Sample-efficiency section is left as data + TODO: RL env-steps are logged;
  supervised sample counts must be filled from W1/W2 when finished.

Usage:
    python -m src.evaluate --checkpoint checkpoints/agent_best.pt --plot
    python -m src.evaluate --episodes 200 --max-steps 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.agent import DQNAgent
from src.baselines import MajorityPolicy, RandomPolicy, SupervisedHeuristicPolicy, evaluate_policy
from src.compare_placeholders import TODO_W1_W2, comparison_table
from src.env import FashionMNISTClassifyEnv


def evaluate_agent(checkpoint: str, n_episodes: int = 200, max_steps: int = 100,
                   reward_mode: str = "plus_minus", data_root: str = "data",
                   seed: int = 1000, device: str = "cpu") -> dict:
    agent = DQNAgent(device=device)
    agent.load(checkpoint)
    agent.eps = 0.0  # purely greedy at test time
    env = FashionMNISTClassifyEnv(split="test", max_steps=max_steps,
                                  reward_mode=reward_mode, data_root=data_root, seed=seed)
    returns, correct, steps = [], 0, 0
    per_class_correct = np.zeros(10, dtype=np.int64)
    per_class_total = np.zeros(10, dtype=np.int64)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, _, truncated, info = env.step(action)
            total += reward
            correct += int(info["correct"])
            steps += 1
            per_class_total[int(info["true_label"])] += 1
            per_class_correct[int(info["true_label"])] += int(info["correct"])
            done = truncated
        returns.append(total)
    arr = np.array(returns, dtype=np.float64)
    return {
        "checkpoint": checkpoint,
        "n_episodes": n_episodes,
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "accuracy": float(correct / max(steps, 1)),
        "f1_macro": None,  # TODO: per-step predictions -> sklearn macro-F1 (needs label/pred dump)
        "per_class_accuracy": {
            str(i): float(per_class_correct[i] / max(per_class_total[i], 1)) for i in range(10)
        },
    }


def collect_trajectories(checkpoint: str, n_trajectories: int = 4, max_steps: int = 12,
                         data_root: str = "data", seed: int = 7,
                         device: str = "cpu") -> list[dict]:
    """Short greedy rollouts for the trajectory figure (state, action, reward)."""
    agent = DQNAgent(device=device)
    agent.load(checkpoint)
    agent.eps = 0.0
    env = FashionMNISTClassifyEnv(split="test", max_steps=max_steps,
                                  reward_mode="plus_minus", data_root=data_root, seed=seed)
    trajs = []
    for ep in range(n_trajectories):
        obs, _ = env.reset(seed=seed + ep)
        frames, actions, rewards, labels, corrects = [obs.copy()], [], [], [], []
        done = False
        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, _, truncated, info = env.step(action)
            frames.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            labels.append(int(info["true_label"]))
            corrects.append(bool(info["correct"]))
            done = truncated
        trajs.append({"frames": np.array(frames), "actions": actions, "rewards": rewards,
                      "labels": labels, "correct": corrects,
                      "return": float(np.sum(rewards))})
    return trajs


def save_trajectory_figure(trajs: list[dict], path: Path, max_cols: int = 12) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.data import CLASS_NAMES

    n = len(trajs)
    cols = min(max_cols, max(len(t["actions"]) for t in trajs))
    fig, axes = plt.subplots(n, cols, figsize=(1.6 * cols, 1.8 * n), squeeze=False)
    for i, traj in enumerate(trajs):
        for j in range(cols):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(traj["actions"]):
                img = traj["frames"][j][0]
                ax.imshow(img, cmap="gray")
                a, r = traj["actions"][j], traj["rewards"][j]
                color = "green" if r > 0 else "red"
                ax.set_title(f"{CLASS_NAMES[a][:8]} {r:+.0f}", fontsize=7, color=color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2)
            else:
                ax.axis("off")
        axes[i][0].set_ylabel(f"ep {i} G={traj['return']:.0f}", fontsize=8)
    fig.suptitle("Sample trajectories — greedy DQN on test (green=+1, red=-1)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] trajectories saved to {path}")
    return path


def save_learning_curves(runs_dir: Path, out_path: Path) -> Path | None:
    """Plot return vs env steps for every config in runs/workshop3/*/train_log.csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as csv_mod

    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for log_path in sorted(runs_dir.glob("*/train_log.csv")):
        steps, rets, vals = [], [], []
        with log_path.open() as fh:
            for row in csv_mod.DictReader(fh):
                try:
                    steps.append(int(row["env_steps"]))
                    rets.append(float(row["return"]))
                    if row.get("val_return_mean"):
                        vals.append(float(row["val_return_mean"]))
                except (ValueError, KeyError):
                    continue
        if steps:
            found = True
            ax.plot(steps, rets, alpha=0.35, label=f"{log_path.parent.name} train")
    # Baselines: random-policy expected return for ±1 rewards at 10% accuracy.
    max_steps_ref = 100
    ax.axhline(-80, color="gray", linestyle="--", label="random baseline (~-80)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title("Learning curves — return vs environment steps")
    ax.legend(fontsize=8)
    # TODO(sample-efficiency): overlay supervised W1/W2 sample counts here once
    # final (RL env-steps vs labeled examples are different units; report both).
    if not found:
        print("[evaluate] no train logs found; skipping learning-curve plot")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] learning curves saved to {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Workshop 3 DQN agent")
    p.add_argument("--checkpoint", type=str, default="checkpoints/agent_best.pt")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--reward-mode", type=str, default="plus_minus")
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--runs-dir", type=str, default="runs/workshop3")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--plot", action="store_true", help="save trajectory + learning-curve figures")
    p.add_argument("--skip-supervised-baseline", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto" and device == "auto":
        device = "cpu"

    rl = evaluate_agent(args.checkpoint, args.episodes, args.max_steps,
                        args.reward_mode, args.data_root, device=device)
    print(f"\n[evaluate] DQN {args.checkpoint}: "
          f"mean_return={rl['mean_return']:.2f}±{rl['std_return']:.2f} "
          f"[{rl['min_return']:.0f},{rl['max_return']:.0f}] accuracy={rl['accuracy']:.4f}")

    baselines = [
        evaluate_policy(RandomPolicy(seed=0), "test", 50, args.max_steps, args.reward_mode, args.data_root),
        evaluate_policy(MajorityPolicy(args.data_root), "test", 50, args.max_steps, args.reward_mode, args.data_root),
    ]
    if not args.skip_supervised_baseline:
        try:
            baselines.append(evaluate_policy(
                SupervisedHeuristicPolicy(device=device), "test", 50,
                args.max_steps, args.reward_mode, args.data_root))
        except Exception as exc:
            print(f"[evaluate] supervised heuristic skipped: {exc}")
    for b in baselines:
        print(f"[evaluate] baseline {b['policy']:<22} return={b['mean_return']:.2f}±{b['std_return']:.2f} "
              f"acc={b['accuracy']:.4f}")

    table = comparison_table(rl)
    print("\n" + table)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rl_agent": rl, "baselines": baselines,
               "comparison_table_markdown": table, "pending_todos": TODO_W1_W2,
               "sample_efficiency_note": (
                   "TODO: compare total RL env-steps (see runs/workshop3/*/summary.json) "
                   "against supervised labeled-example counts from W1/W2 once closed.")}
    out_path = runs_dir / "test_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[evaluate] metrics written to {out_path}")

    if args.plot:
        trajs = collect_trajectories(args.checkpoint, n_trajectories=4,
                                     max_steps=min(args.max_steps, 12),
                                     data_root=args.data_root, device=device)
        save_trajectory_figure(trajs, runs_dir / "trajectories.png")
        save_learning_curves(runs_dir, runs_dir / "learning_curves.png")


if __name__ == "__main__":
    main()
