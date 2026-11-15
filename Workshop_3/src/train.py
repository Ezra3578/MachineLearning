"""DQN training script for Workshop 3.

Covers the guide's Agent Training + Hyperparameter Search requirements:
- epsilon-greedy exploration with decay (logged per episode)
- experience replay, target network, gradient clipping (see src/agent.py)
- periodic validation without exploration + smoothed validation return
- best checkpoint -> checkpoints/agent_best.pt
- 3 hyperparameter configurations (--config a|b|c or --all-configs)

Usage:
    python -m src.train --validate-env            # validate env first (guide requirement)
    python -m src.train --config b --episodes 300
    python -m src.train --all-configs --episodes 300
    python -m src.train --config b --smoke-test  # tiny CPU run to verify plumbing
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.agent import DQNAgent, ReplayBuffer, count_parameters
from src.env import FashionMNISTClassifyEnv, validate_random_policy

# --------------------------------------------------------------------------- #
# Hyperparameter search space (guide: 3+ configurations varying key params)
# --------------------------------------------------------------------------- #
CONFIGS: dict[str, dict] = {
    "a": {"lr": 1e-4, "gamma": 0.90, "eps_decay": 0.990, "weight_decay": 0.0,
          "desc": "low LR, low gamma, fast exploration decay, no regularization"},
    "b": {"lr": 5e-4, "gamma": 0.95, "eps_decay": 0.995, "weight_decay": 1e-5,
          "desc": "mid LR/gamma/decay, light L2 (default)"},
    "c": {"lr": 1e-3, "gamma": 0.99, "eps_decay": 0.999, "weight_decay": 1e-4,
          "desc": "high LR/gamma, slow decay, stronger L2"},
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_validation(agent: DQNAgent, split: str, n_episodes: int,
                   max_steps: int, reward_mode: str, data_root: str, seed: int) -> dict:
    """Evaluate greedily (no exploration) on a held-out split."""
    env = FashionMNISTClassifyEnv(split=split, max_steps=max_steps,
                                  reward_mode=reward_mode, data_root=data_root, seed=seed)
    returns = []
    correct, steps = 0, 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action = agent.select_action(obs, explore=False)
            obs, reward, _, truncated, info = env.step(action)
            total += reward
            correct += int(info["correct"])
            steps += 1
            done = truncated
        returns.append(total)
    arr = np.array(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "accuracy": float(correct / max(steps, 1)),
        "n_episodes": n_episodes,
    }


def train_config(
    name: str,
    episodes: int = 300,
    max_steps: int = 100,
    reward_mode: str = "plus_minus",
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    warmup_steps: int = 1000,
    updates_per_step: int = 1,
    val_every: int = 10,
    val_episodes: int = 20,
    seed: int = 42,
    data_root: str = "data",
    checkpoint_dir: str = "checkpoints",
    runs_dir: str = "runs/workshop3",
    device: str = "cpu",
) -> dict:
    cfg = CONFIGS[name]
    seed_everything(seed)
    dev = torch.device(device)
    print(f"[train:{name}] {cfg['desc']}")
    print(f"[train:{name}] device={dev} episodes={episodes} max_steps={max_steps} seed={seed}")

    train_env = FashionMNISTClassifyEnv(split="train", max_steps=max_steps,
                                        reward_mode=reward_mode, data_root=data_root, seed=seed)
    agent = DQNAgent(lr=cfg["lr"], gamma=cfg["gamma"], eps_decay=cfg["eps_decay"],
                     weight_decay=cfg["weight_decay"], device=str(dev))
    trainable, total = count_parameters(agent.q_net)
    print(f"[train:{name}] QNetwork params: trainable={trainable:,} total={total:,}")
    buffer = ReplayBuffer(capacity=buffer_capacity)

    run_dir = Path(runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    except Exception:
        writer = None

    log_path = run_dir / "train_log.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "return", "length", "epsilon", "loss_mean",
                         "val_return_mean", "val_accuracy", "env_steps"])

    best_val, best_ep = -1e18, -1
    val_history: list[float] = []
    env_steps, total_loss, loss_count = 0, 0.0, 0

    for ep in range(1, episodes + 1):
        obs, _ = train_env.reset(seed=seed + ep)
        done, ep_return, length = False, 0.0, 0
        while not done:
            action = agent.select_action(obs, explore=True)
            next_obs, reward, _, truncated, _ = train_env.step(action)
            buffer.push(obs, action, reward, next_obs, truncated)
            obs = next_obs
            ep_return += reward
            length += 1
            env_steps += 1
            done = truncated
            if len(buffer) >= max(warmup_steps, batch_size):
                for _ in range(updates_per_step):
                    total_loss += agent.update(buffer.sample(batch_size))
                    loss_count += 1
        agent.decay_epsilon()

        val_mean, val_acc = float("nan"), float("nan")
        if ep % val_every == 0 or ep == episodes:
            val = run_validation(agent, "val", val_episodes, max_steps, reward_mode, data_root, seed)
            val_history.append(val["mean_return"])
            # Smoothed validation return (mean of last 3) for model selection.
            smoothed = float(np.mean(val_history[-3:]))
            val_mean, val_acc = val["mean_return"], val["accuracy"]
            if writer is not None:
                writer.add_scalar("return/val", val_mean, ep)
                writer.add_scalar("return/val_smoothed3", smoothed, ep)
                writer.add_scalar("accuracy/val", val_acc, ep)
            if smoothed > best_val:
                best_val, best_ep = smoothed, ep
                agent.save(ckpt_dir / f"agent_best_{name}.pt")
        loss_mean = total_loss / max(loss_count, 1)
        log_writer.writerow([ep, f"{ep_return:.2f}", length, f"{agent.eps:.4f}",
                             f"{loss_mean:.4f}", f"{val_mean:.2f}" if val_mean == val_mean else "",
                             f"{val_acc:.4f}" if val_acc == val_acc else "", env_steps])
        log_file.flush()
        if writer is not None:
            writer.add_scalar("return/train_episode", ep_return, ep)
            writer.add_scalar("exploration/epsilon", agent.eps, ep)
            writer.add_scalar("loss/mean", loss_mean, ep)
        if ep % max(val_every, 1) == 0 or ep == 1:
            print(f"[train:{name}] ep {ep:4d}/{episodes} | return {ep_return:7.2f} | "
                  f"eps {agent.eps:.3f} | loss {loss_mean:.4f} | "
                  f"val {val_mean:.2f} acc {val_acc:.3f} | steps {env_steps}")

    agent.save(ckpt_dir / f"agent_last_{name}.pt")
    if writer is not None:
        writer.close()
    log_file.close()
    summary = {"config": name, "hparams": cfg, "episodes": episodes,
               "env_steps": env_steps, "best_val_smoothed": best_val,
               "best_episode": best_ep, "trainable_params": trainable}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train:{name}] done. best smoothed val return {best_val:.2f} at ep {best_ep}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Fashion-MNIST (Workshop 3)")
    p.add_argument("--config", choices=list(CONFIGS) + ["all"], default="b")
    p.add_argument("--all-configs", action="store_true", help="train a, b and c sequentially")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--reward-mode", choices=["plus_minus", "zero_one", "sparse"], default="plus_minus")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="replay warmup steps before first update (smoke-test overrides to 64)")
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--val-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--runs-dir", type=str, default="runs/workshop3")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--validate-env", action="store_true", help="run 100 random-policy episodes and exit")
    p.add_argument("--smoke-test", action="store_true", help="tiny run (5 episodes, CPU) to verify plumbing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = str(pick_device(args.device))
    if args.validate_env:
        validate_random_policy(split="train", n_episodes=100, max_steps=args.max_steps,
                               reward_mode=args.reward_mode, data_root=args.data_root)
        return
    if args.smoke_test:
        args.episodes, args.val_every, args.val_episodes = 5, 5, 2
        args.warmup_steps, args.batch_size = 64, 32
        device = "cpu"
    names = list(CONFIGS) if (args.all_configs or args.config == "all") else [args.config]
    start = time.time()
    results = [train_config(n, episodes=args.episodes, max_steps=args.max_steps,
                            reward_mode=args.reward_mode, batch_size=args.batch_size,
                            warmup_steps=args.warmup_steps,
                            val_every=args.val_every, val_episodes=args.val_episodes,
                            seed=args.seed, data_root=args.data_root,
                            checkpoint_dir=args.checkpoint_dir, runs_dir=args.runs_dir,
                            device=device) for n in names]
    out = Path(args.runs_dir) / "hparam_results.json"
    out.write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda r: r["best_val_smoothed"])
    # 'agent_best.pt' = the best config's checkpoint (single entry point for evaluate.py).
    import shutil

    best_ckpt = Path(args.checkpoint_dir) / f"agent_best_{best['config']}.pt"
    if best_ckpt.exists():
        shutil.copyfile(best_ckpt, Path(args.checkpoint_dir) / "agent_best.pt")
    print(f"[train] search done in {time.time() - start:.1f}s. "
          f"best config={best['config']} val={best['best_val_smoothed']:.2f} -> {out}")


if __name__ == "__main__":
    main()
