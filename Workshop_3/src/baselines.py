"""Baseline strategies for Workshop 3 (guide: Evaluation and Comparison).

- RandomPolicy: uniform over the 10 classes (lower bound, ~10% accuracy).
- MajorityPolicy: always predicts the most frequent training label.
- SupervisedHeuristicPolicy: loads a Workshop 2 checkpoint (SmallCNN/ResNet-18)
  and acts greedily. Degrades gracefully to random if no checkpoint is found.

All policies expose `act(obs) -> int` and are scored with `evaluate_policy`
on the same test environment as the RL agent for a fair comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data import CLASS_NAMES, get_split_arrays
from src.env import FashionMNISTClassifyEnv


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0, n_actions: int = 10) -> None:
        self._rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, obs) -> int:  # noqa: ARG002
        return int(self._rng.integers(0, self.n_actions))


class MajorityPolicy:
    """Fixed strategy: always predict the majority training label."""

    name = "majority"

    def __init__(self, data_root: str = "data", seed: int = 42) -> None:
        _, labels = get_split_arrays(split="train", data_root=data_root, seed=seed)
        values, counts = np.unique(labels, return_counts=True)
        self.label = int(values[int(np.argmax(counts))])

    def act(self, obs) -> int:  # noqa: ARG002
        return self.label


class SupervisedHeuristicPolicy:
    """Greedy heuristic = Workshop 2 supervised CNN used as a fixed policy.

    Loads `checkpoints/model_best.pt` (SmallCNN or ResNet-18). The ResNet variant
    needs 224x224 RGB inputs; the SmallCNN path reuses raw normalization.
    """

    name = "supervised-heuristic"

    def __init__(
        self,
        checkpoint: str | Path = "../Workshop_2/checkpoints/model_best.pt",
        data_root: str = "data",
        device: str = "cpu",
    ) -> None:
        from src.data import FMNIST_MEAN, FMNIST_STD

        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        self.model = None
        self.model_name = "none"
        if not self.checkpoint.exists():
            print(f"[baselines] WARNING: W2 checkpoint not found at {self.checkpoint}; "
                  "heuristic falls back to random predictions.")
            self._fallback = RandomPolicy(seed=0)
            return
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model_name = ckpt.get("model_name", "smallcnn")
        import sys

        w2_src = Path(__file__).resolve().parents[2] / "Workshop_2" / "src"
        if str(w2_src) not in sys.path:
            sys.path.insert(0, str(w2_src))
        from model import build_model  # type: ignore

        self.model = build_model(
            self.model_name, num_classes=len(ckpt.get("classes", CLASS_NAMES)), pretrained=False
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self._mean = FMNIST_MEAN
        self._std = FMNIST_STD
        self._fallback = None
        print(f"[baselines] loaded supervised heuristic: {self.model_name} from {self.checkpoint}")

    def act(self, obs) -> int:
        if self.model is None:
            return self._fallback.act(obs)
        import torch.nn.functional as F

        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if self.model_name == "resnet18":
            # Grayscale 28x28 -> RGB 224x224 with ImageNet stats (same as W2 data.py).
            from torchvision import transforms

            tfm = transforms.Compose([
                transforms.Lambda(lambda t: t.repeat(1, 3, 1, 1)),
                transforms.Resize(224, antialias=True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # Undo FMNIST normalization back to [0,1] first (approx).
            x = x * self._std + self._mean
            x = tfm(x)
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            probs = F.softmax(logits, dim=1)
        return int(probs.argmax(dim=1).item())


def evaluate_policy(
    policy,
    split: str = "test",
    n_episodes: int = 50,
    max_steps: int = 100,
    reward_mode: str = "plus_minus",
    data_root: str = "data",
    seed: int = 0,
) -> dict:
    """Score any `act(obs)` policy with the same protocol as the RL agent."""
    env = FashionMNISTClassifyEnv(
        split=split, max_steps=max_steps, reward_mode=reward_mode,
        data_root=data_root, seed=seed,
    )
    returns, correct, total_steps = [], 0, 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_return = False, 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, _, truncated, info = env.step(action)
            ep_return += reward
            correct += int(info["correct"])
            total_steps += 1
            done = truncated
        returns.append(ep_return)
    returns_arr = np.array(returns, dtype=np.float64)
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "n_episodes": n_episodes,
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "accuracy": float(correct / max(total_steps, 1)),
    }


if __name__ == "__main__":
    for pol in (RandomPolicy(), MajorityPolicy()):
        print(evaluate_policy(pol, n_episodes=5))
