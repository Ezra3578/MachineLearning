"""Custom gymnasium environment for Workshop 3.

Formulation (see docs/mdp_spec.md): sequential Fashion-MNIST classification.
Each step the agent observes one 28x28 image and predicts a class (0-9).
Reward is +1/-1 by default. Episodes last `max_steps` i.i.d. samples.

Guide requirements covered here:
- subclass gymnasium.Env with action_space / observation_space, reset(), step()
- render() for debugging
- validate_random_policy(n=100): mean return, length distribution, action frequency
"""

from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, get_split_arrays

REWARD_MODES = ("plus_minus", "zero_one", "sparse")


class FashionMNISTClassifyEnv(gym.Env):
    """Sequential Fashion-MNIST classifier environment (contextual bandit)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        split: str = "train",
        max_steps: int = 100,
        reward_mode: str = "plus_minus",
        data_root: str = "data",
        seed: int = 42,
    ) -> None:
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        self.split = split
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode
        self.data_root = data_root

        self.action_space = spaces.Discrete(len(CLASS_NAMES))
        # Normalized float image in [0, 1]-ish range after standardization clip.
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(1, 28, 28), dtype=np.float32
        )

        self._images, self._labels = get_split_arrays(
            split=split, data_root=data_root, seed=seed
        )
        self._rng = np.random.default_rng(seed)
        self._order: np.ndarray = np.array([], dtype=np.int64)
        self._t = 0
        self._current_label = 0
        self._current_image: np.ndarray = np.zeros((1, 28, 28), dtype=np.float32)
        self._correct_so_far = 0

    # ------------------------------------------------------------------ #
    def _sample_obs(self) -> np.ndarray:
        pos = self._order[self._t] if self._t < len(self._order) else self._rng.integers(0, len(self._images))
        img = self._images[int(pos)].astype(np.float32) / 255.0
        self._current_label = int(self._labels[int(pos)])
        obs = ((img - FMNIST_MEAN) / FMNIST_STD).reshape(1, 28, 28).astype(np.float32)
        self._current_image = obs
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Pre-shuffle one episode worth of indices for reproducibility.
        self._order = self._rng.integers(0, len(self._images), size=self.max_steps)
        self._t = 0
        self._correct_so_far = 0
        return self._sample_obs(), {"true_label": self._current_label}

    def step(self, action):
        action = int(action)
        correct = action == self._current_label
        if self.reward_mode == "plus_minus":
            reward = 1.0 if correct else -1.0
        elif self.reward_mode == "zero_one":
            reward = 1.0 if correct else 0.0
        else:  # sparse: no intermediate signal, lump sum at the end
            reward = 0.0
        self._correct_so_far += int(correct)
        self._t += 1
        truncated = self._t >= self.max_steps
        if truncated and self.reward_mode == "sparse":
            reward = float(self._correct_so_far)
        terminated = False
        info = {"true_label": self._current_label, "correct": correct}
        obs = self._sample_obs() if not truncated else self._current_image
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        """Return an rgb_array with the current image (for debugging/trajectories)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = self._current_image[0]
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"true={CLASS_NAMES[self._current_label]} t={self._t}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf


def validate_random_policy(
    split: str = "train",
    n_episodes: int = 100,
    max_steps: int = 100,
    reward_mode: str = "plus_minus",
    data_root: str = "data",
    seed: int = 0,
) -> dict:
    """Run `n_episodes` random episodes; report return/length/action diagnostics.

    Expected healthy values (plus_minus): mean return ~= -80 (10% accuracy on
    100 steps -> 10*(+1) + 90*(-1) = -80), all lengths == max_steps, and every
    action appearing roughly n_episodes*max_steps/10 times.
    """
    env = FashionMNISTClassifyEnv(
        split=split, max_steps=max_steps, reward_mode=reward_mode,
        data_root=data_root, seed=seed,
    )
    rng = np.random.default_rng(seed)
    returns, lengths = [], []
    action_counts: Counter = Counter()
    for _ in range(n_episodes):
        env.reset()
        total, steps = 0.0, 0
        done = False
        while not done:
            action = int(rng.integers(0, env.action_space.n))
            action_counts[action] += 1
            _, reward, _, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = truncated
        returns.append(total)
        lengths.append(steps)
    returns_arr = np.array(returns, dtype=np.float64)
    report = {
        "n_episodes": n_episodes,
        "reward_mode": reward_mode,
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "min_return": float(returns_arr.min()),
        "max_return": float(returns_arr.max()),
        "mean_length": float(np.mean(lengths)),
        "all_lengths_equal_max": bool(all(L == max_steps for L in lengths)),
        "action_counts": {int(k): int(v) for k, v in sorted(action_counts.items())},
        "all_actions_seen": bool(len(action_counts) == env.action_space.n),
        "implied_accuracy": float((returns_arr.mean() + max_steps) / (2 * max_steps))
        if reward_mode == "plus_minus"
        else None,
    }
    print(
        f"[validate-env] episodes={n_episodes} mode={reward_mode} | "
        f"mean_return={report['mean_return']:.2f}±{report['std_return']:.2f} "
        f"[{report['min_return']:.0f},{report['max_return']:.0f}] | "
        f"lengths_ok={report['all_lengths_equal_max']} | "
        f"actions_seen={report['all_actions_seen']} counts={report['action_counts']}"
    )
    return report


if __name__ == "__main__":
    validate_random_policy()
