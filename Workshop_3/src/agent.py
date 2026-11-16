"""DQN agent for Workshop 3 (Mnih et al., 2015).

Components required by the guide:
- Q-network (CNN backbone mirroring Workshop 2 SmallCNN, ~140k params)
- experience replay buffer
- target network with periodic sync
- epsilon-greedy exploration with decay
- gradient clipping, double-DQN update, Adam + weight decay

Run `python -m src.agent --help` for a smoke test of the network/buffer.
"""

from __future__ import annotations

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# Q-network (same family as Workshop 2 SmallCNN for a fair comparison)
# --------------------------------------------------------------------------- #
def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class QNetwork(nn.Module):
    """CNN mapping (1,28,28) -> Q-values for the 10 classes."""

    def __init__(self, num_actions: int = NUM_CLASSES, head_dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 32),
            nn.MaxPool2d(2),  # 28 -> 14
            _conv_block(32, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(2),  # 14 -> 7
            _conv_block(64, 128),
            nn.MaxPool2d(2),  # 7 -> 3
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gap(self.features(x)))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Uniform experience replay (s, a, r, s', done)."""

    def __init__(self, capacity: int = 50_000) -> None:
        self._buf: deque = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, s, a: int, r: float, s_next, done: bool) -> None:
        self._buf.append((s, int(a), float(r), s_next, bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.randint(0, len(self._buf), size=batch_size)
        batch = [self._buf[i] for i in idx]
        s = np.stack([b[0] for b in batch]).astype(np.float32)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        s_next = np.stack([b[3] for b in batch]).astype(np.float32)
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return s, a, r, s_next, d


# --------------------------------------------------------------------------- #
# DQN agent
# --------------------------------------------------------------------------- #
class DQNAgent:
    def __init__(
        self,
        lr: float = 5e-4,
        gamma: float = 0.95,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        target_sync: int = 500,
        grad_clip: float = 10.0,
        weight_decay: float = 1e-4,
        double_dqn: bool = True,
        device: str = "cpu",
    ) -> None:
        self.gamma = float(gamma)
        self.eps = float(eps_start)
        self.eps_end = float(eps_end)
        self.eps_decay = float(eps_decay)
        self.target_sync = int(target_sync)
        self.grad_clip = float(grad_clip)
        self.double_dqn = bool(double_dqn)
        self.device = torch.device(device)

        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.SmoothL1Loss()  # Huber, standard for DQN
        self.updates = 0

    # -- interaction ----------------------------------------------------- #
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.eps:
            return random.randrange(NUM_CLASSES)
        state = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.q_net.eval()
        q = self.q_net(state)
        return int(q.argmax(dim=1).item())

    def decay_epsilon(self) -> float:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return self.eps

    # -- learning -------------------------------------------------------- #
    def update(self, batch) -> float:
        s, a, r, s_next, d = batch
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).unsqueeze(1).to(self.device)
        r_t = torch.from_numpy(r).unsqueeze(1).to(self.device)
        sn_t = torch.from_numpy(s_next).to(self.device)
        d_t = torch.from_numpy(d).unsqueeze(1).to(self.device)

        self.q_net.train()
        q_pred = self.q_net(s_t).gather(1, a_t)
        with torch.no_grad():
            if self.double_dqn:
                best = self.q_net(sn_t).argmax(dim=1, keepdim=True)
                q_next = self.target_net(sn_t).gather(1, best)
            else:
                q_next = self.target_net(sn_t).max(dim=1, keepdim=True)[0]
            target = r_t + self.gamma * (1.0 - d_t) * q_next

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    # -- persistence ------------------------------------------------------ #
    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.eps,
                "updates": self.updates,
                "config": {
                    "gamma": self.gamma, "eps_end": self.eps_end,
                    "eps_decay": self.eps_decay, "target_sync": self.target_sync,
                    "grad_clip": self.grad_clip, "double_dqn": self.double_dqn,
                },
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["state_dict"])
        self.target_net.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        self.eps = float(ckpt.get("epsilon", self.eps_end))
        self.updates = int(ckpt.get("updates", 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the DQN network and buffer")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    net = QNetwork()
    trainable, total = count_parameters(net)
    print(f"QNetwork: trainable={trainable:,} total={total:,}")
    with torch.no_grad():
        out = net(torch.zeros(4, 1, 28, 28))
    assert out.shape == (4, NUM_CLASSES), out.shape
    buf = ReplayBuffer(capacity=1000)
    obs = np.zeros((1, 28, 28), dtype=np.float32)
    for i in range(100):
        buf.push(obs, i % 10, 1.0 if i % 2 == 0 else -1.0, obs, False)
    agent = DQNAgent(device="cpu")
    loss = agent.update(buf.sample(args.batch))
    print(f"forward + update OK [loss={loss:.4f}]")
