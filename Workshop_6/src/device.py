"""Device selection shared by the Workshop 6 scripts.

Priority follows the course rule for this workspace: CUDA, then Apple MPS,
then AMD DirectML (torch_directml), then CPU. The same helper is used by the
training, evaluation, interpretability and figure scripts so a run reproduces
on a different machine.
"""

from __future__ import annotations

import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Resolve a device string; `auto` walks the supported backends in order."""
    pref = (preference or "auto").lower()
    if pref != "auto":
        return torch.device(pref)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    try:  # AMD GPUs on Windows through torch-directml
        import torch_directml

        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")
