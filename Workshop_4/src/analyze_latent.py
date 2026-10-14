"""Latent space visualization and analysis script for Workshop 4.

Deliverable: ``src/analyze_latent.py``
Encodes all 10,500 test-set examples using:
- Standard Autoencoder (ConvAE, z_dim=64)
- Variational Autoencoders (ConvVAE, beta in {0.5, 1.0, 4.0}, deterministic mu)
- Frozen Workshop 2 SmallCNN encoder (128-d Global Average Pooling representation)
Computes Silhouette Scores in both high-D latent space and 2D embeddings,
and produces side-by-side t-SNE and UMAP scatter plots colored by ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.autoencoder import ConvVAE, DenoisingConvAE, load_autoencoder_checkpoint, pick_device, set_seed
from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, FashionMNISTTensorDataset, get_full_dataset, get_splits


def extract_ae_vae_latents(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (latents [N, D], labels [N]) using deterministic mu for VAE."""
    model.eval().to(device)
    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            if isinstance(model, ConvVAE):
                mu, _ = model.encode(x_dev)
                z = mu
            elif isinstance(model, DenoisingConvAE):
                z = model.encode(x_dev)
            else:
                z = model.encode(x_dev)
            zs.append(z.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def extract_w2_smallcnn_latents(
    w2_ckpt_path: str | Path,
    images: np.ndarray,
    labels: np.ndarray,
    test_indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 128-d penultimate GAP features from frozen Workshop 2 SmallCNN."""
    w2_src = Path(__file__).resolve().parents[2] / "Workshop_2" / "src"
    if str(w2_src) not in sys.path:
        sys.path.insert(0, str(w2_src))
    from model import SmallCNN  # type: ignore

    try:
        ckpt = torch.load(w2_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(w2_ckpt_path, map_location="cpu")

    model = SmallCNN(num_classes=10)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(device)

    ds = FashionMNISTTensorDataset(images, labels, test_indices, normalize_mode="zscore")
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    feats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            h = model.gap(model.features(x_dev)).flatten(1)
            feats.append(h.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(ys, axis=0)


def compute_2d_projections(
    latents: np.ndarray,
    labels: np.ndarray,
    max_points: int = 3500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D t-SNE and UMAP embeddings on a stratified subset of test latents."""
    rng = np.random.default_rng(seed)
    if len(latents) > max_points:
        idx = rng.choice(len(latents), size=max_points, replace=False)
        idx = np.sort(idx)
        sub_z = latents[idx]
        sub_y = labels[idx]
    else:
        sub_z = latents
        sub_y = labels

    tsne = TSNE(n_components=2, perplexity=30.0, init="pca", learning_rate="auto", random_state=seed)
    z_tsne = tsne.fit_transform(sub_z)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
    z_umap = reducer.fit_transform(sub_z)
    return z_tsne, z_umap, sub_y


def plot_latent_grid(
    records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float, float, float]],
    out_path: Path,
    title: str,
) -> None:
    """Plot 2xN grid: top row = t-SNE, bottom row = UMAP, colored by 10 Fashion-MNIST classes."""
    n_cols = len(records)
    fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9.2))
    cmap = plt.get_cmap("tab10")

    for col, (name, z_tsne, z_umap, sub_y, sil_hd, sil_tsne, sil_umap) in enumerate(records):
        ax_top = axes[0, col] if n_cols > 1 else axes[0]
        ax_bot = axes[1, col] if n_cols > 1 else axes[1]

        for c_idx, c_name in enumerate(CLASS_NAMES):
            mask = sub_y == c_idx
            ax_top.scatter(
                z_tsne[mask, 0],
                z_tsne[mask, 1],
                s=7,
                alpha=0.72,
                color=cmap(c_idx),
                label=c_name if col == 0 else None,
            )
            ax_bot.scatter(
                z_umap[mask, 0],
                z_umap[mask, 1],
                s=7,
                alpha=0.72,
                color=cmap(c_idx),
            )

        ax_top.set_title(
            f"{name} — t-SNE (2D)\nLatent Sil={sil_hd:.3f} | 2D Sil={sil_tsne:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_top.set_xlabel("t-SNE Dim 1")
        ax_top.set_ylabel("t-SNE Dim 2")
        ax_top.grid(True, alpha=0.25)

        ax_bot.set_title(
            f"{name} — UMAP (2D)\nLatent Sil={sil_hd:.3f} | 2D Sil={sil_umap:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax_bot.set_xlabel("UMAP Dim 1")
        ax_bot.set_ylabel("UMAP Dim 2")
        ax_bot.grid(True, alpha=0.25)

    handles, lbls = (axes[0, 0] if n_cols > 1 else axes[0]).get_legend_handles_labels()
    fig.legend(
        handles,
        lbls,
        loc="lower center",
        ncol=5,
        markerscale=2.2,
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze_latent] saved latent visualization -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AE, VAE, and W2 supervised latent spaces")
    parser.add_argument("--ae-ckpt", type=str, default="checkpoints/ae_best.pt")
    parser.add_argument("--vae-ckpt", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--w2-ckpt", type=str, default="../Workshop_2/checkpoints/model_best.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--runs-dir", type=str, default="runs/workshop4")
    parser.add_argument("--max-points", type=int, default=3500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    runs_dir = Path(args.runs_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=args.seed)
    test_ds = FashionMNISTTensorDataset(images, labels, splits["test"], normalize_mode="unit")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    ae_model, _ = load_autoencoder_checkpoint(args.ae_ckpt, map_location="cpu")
    vae_model, _ = load_autoencoder_checkpoint(args.vae_ckpt, map_location="cpu")

    z_ae, y_test = extract_ae_vae_latents(ae_model, test_loader, device)
    z_vae, _ = extract_ae_vae_latents(vae_model, test_loader, device)
    z_w2, _ = extract_w2_smallcnn_latents(args.w2_ckpt, images, labels, splits["test"], device)

    main_models = [
        ("Unsupervised ConvAE (z=64)", z_ae),
        ("Unsupervised ConvVAE (mu, beta=1.0)", z_vae),
        ("Supervised W2 SmallCNN (128-d GAP)", z_w2),
    ]

    summary_metrics: dict[str, dict] = {}
    records = []
    for title_name, z_mat in main_models:
        sil_hd = float(silhouette_score(z_mat, y_test, sample_size=5000, random_state=args.seed))
        z_tsne, z_umap, sub_y = compute_2d_projections(z_mat, y_test, max_points=args.max_points, seed=args.seed)
        sil_tsne = float(silhouette_score(z_tsne, sub_y))
        sil_umap = float(silhouette_score(z_umap, sub_y))
        records.append((title_name, z_tsne, z_umap, sub_y, sil_hd, sil_tsne, sil_umap))
        summary_metrics[title_name] = {
            "latent_dim": int(z_mat.shape[1]),
            "silhouette_high_dim": sil_hd,
            "silhouette_tsne_2d": sil_tsne,
            "silhouette_umap_2d": sil_umap,
        }
        print(
            f"[analyze_latent] {title_name:<36} | HD Sil={sil_hd:.4f} | "
            f"t-SNE Sil={sil_tsne:.4f} | UMAP Sil={sil_umap:.4f}"
        )

    plot_latent_grid(
        records,
        runs_dir / "latent_tsne_umap_comparison.png",
        "Workshop 4: Unsupervised (ConvAE & ConvVAE) vs Supervised (W2 SmallCNN) Latent Spaces",
    )

    # Also compare beta in {0.5, 1.0, 4.0} if all three exist
    beta_ckpts = [
        ("beta = 0.5", ckpt_dir / "vae_z64_beta0.5_best.pt"),
        ("beta = 1.0", ckpt_dir / "vae_z64_beta1.0_best.pt"),
        ("beta = 4.0", ckpt_dir / "vae_z64_beta4.0_best.pt"),
    ]
    if all(p.exists() for _, p in beta_ckpts):
        beta_records = []
        for b_name, b_path in beta_ckpts:
            m_b, _ = load_autoencoder_checkpoint(str(b_path), map_location="cpu")
            z_b, _ = extract_ae_vae_latents(m_b, test_loader, device)
            sil_hd = float(silhouette_score(z_b, y_test, sample_size=5000, random_state=args.seed))
            z_tsne, z_umap, sub_y = compute_2d_projections(z_b, y_test, max_points=args.max_points, seed=args.seed)
            sil_tsne = float(silhouette_score(z_tsne, sub_y))
            sil_umap = float(silhouette_score(z_umap, sub_y))
            beta_records.append((f"ConvVAE ({b_name})", z_tsne, z_umap, sub_y, sil_hd, sil_tsne, sil_umap))
            summary_metrics[f"ConvVAE ({b_name})"] = {
                "latent_dim": int(z_b.shape[1]),
                "silhouette_high_dim": sil_hd,
                "silhouette_tsne_2d": sil_tsne,
                "silhouette_umap_2d": sil_umap,
            }
        plot_latent_grid(
            beta_records,
            runs_dir / "beta_vae_latent_comparison.png",
            "Workshop 4: Effect of Beta on ConvVAE Latent Space Structure (beta in {0.5, 1.0, 4.0})",
        )

    out_json = runs_dir / "latent_analysis.json"
    out_json.write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    print(f"[analyze_latent] saved latent analysis metrics -> {out_json} [OK]")


if __name__ == "__main__":
    main()
