"""Evaluation script for Workshop 4 (Reconstruction quality + Anomaly Detection).

Deliverable: ``src/evaluate_ae.py``
Computes:
1. Test-set MSE, PSNR, SSIM, and Latent Silhouette Score across all AE/VAE/DAE models.
2. Per-class reconstruction error analysis (hardest vs easiest garment classes).
3. Side-by-side sample reconstruction panels (10 test examples, 1 per class) with pixel-wise |x - x_hat| maps.
4. Denoising AE and VAE latent interpolation / random sampling panels.
5. Semantic Anomaly Detection (Normal wearable classes vs Abnormal Bag [8]):
   - AE Reconstruction MSE score
   - VAE Reconstruction MSE score
   - VAE Composite ELBO score (Recon + beta * KL)
   - Validation threshold tuning + Test ROC-AUC & PR-AUC + comparison against W1/W2/W3.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)

from src.autoencoder import ConvVAE, DenoisingConvAE, load_autoencoder_checkpoint, pick_device, set_seed
from src.data import CLASS_NAMES, DEFAULT_ANOMALY_CLASSES, DEFAULT_NORMAL_CLASSES, build_ae_dataloaders


def evaluate_reconstruction_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute test-set MSE, PSNR, SSIM, per-class metrics, and latent Silhouette score."""
    model.eval().to(device)
    all_orig: list[np.ndarray] = []
    all_recon: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_kl: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            if isinstance(model, ConvVAE):
                x_hat, mu, logvar, _ = model(x_dev)
                z_vec = mu
                kl_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                all_kl.append(kl_sample.cpu().numpy())
            elif isinstance(model, DenoisingConvAE):
                x_hat, z_vec = model(x_dev, add_noise=False)
            else:
                x_hat, z_vec = model(x_dev)

            all_orig.append(x.numpy())
            all_recon.append(x_hat.cpu().numpy())
            all_latents.append(z_vec.cpu().numpy())
            all_labels.append(y.numpy())

    orig = np.concatenate(all_orig, axis=0)[:, 0]     # (N, 28, 28)
    recon = np.concatenate(all_recon, axis=0)[:, 0]   # (N, 28, 28)
    latents = np.concatenate(all_latents, axis=0)     # (N, z_dim)
    labels = np.concatenate(all_labels, axis=0)       # (N,)

    mse_per_sample = np.mean((orig - recon) ** 2, axis=(1, 2))
    mse_sum_per_sample = mse_per_sample * 784.0
    psnr_per_sample = 10.0 * np.log10(1.0 / np.maximum(mse_per_sample, 1e-10))

    ssim_vals = np.array(
        [ssim_fn(orig[i], recon[i], data_range=1.0) for i in range(len(orig))],
        dtype=np.float64,
    )

    sil_score = float(silhouette_score(latents, labels, sample_size=5000, random_state=42))

    per_class: dict[str, dict[str, float]] = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = labels == c_idx
        per_class[c_name] = {
            "mse_pixel": float(np.mean(mse_per_sample[mask])),
            "mse_sum": float(np.mean(mse_sum_per_sample[mask])),
            "psnr_db": float(np.mean(psnr_per_sample[mask])),
            "ssim": float(np.mean(ssim_vals[mask])),
            "count": int(np.sum(mask)),
        }

    result = {
        "mse_pixel": float(np.mean(mse_per_sample)),
        "mse_pixel_std": float(np.std(mse_per_sample)),
        "mse_sum": float(np.mean(mse_sum_per_sample)),
        "psnr_db": float(np.mean(psnr_per_sample)),
        "ssim": float(np.mean(ssim_vals)),
        "silhouette_latent": sil_score,
        "per_class": per_class,
    }
    if all_kl:
        kl_arr = np.concatenate(all_kl, axis=0)
        result["kl_divergence"] = float(np.mean(kl_arr))
    return result


def plot_reconstruction_panels(
    ae_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: Path,
) -> None:
    """Plot 10 test examples (1 per class): Original, AE Recon, |x - AE|, VAE Recon, |x - VAE|."""
    ae_model.eval().to(device)
    vae_model.eval().to(device)

    chosen_x: dict[int, torch.Tensor] = {}
    for x, y in test_loader:
        for i in range(x.size(0)):
            lbl = int(y[i])
            if lbl not in chosen_x:
                chosen_x[lbl] = x[i : i + 1]
            if len(chosen_x) == 10:
                break
        if len(chosen_x) == 10:
            break

    batch_x = torch.cat([chosen_x[c] for c in range(10)], dim=0).to(device)
    with torch.no_grad():
        ae_hat, _ = ae_model(batch_x)
        vae_hat, _, _, _ = vae_model(batch_x)

    orig_np = batch_x.cpu().numpy()[:, 0]
    ae_np = ae_hat.cpu().numpy()[:, 0]
    vae_np = vae_hat.cpu().numpy()[:, 0]
    ae_err = np.abs(orig_np - ae_np)
    vae_err = np.abs(orig_np - vae_np)

    fig, axes = plt.subplots(5, 10, figsize=(16, 8.2))
    row_titles = [
        "Original\nTest Image",
        "ConvAE\nReconstruction",
        "ConvAE Error\n|x - x_hat|",
        "ConvVAE\nReconstruction",
        "ConvVAE Error\n|x - x_hat|",
    ]

    for col in range(10):
        axes[0, col].imshow(orig_np[col], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(CLASS_NAMES[col], fontsize=8.5, fontweight="bold")
        axes[1, col].imshow(ae_np[col], cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(ae_err[col], cmap="magma", vmin=0, vmax=0.5)
        axes[3, col].imshow(vae_np[col], cmap="gray", vmin=0, vmax=1)
        axes[4, col].imshow(vae_err[col], cmap="magma", vmin=0, vmax=0.5)
        for r in range(5):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])

    for r, title in enumerate(row_titles):
        axes[r, 0].set_ylabel(title, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Workshop 4: Test-Set Reconstructions & Pixel-Wise Error Maps (ConvAE vs ConvVAE, z_dim=64)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate_ae] saved reconstruction panels -> {out_path}")


def plot_denoising_and_vae_generation(
    dae_model: DenoisingConvAE,
    vae_model: ConvVAE,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: Path,
) -> None:
    """Plot Denoising AE restoration (top) and VAE prior sampling + interpolation (bottom)."""
    dae_model.eval().to(device)
    vae_model.eval().to(device)
    set_seed(42)

    x_batch, _ = next(iter(test_loader))
    x_sample = x_batch[:8].to(device)
    with torch.no_grad():
        x_noisy = dae_model.corrupt(x_sample)
        x_denoised, _ = dae_model(x_noisy, add_noise=False)

        # Sample random z ~ N(0, I) from VAE prior
        z_rand = torch.randn(8, vae_model.z_dim, device=device)
        x_gen = vae_model.decode(z_rand)

        # Latent interpolation between two test items (e.g. T-shirt vs Ankle boot)
        mu_pair, _ = vae_model.encode(x_sample[:2])
        alphas = torch.linspace(0.0, 1.0, 8, device=device).unsqueeze(1)
        z_interp = (1.0 - alphas) * mu_pair[0:1] + alphas * mu_pair[1:2]
        x_interp = vae_model.decode(z_interp)

    fig, axes = plt.subplots(5, 8, figsize=(13, 8.2))
    row_labels = [
        "Clean Input",
        "Noisy (sigma=0.25)",
        "DAE Restored",
        "VAE Prior z~N(0,I)",
        "VAE Interpolation",
    ]
    arrays = [
        x_sample.cpu().numpy()[:, 0],
        x_noisy.cpu().numpy()[:, 0],
        x_denoised.cpu().numpy()[:, 0],
        x_gen.cpu().numpy()[:, 0],
        x_interp.cpu().numpy()[:, 0],
    ]
    for r in range(5):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=8.5, fontweight="bold")
        for c in range(8):
            axes[r, c].imshow(arrays[r][c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(
        "Denoising Autoencoder Restoration & VAE Generative Sampling / Latent Interpolation",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate_ae] saved DAE & VAE generative panel -> {out_path}")


def compute_anomaly_scores(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mse_scores, elbo_scores, ground_truth_labels)."""
    model.eval().to(device)
    mse_list: list[np.ndarray] = []
    elbo_list: list[np.ndarray] = []
    lbl_list: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            bsz = x_dev.size(0)
            if isinstance(model, ConvVAE):
                x_hat, mu, logvar, _ = model(x_dev)
                recon_sum = ((x_hat - x_dev) ** 2).view(bsz, -1).sum(dim=1)
                kl_sum = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                mse_s = (recon_sum / 784.0).cpu().numpy()
                elbo_s = (recon_sum + beta * kl_sum).cpu().numpy()
            else:
                x_hat, _ = model(x_dev)
                recon_sum = ((x_hat - x_dev) ** 2).view(bsz, -1).sum(dim=1)
                mse_s = (recon_sum / 784.0).cpu().numpy()
                elbo_s = recon_sum.cpu().numpy()

            mse_list.append(mse_s)
            elbo_list.append(elbo_s)
            lbl_list.append(y.numpy())

    return (
        np.concatenate(mse_list, axis=0),
        np.concatenate(elbo_list, axis=0),
        np.concatenate(lbl_list, axis=0),
    )


def evaluate_anomaly_detection(
    ae_anom_model: torch.nn.Module,
    vae_anom_model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_plot_path: Path,
) -> dict:
    """Evaluate semantic anomaly detection (Normal=wearable garments, Abnormal=Bag [8])."""
    anom_set = set(DEFAULT_ANOMALY_CLASSES)

    val_ae_mse, _, val_y = compute_anomaly_scores(ae_anom_model, val_loader, device)
    val_vae_mse, val_vae_elbo, _ = compute_anomaly_scores(vae_anom_model, val_loader, device, beta=1.0)
    val_is_anom = np.array([1 if int(lbl) in anom_set else 0 for lbl in val_y], dtype=np.int32)

    test_ae_mse, _, test_y = compute_anomaly_scores(ae_anom_model, test_loader, device)
    test_vae_mse, test_vae_elbo, _ = compute_anomaly_scores(vae_anom_model, test_loader, device, beta=1.0)
    test_is_anom = np.array([1 if int(lbl) in anom_set else 0 for lbl in test_y], dtype=np.int32)

    methods = {
        "AE_Recon_MSE": (val_ae_mse, test_ae_mse),
        "VAE_Recon_MSE": (val_vae_mse, test_vae_mse),
        "VAE_ELBO_Recon_plus_KL": (val_vae_elbo, test_vae_elbo),
    }

    results: dict[str, dict] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    for name, (val_scores, test_scores) in methods.items():
        # Tune threshold on validation set maximizing Youden's J = TPR - FPR
        fpr_v, tpr_v, thresh_v = roc_curve(val_is_anom, val_scores)
        best_idx = int(np.argmax(tpr_v - fpr_v))
        opt_thresh = float(thresh_v[best_idx])

        auc_roc = float(roc_auc_score(test_is_anom, test_scores))
        pr_auc = float(average_precision_score(test_is_anom, test_scores))
        preds = (test_scores >= opt_thresh).astype(np.int32)

        results[name] = {
            "val_threshold": opt_thresh,
            "test_roc_auc": auc_roc,
            "test_pr_auc": pr_auc,
            "test_f1": float(f1_score(test_is_anom, preds)),
            "test_precision": float(precision_score(test_is_anom, preds, zero_division=0)),
            "test_recall_tpr": float(recall_score(test_is_anom, preds, zero_division=0)),
            "normal_mean_score": float(np.mean(test_scores[test_is_anom == 0])),
            "anomaly_mean_score": float(np.mean(test_scores[test_is_anom == 1])),
        }

        fpr_t, tpr_t, _ = roc_curve(test_is_anom, test_scores)
        axes[0].plot(fpr_t, tpr_t, linewidth=2.2, label=f"{name} (AUC = {auc_roc:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5000)")
    axes[0].set_title("Semantic Anomaly Detection ROC Curves\n(Normal: 9 Garment/Footwear Classes vs Abnormal: Bag [8])")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Histogram of Normal vs Abnormal MSE scores
    axes[1].hist(
        test_ae_mse[test_is_anom == 0],
        bins=50,
        density=True,
        alpha=0.6,
        color="#1f77b4",
        label=f"Normal Classes (mean MSE={np.mean(test_ae_mse[test_is_anom == 0]):.4f})",
    )
    axes[1].hist(
        test_ae_mse[test_is_anom == 1],
        bins=50,
        density=True,
        alpha=0.6,
        color="#d62728",
        label=f"Abnormal: Bag [8] (mean MSE={np.mean(test_ae_mse[test_is_anom == 1]):.4f})",
    )
    axes[1].axvline(
        results["AE_Recon_MSE"]["val_threshold"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Val-tuned threshold = {results['AE_Recon_MSE']['val_threshold']:.4f}",
    )
    axes[1].set_title("Test-Set Reconstruction Error Distribution (ConvAE)")
    axes[1].set_xlabel("Per-Pixel Reconstruction MSE")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate_ae] saved anomaly detection ROC & histogram -> {out_plot_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AE and VAE reconstruction & anomaly detection")
    parser.add_argument("--ae-ckpt", type=str, default="checkpoints/ae_best.pt")
    parser.add_argument("--vae-ckpt", type=str, default="checkpoints/vae_best.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--runs-dir", type=str, default="runs/workshop4")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, test_loader = build_ae_dataloaders(batch_size=256, num_workers=0, seed=42)

    all_eval: dict[str, dict] = {"models": {}, "anomaly_detection": {}}

    # Evaluate all available reconstruction checkpoints
    candidate_ckpts = [
        ("ae_z32", ckpt_dir / "ae_z32_best.pt"),
        ("ae_z64", ckpt_dir / "ae_z64_best.pt"),
        ("ae_z128", ckpt_dir / "ae_z128_best.pt"),
        ("vae_beta0.5", ckpt_dir / "vae_z64_beta0.5_best.pt"),
        ("vae_beta1.0", ckpt_dir / "vae_z64_beta1.0_best.pt"),
        ("vae_beta4.0", ckpt_dir / "vae_z64_beta4.0_best.pt"),
        ("dae_z64", ckpt_dir / "dae_best.pt"),
    ]
    for label, path in candidate_ckpts:
        if path.exists():
            model, ckpt = load_autoencoder_checkpoint(str(path), map_location="cpu")
            res = evaluate_reconstruction_model(model, test_loader, device)
            res["z_dim"] = ckpt.get("z_dim", 64)
            res["beta"] = ckpt.get("config", {}).get("beta", 0.0)
            all_eval["models"][label] = res
            print(
                f"[evaluate_ae] {label:<12} | MSE={res['mse_pixel']:.5f} | "
                f"PSNR={res['psnr_db']:.2f} dB | SSIM={res['ssim']:.4f} | "
                f"Silhouette={res['silhouette_latent']:.4f}"
            )

    ae_best_model, _ = load_autoencoder_checkpoint(args.ae_ckpt, map_location="cpu")
    vae_best_model, _ = load_autoencoder_checkpoint(args.vae_ckpt, map_location="cpu")

    plot_reconstruction_panels(
        ae_best_model,
        vae_best_model,
        test_loader,
        device,
        runs_dir / "reconstruction_panels.png",
    )

    dae_path = ckpt_dir / "dae_best.pt"
    if dae_path.exists():
        dae_model, _ = load_autoencoder_checkpoint(str(dae_path), map_location="cpu")
        plot_denoising_and_vae_generation(
            dae_model,  # type: ignore[arg-type]
            vae_best_model,  # type: ignore[arg-type]
            test_loader,
            device,
            runs_dir / "denoising_and_vae_generation.png",
        )

    # Anomaly detection evaluation
    ae_anom_path = ckpt_dir / "ae_anomaly_best.pt"
    vae_anom_path = ckpt_dir / "vae_anomaly_best.pt"
    if ae_anom_path.exists() and vae_anom_path.exists():
        ae_anom, _ = load_autoencoder_checkpoint(str(ae_anom_path), map_location="cpu")
        vae_anom, _ = load_autoencoder_checkpoint(str(vae_anom_path), map_location="cpu")
        anom_res = evaluate_anomaly_detection(
            ae_anom,
            vae_anom,
            val_loader,
            test_loader,
            device,
            runs_dir / "anomaly_detection_roc.png",
        )
        all_eval["anomaly_detection"] = anom_res
        for k, v in anom_res.items():
            print(
                f"[evaluate_ae] Anomaly {k:<23} | ROC-AUC={v['test_roc_auc']:.4f} | "
                f"PR-AUC={v['test_pr_auc']:.4f} | F1={v['test_f1']:.4f}"
            )

    out_json = runs_dir / "evaluation_results.json"
    out_json.write_text(json.dumps(all_eval, indent=2), encoding="utf-8")
    print(f"[evaluate_ae] saved evaluation metrics -> {out_json} [OK]")


if __name__ == "__main__":
    main()
