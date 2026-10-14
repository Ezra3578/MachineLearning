"""Generation quality evaluation script for Workshop 5.

Deliverable: ``src/evaluate_gan.py``
Computes on 5,000 generated samples from Vanilla GAN and WGAN-GP:
- Fréchet Fashion-MNIST Distance (FFD / FID) using frozen Workshop 2 SmallCNN 128-d features.
- Domain Inception Score (IS, mean +/- std over 10 splits).
- Class Balance Entropy & Mode Coverage (10 classes).
- Intra-class Pixel Diversity (diagnosing mode collapse).
- Mean Classifier Confidence.
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
import scipy.linalg
import torch
import torch.nn.functional as F

from src.data import CLASS_NAMES, FMNIST_MEAN, FMNIST_STD, get_full_dataset, get_splits
from src.gan import load_generator_checkpoint, pick_device, set_seed
from src.generate_samples import sample_from_generator


def load_w2_evaluator(w2_ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load frozen Workshop 2 SmallCNN (93.05% test accuracy) as domain feature extractor."""
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
    return model


def extract_features_and_probs(
    model: torch.nn.Module,
    imgs_unit: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 128-d GAP features and 10-class softmax probabilities for images in [0, 1]."""
    feats_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    n = len(imgs_unit)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(imgs_unit[i : i + batch_size]).unsqueeze(1).float().to(device)
            batch_norm = (batch - FMNIST_MEAN) / FMNIST_STD
            h = model.gap(model.features(batch_norm)).flatten(1)
            logits = model.classifier(h)
            probs = F.softmax(logits, dim=1)
            feats_list.append(h.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
    return np.concatenate(feats_list, axis=0), np.concatenate(probs_list, axis=0)


def compute_frechet_distance(feats_r: np.ndarray, feats_g: np.ndarray) -> float:
    """Compute Fréchet Distance ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r*Sigma_g)^{1/2})."""
    mu1 = np.mean(feats_r, axis=0)
    sigma1 = np.cov(feats_r, rowvar=False)
    mu2 = np.mean(feats_g, axis=0)
    sigma2 = np.cov(feats_g, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(0.0, fid)


def compute_inception_score(probs: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """Compute Domain Inception Score exp(E[KL(p(y|x) || p(y))]) over ``splits`` subsets."""
    scores: list[float] = []
    n = len(probs)
    split_size = n // splits
    for k in range(splits):
        part = probs[k * split_size : (k + 1) * split_size]
        part = np.clip(part, 1e-10, 1.0)
        py = np.mean(part, axis=0, keepdims=True)
        kl = np.sum(part * (np.log(part) - np.log(py)), axis=1)
        scores.append(float(np.exp(np.mean(kl))))
    return float(np.mean(scores)), float(np.std(scores))


def compute_diversity_and_coverage(imgs_unit: np.ndarray, probs: np.ndarray) -> dict:
    """Compute mode coverage, class distribution KL vs uniform, and intra-class pixel L2 diversity."""
    preds = np.argmax(probs, axis=1)
    max_conf = np.max(probs, axis=1)
    counts = np.bincount(preds, minlength=10).astype(np.float64)
    freq = counts / np.sum(counts)
    uniform = np.full(10, 0.1, dtype=np.float64)
    freq_safe = np.clip(freq, 1e-10, 1.0)
    kl_to_uniform = float(np.sum(freq_safe * np.log(freq_safe / uniform)))
    class_entropy = float(-np.sum(freq_safe * np.log(freq_safe)))

    # Intra-class pairwise L2 distance (sample up to 100 images per class)
    rng = np.random.default_rng(42)
    intra_dists: list[float] = []
    for c in range(10):
        idx = np.where(preds == c)[0]
        if len(idx) >= 10:
            sub = imgs_unit[rng.choice(idx, size=min(100, len(idx)), replace=False)].reshape(-1, 784)
            dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            triu = dists[np.triu_indices(len(sub), k=1)]
            intra_dists.append(float(np.mean(triu)))

    return {
        "mean_confidence": float(np.mean(max_conf)),
        "high_conf_fraction_085": float(np.mean(max_conf >= 0.85)),
        "covered_modes_gt_2pct": int(np.sum(freq >= 0.02)),
        "class_entropy_nats": class_entropy,
        "max_entropy_nats": float(np.log(10.0)),
        "kl_to_uniform": kl_to_uniform,
        "intra_class_l2_diversity": float(np.mean(intra_dists)) if intra_dists else 0.0,
        "class_frequencies": {CLASS_NAMES[c]: round(float(freq[c]), 4) for c in range(10)},
    }


def plot_quality_diagnostics(
    real_Metrics: dict,
    van_metrics: dict,
    wgan_metrics: dict,
    out_path: Path,
) -> None:
    """Plot class balance distribution and quality metric comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))

    classes = CLASS_NAMES
    x = np.arange(len(classes))
    w = 0.26
    ax = axes[0]
    ax.bar(x - w, [real_Metrics["class_frequencies"][c] * 100 for c in classes], width=w, label="Real Test Set")
    ax.bar(x, [van_metrics["class_frequencies"][c] * 100 for c in classes], width=w, label="Vanilla GAN")
    ax.bar(x + w, [wgan_metrics["class_frequencies"][c] * 100 for c in classes], width=w, label="WGAN-GP")
    ax.axhline(10.0, color="black", linestyle="--", alpha=0.5, label="Ideal Uniform (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Predicted Class Share (%)")
    ax.set_title("Mode Coverage & Class Balance Across Generated Samples\n(Classified by Frozen W2 SmallCNN)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    metric_names = ["Fréchet Dist\n(FFD, lower=better)", "Inception Score\n(IS, higher=better)", "Intra-Class L2\nDiversity (higher=better)"]
    van_vals = [van_metrics["frechet_distance"], van_metrics["inception_score_mean"], van_metrics["intra_class_l2_diversity"]]
    wgan_vals = [wgan_metrics["frechet_distance"], wgan_metrics["inception_score_mean"], wgan_metrics["intra_class_l2_diversity"]]
    xm = np.arange(len(metric_names))
    b1 = ax2.bar(xm - 0.18, van_vals, width=0.34, label="Vanilla GAN", color="#ff7f0e")
    b2 = ax2.bar(xm + 0.18, wgan_vals, width=0.34, label="WGAN-GP", color="#2ca02c")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(xm)
    ax2.set_xticklabels(metric_names, fontsize=9)
    ax2.set_title("Quantitative Generation Quality: Vanilla GAN vs WGAN-GP")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate_gan] saved quality & mode coverage figure -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vanilla GAN and WGAN-GP sample quality")
    parser.add_argument("--vanilla-ckpt", type=str, default="checkpoints/gan_vanilla_best.pt")
    parser.add_argument("--wgan-ckpt", type=str, default="checkpoints/gan_wgan_best.pt")
    parser.add_argument("--w2-ckpt", type=str, default="../Workshop_2/checkpoints/model_best.pt")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--runs-dir", type=str, default="runs/workshop5")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    images, labels = get_full_dataset()
    splits = get_splits(labels, seed=args.seed)
    real_test = images[splits["test"]].astype(np.float32) / 255.0

    evaluator = load_w2_evaluator(args.w2_ckpt, device)
    feats_r, probs_r = extract_features_and_probs(evaluator, real_test, device)
    is_r_mean, is_r_std = compute_inception_score(probs_r)
    real_div = compute_diversity_and_coverage(real_test, probs_r)
    real_metrics = {
        "frechet_distance": 0.0,
        "inception_score_mean": is_r_mean,
        "inception_score_std": is_r_std,
        **real_div,
    }

    gen_van, _ = load_generator_checkpoint(args.vanilla_ckpt, map_location="cpu")
    gen_wgan, _ = load_generator_checkpoint(args.wgan_ckpt, map_location="cpu")

    van_imgs, _ = sample_from_generator(gen_van, args.num_samples, device, seed=args.seed)
    wgan_imgs, _ = sample_from_generator(gen_wgan, args.num_samples, device, seed=args.seed + 1)

    feats_v, probs_v = extract_features_and_probs(evaluator, van_imgs, device)
    ffd_v = compute_frechet_distance(feats_r, feats_v)
    is_v_mean, is_v_std = compute_inception_score(probs_v)
    div_v = compute_diversity_and_coverage(van_imgs, probs_v)
    van_metrics = {
        "frechet_distance": ffd_v,
        "inception_score_mean": is_v_mean,
        "inception_score_std": is_v_std,
        **div_v,
    }

    feats_w, probs_w = extract_features_and_probs(evaluator, wgan_imgs, device)
    ffd_w = compute_frechet_distance(feats_r, feats_w)
    is_w_mean, is_w_std = compute_inception_score(probs_w)
    div_w = compute_diversity_and_coverage(wgan_imgs, probs_w)
    wgan_metrics = {
        "frechet_distance": ffd_w,
        "inception_score_mean": is_w_mean,
        "inception_score_std": is_w_std,
        **div_w,
    }

    print(
        f"[evaluate_gan] Real Test   | FFD=0.0000 | IS={is_r_mean:.3f}+/-{is_r_std:.3f} | "
        f"L2_div={real_metrics['intra_class_l2_diversity']:.3f} | modes={real_metrics['covered_modes_gt_2pct']}/10"
    )
    print(
        f"[evaluate_gan] Vanilla GAN | FFD={ffd_v:.4f} | IS={is_v_mean:.3f}+/-{is_v_std:.3f} | "
        f"L2_div={van_metrics['intra_class_l2_diversity']:.3f} | KL_unif={van_metrics['kl_to_uniform']:.4f}"
    )
    print(
        f"[evaluate_gan] WGAN-GP     | FFD={ffd_w:.4f} | IS={is_w_mean:.3f}+/-{is_w_std:.3f} | "
        f"L2_div={wgan_metrics['intra_class_l2_diversity']:.3f} | KL_unif={wgan_metrics['kl_to_uniform']:.4f}"
    )

    plot_quality_diagnostics(real_metrics, van_metrics, wgan_metrics, runs_dir / "quality_and_mode_coverage.png")

    payload = {
        "num_samples_evaluated": args.num_samples,
        "real_test_reference": real_metrics,
        "vanilla_gan": van_metrics,
        "wgan_gp": wgan_metrics,
    }
    out_json = runs_dir / "gan_quality_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[evaluate_gan] saved quality evaluation JSON -> {out_json} [OK]")


if __name__ == "__main__":
    main()
