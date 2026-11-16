# Workshop 4. Autoencoders for Fashion-MNIST

Team 3 (computer vision and object recognition)

Jhojan Stiven Aragón Ramírez (20221020060), Santiago Reyes Gómez (20221020068), and Juan Andrés Jiménez Palomino (20221020087)

We trained a convolutional autoencoder (`ConvAE`), a denoising autoencoder (`DenoisingConvAE`), and $\beta$-variational autoencoders (`ConvVAE`) on Fashion-MNIST. We measure reconstruction with MSE, PSNR, and SSIM, and use ROC-AUC to evaluate how well the anomaly scores separate bags from wearable items. We compare the latent features with the frozen Workshop 2 `SmallCNN` encoder using silhouette scores and t-SNE and UMAP plots.

The best ConvAE reached 26.11 dB PSNR and 0.909 SSIM. For anomaly detection, the best ROC-AUC was 0.9530 with the VAE's combined reconstruction and KL score.

## Structure

```
Workshop_4/
├── src/
│   ├── data.py            # Shared 70/15/15 split (seed 42), [0,1] scaling, anomaly loaders
│   ├── autoencoder.py     # ConvAE, ConvVAE (reparameterization + ELBO), DenoisingConvAE
│   ├── train_ae.py        # Training pipeline for AE, β-VAE (β ∈ {0.5, 1.0, 4.0}), DAE & anomaly runs
│   ├── evaluate_ae.py     # Test MSE/PSNR/SSIM, 6–10 sample panels + error maps, anomaly ROC-AUC
│   └── analyze_latent.py  # Latent extraction (AE vs VAE μ vs frozen W2 SmallCNN), Silhouette, t-SNE & UMAP
├── checkpoints/
│   ├── ae_best.pt         # Best Standard Autoencoder checkpoint
│   └── vae_best.pt        # Best Variational Autoencoder checkpoint
├── runs/workshop4/        # Training logs, JSON metrics, TensorBoard events, and PNG figures
├── notebooks/             # Interactive analysis notebooks
└── pyproject.toml         # Package metadata and dependencies
```

## Setup

```bash
cd Workshop_4
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Quickstart

```bash
# 1. Verify AE & VAE architectures and parameter counts
python -m src.autoencoder

# 2. Train Standard Autoencoder (z_dim ∈ {32, 64, 128})
python -m src.train_ae --model ae --z-dim 64 --epochs 40

# 3. Train β-VAE across β ∈ {0.5, 1.0, 4.0}
python -m src.train_ae --model vae --z-dim 64 --beta 0.5 --epochs 40
python -m src.train_ae --model vae --z-dim 64 --beta 1.0 --epochs 40
python -m src.train_ae --model vae --z-dim 64 --beta 4.0 --epochs 40

# 4. Train anomaly-detection models on Normal classes only (excluding Bag [8])
python -m src.train_ae --model ae --z-dim 64 --anomaly --epochs 40
python -m src.train_ae --model vae --z-dim 64 --beta 1.0 --anomaly --epochs 40

# 5. Evaluate reconstruction quality (MSE, PSNR, SSIM + sample panels) & anomaly ROC-AUC
python -m src.evaluate_ae --ae-ckpt checkpoints/ae_best.pt --vae-ckpt checkpoints/vae_best.pt --plot

# 6. Latent space analysis (Silhouette Score + t-SNE + UMAP vs frozen W2 SmallCNN)
python -m src.analyze_latent --ae-ckpt checkpoints/ae_best.pt --vae-ckpt checkpoints/vae_best.pt \
    --w2-ckpt ../Workshop_2/checkpoints/model_best.pt
```

## Files required by `Workshop_4.pdf`

| Requirement | File / Path |
|---|---|
| AE and VAE implementations | `src/autoencoder.py` |
| Training script ($\beta \in \{0.5, 1.0, 4.0\}$, $z_{\text{dim}} \in \{32, 64, 128\}$) | `src/train_ae.py` |
| Evaluation (MSE, PSNR, SSIM, sample panels, anomaly ROC-AUC) | `src/evaluate_ae.py` |
| Latent space visualization (t-SNE, UMAP, Silhouette vs W2) | `src/analyze_latent.py` |
| Best checkpoints | `checkpoints/ae_best.pt`, `checkpoints/vae_best.pt` |
| Training logs and figures | `runs/workshop4/` |
| Report (PDF) | [Workshop 4 report](Workshop4.pdf) |
