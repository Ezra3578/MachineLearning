# Workshop 5. GANs for Fashion-MNIST

Team 3 (computer vision and object recognition)

Jhojan Stiven Aragón Ramírez (20221020060), Santiago Reyes Gómez (20221020098), and Juan Andrés Jiménez Palomino (20221020087)

We compare a vanilla DCGAN with a Wasserstein GAN with gradient penalty (`WGAN-GP`). We inspect failure modes and measure sample quality using Fréchet Fashion-MNIST Distance, domain Inception Score, class balance, and within-class diversity. The code also supports an optional conditional GAN (`cGAN`).

We retrained the Workshop 2 `SmallCNN` for 25 epochs in each condition. Test accuracy was 91.97% with real data alone, 92.29% with real and vanilla GAN samples, and 92.25% with real and WGAN-GP samples.

## Structure

```
Workshop_5/
├── src/
│   ├── data.py                 # Shared 70/15/15 split (seed 42), [-1, 1] GAN scaling & W2 scaling
│   ├── gan.py                  # Generator, Discriminator, Critic, compute_gradient_penalty (λ=10)
│   ├── train_gan.py            # Training loop for Vanilla GAN (BCE + label smoothing 0.9) & WGAN-GP (n_critic=5)
│   ├── evaluate_gan.py         # FID/FFD, Domain Inception Score (via W2 SmallCNN), diversity & failure modes
│   ├── generate_samples.py     # Generate 1,000–10,000 synthetic images & 4×4 / 8×8 side-by-side grids
│   └── augment_and_retrain.py  # 3-condition augmentation experiment retraining W2 SmallCNN
├── checkpoints/
│   ├── gan_vanilla_best.pt     # Best Vanilla GAN generator checkpoint
│   └── gan_wgan_best.pt        # Best WGAN-GP generator checkpoint
├── runs/workshop5/             # Training logs, epoch sample grids (every 10 epochs), metrics JSONs
├── notebooks/
│   ├── Training_Pipeline.ipynb # Documented end-to-end adversarial training flow (executed)
│   └── GAN_Evaluation.ipynb    # Quality metrics, failure modes and augmentation analysis (executed)
└── pyproject.toml              # Package metadata and dependencies
```

## Setup

```bash
cd Workshop_5
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Notebooks

Both notebooks use the same `src/` code as the CLI. The training notebook runs both GANs, and the evaluation notebook reads saved results.

| Notebook | What it walks through | Runtime |
|---|---|---|
| `notebooks/Training_Pipeline.ipynb` | Data in `[-1, 1]`, Generator/Discriminator/Critic, gradient penalty, Vanilla GAN and WGAN-GP demo training, sample grids, FFD/IS evaluation, synthetic-set construction, checkpoint verification, full CLI commands | ~2 min demo (MPS/CPU) |
| `notebooks/GAN_Evaluation.ipynb` | 100-epoch training curves, report quality table, real vs generated grids, mode-collapse/discriminator/WGAN-GP/checkerboard analysis, latent interpolation, 3-condition augmentation results | ~1 min (no retraining) |

`Training_Pipeline.ipynb` currently sets `DEMO = False`, which trains each GAN for 100 epochs. Set `DEMO = True` before running it for a two-epoch demonstration. Both modes put notebook training outputs in `runs/notebook_demo/` and `checkpoints/notebook_demo/`.

## Quickstart

```bash
# 1. Verify Generator, Discriminator, Critic, and Gradient Penalty
python -m src.gan

# 2. Train Vanilla GAN (100+ epochs, lr=2e-4, label smoothing=0.9, grids every 10 epochs)
python -m src.train_gan --model vanilla --epochs 100 --lr 2e-4 --label-smoothing 0.9

# 3. Train WGAN-GP (100+ epochs, n_critic=5, lambda_gp=10)
python -m src.train_gan --model wgan_gp --epochs 100 --n-critic 5 --lambda-gp 10.0

# 4. Generate 5,000 synthetic samples & 4×4 / 8×8 comparison grids
python -m src.generate_samples --vanilla-ckpt checkpoints/gan_vanilla_best.pt \
    --wgan-ckpt checkpoints/gan_wgan_best.pt --num-samples 5000

# 5. Evaluate generation quality (Fréchet Distance, Inception Score, diversity)
python -m src.evaluate_gan --vanilla-ckpt checkpoints/gan_vanilla_best.pt \
    --wgan-ckpt checkpoints/gan_wgan_best.pt --w2-ckpt ../Workshop_2/checkpoints/model_best.pt

# 6. Run 3-condition Data Augmentation Experiment (Real vs Real+Vanilla vs Real+WGAN-GP)
python -m src.augment_and_retrain --vanilla-ckpt checkpoints/gan_vanilla_best.pt \
    --wgan-ckpt checkpoints/gan_wgan_best.pt --num-synthetic 3000 --epochs 25
```

## Files required by `Workshop_5.pdf`

| Requirement | File / Path |
|---|---|
| Generator, Discriminator, Critic implementations | `src/gan.py` |
| Training loop for Vanilla GAN and WGAN-GP | `src/train_gan.py` |
| Generation quality evaluation and metrics (FID, IS) | `src/evaluate_gan.py` |
| Sample generation and 4×4 / 8×8 grid visualization | `src/generate_samples.py` |
| Data augmentation experiment (3 conditions) | `src/augment_and_retrain.py` |
| Best generator checkpoints | `checkpoints/gan_vanilla_best.pt`, `checkpoints/gan_wgan_best.pt` |
| Training logs and generated samples | `runs/workshop5/` |
| Report (PDF) | [Workshop 5 report](Workshop5.pdf) |
