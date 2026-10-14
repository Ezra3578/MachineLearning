# Workshop 5 — Generative Adversarial Networks (GAN & WGAN-GP) for Fashion-MNIST

**Team 3** — Computer vision and object recognition.  
Jhojan Stiven Aragón Ramírez (20221020060) · Santiago Reyes Gómez (20221020068) · Juan Andrés Jiménez Palomino (20221020087)

Adversarial generative modeling on Fashion-MNIST comparing **Vanilla DCGAN** and **Wasserstein GAN with Gradient Penalty (`WGAN-GP`)** (plus optional **Conditional GAN (`cGAN`)**), evaluated via quantitative domain metrics (Fréchet Fashion-MNIST Distance / FID, Domain Inception Score, class entropy, and intra-class diversity), failure mode analysis, and a 3-condition **data augmentation experiment** retraining the Workshop 2 `SmallCNN` classifier.

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
├── notebooks/                  # Interactive notebooks
└── pyproject.toml              # Package metadata and dependencies
```

## Setup

```bash
cd Workshop_5
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

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
    --wgan-ckpt checkpoints/gan_wgan_best.pt --num-synthetic 2500 --epochs 30
```

## Deliverables Mapping (`Workshop_5.pdf`)

| Requirement | File / Path |
|---|---|
| Generator, Discriminator, Critic implementations | `src/gan.py` |
| Training loop for Vanilla GAN and WGAN-GP | `src/train_gan.py` |
| Generation quality evaluation and metrics (FID, IS) | `src/evaluate_gan.py` |
| Sample generation and 4×4 / 8×8 grid visualization | `src/generate_samples.py` |
| Data augmentation experiment (3 conditions) | `src/augment_and_retrain.py` |
| Best generator checkpoints | `checkpoints/gan_vanilla_best.pt`, `checkpoints/gan_wgan_best.pt` |
| Training logs and generated samples | `runs/workshop5/` |
| Report (PDF) | `tex-team3-machine-learning/Workshop_5/main.pdf` (`group3_workshop5.pdf`) |
