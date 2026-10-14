# Machine Learning — Team 3 Repository

This repository contains the implementations, checkpoints, training logs, and reports for the **Machine Learning (Semester 2026-III)** course project at Universidad Distrital Francisco José de Caldas.

## Members (Team 3)
- Jhojan Stiven Aragón Ramírez — `20221020060`
- Santiago Reyes Gómez — `20221020068`
- Juan Andrés Jiménez Palomino — `20221020087`

## Problem Domain
**Computer Vision & Object Recognition — Image Classification on Fashion-MNIST** (`70,000` grayscale `28×28` images across `10` balanced classes, stratified **70/15/15** split with `seed = 42`).

## Workshops Overview

| Folder | Workshop Topic | Key Models & Deliverables | Status |
|---|---|---|---|
| `Workshop_1/` | Supervised Learning (Classical Baselines) | Logistic Regression (`85.33%`), Random Forest (`85.90%`), HistGradientBoosting (`87.48%` test acc / `0.8740` F1) | ✅ Completed |
| `Workshop_2/` | Deep Learning (Convolutional Neural Networks) | `SmallCNN` from scratch (`93.05%` acc / `0.9302` F1) & ImageNet `ResNet-18` transfer (`94.94%` acc / `0.9494` F1) + Grad-CAM | ✅ Completed |
| `Workshop_3/` | Reinforcement Learning (Sequential Classification) | `FashionMNISTClassifyEnv` (`gymnasium`) + Double-DQN (`85.20%` test acc, mean return `70.39`) | ✅ Completed |
| `Workshop_4/` | Unsupervised Learning (Autoencoders & VAEs) | `ConvAE`, $\beta$-`ConvVAE` ($\beta \in \{0.5, 1.0, 4.0\}$), `DenoisingConvAE`, semantic anomaly detection, t-SNE/UMAP vs W2 | 🚧 In Progress (`workshop4-5`) |
| `Workshop_5/` | Generative Adversarial Networks (GAN & WGAN-GP) | `VanillaGAN`, `WGAN-GP` ($\lambda=10, n_{\text{critic}}=5$), FID/IS quality metrics, 3-condition W2 data augmentation | 🚧 In Progress (`workshop4-5`) |
