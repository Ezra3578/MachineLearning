# Machine Learning (Team 3)

Code, checkpoints, training logs, and reports for the Machine Learning course project at Universidad Distrital Francisco José de Caldas (semester 2026-III).

## Team

- Jhojan Stiven Aragón Ramírez (`20221020060`)
- Santiago Reyes Gómez (`20221020068`)
- Juan Andrés Jiménez Palomino (`20221020087`)

## Dataset

We use Fashion-MNIST for image classification and the related reconstruction and generation tasks. It has `70,000` grayscale `28×28` images across `10` balanced classes. The workshop experiments use a stratified 70/15/15 split with `seed = 42`.

## Workshops

| Folder | Topic | Models and results |
|---|---|---|
| [`Workshop_1/`](Workshop_1/) | Classical supervised learning | Logistic regression, random forest (`85.90%`) and HistGradientBoosting (`87.48%` test accuracy, `0.8740` macro F1) |
| [`Workshop_2/`](Workshop_2/) | Convolutional neural networks | `SmallCNN` trained from scratch (`93.05%`, `0.9302` macro F1) and ImageNet-pretrained `ResNet-18` (`94.94%`, `0.9494` macro F1), with Grad-CAM analysis |
| [`Workshop_3/`](Workshop_3/) | Sequential classification with reinforcement learning | `FashionMNISTClassifyEnv` (`gymnasium`) and Double DQN (`85.20%` accuracy over test episodes, mean return `70.39`) |
| [`Workshop_4/`](Workshop_4/) | Autoencoders and VAEs | `ConvAE`, `ConvVAE` ($\beta \in \{0.5, 1.0, 4.0\}$), and `DenoisingConvAE`. The best reconstruction reached `26.11 dB` PSNR, and the best anomaly score reached `0.9530` ROC-AUC. Latent features are compared with W2 using t-SNE and UMAP. |
| [`Workshop_5/`](Workshop_5/) | Generative adversarial networks | `VanillaGAN` and `WGAN-GP` ($\lambda=10, n_{\text{critic}}=5$), evaluated with domain Fréchet distance and Inception Score. Adding vanilla GAN samples to the W2 classifier's training data improved accuracy by `0.32` percentage points over the 25-epoch real-only run. |
