# Machine Learning (Team 3)

Code, checkpoints, training logs, and reports for the Machine Learning course project at Universidad Distrital Francisco José de Caldas (semester 2026-III).

## Team

- Jhojan Stiven Aragón Ramírez (`20221020060`)
- Santiago Reyes Gómez (`20221020098`)
- Juan Andrés Jiménez Palomino (`20221020087`)

## Datasets

- **Fashion-MNIST (W1–W5)** has `70,000` grayscale `28×28` images across `10` balanced classes. These workshops use a stratified 70/15/15 split with `seed = 42`.
- **PneumoniaMNIST (W6)** contains pediatric chest X-rays for binary pneumonia classification. W6 uses the dataset's predefined train, validation, and test splits.

## Workshops

| Folder | Topic | Models and results |
|---|---|---|
| [`Workshop_1/`](Workshop_1/) | Classical supervised learning | Logistic regression, random forest (`85.90%`) and HistGradientBoosting (`87.48%` test accuracy, `0.8740` macro F1) .See the [report](Workshop_1/Workshop1.pdf).|
| [`Workshop_2/`](Workshop_2/) | Convolutional neural networks | `SmallCNN` trained from scratch (`93.05%`, `0.9302` macro F1) and ImageNet-pretrained `ResNet-18` (`94.94%`, `0.9494` macro F1), with Grad-CAM analysis.See the [report](Workshop_2/Workshop2.pdf). |
| [`Workshop_3/`](Workshop_3/) | Sequential classification with reinforcement learning | `FashionMNISTClassifyEnv` (`gymnasium`) and Double DQN (`85.20%` accuracy over test episodes, mean return `70.39`).See the [report](Workshop_3/Workshop3.pdf). |
| [`Workshop_4/`](Workshop_4/) | Autoencoders and VAEs | `ConvAE`, `ConvVAE` ($\beta \in \{0.5, 1.0, 4.0\}$), and `DenoisingConvAE`. The best reconstruction reached `26.11 dB` PSNR, and the best anomaly score reached `0.9530` ROC-AUC. Latent features are compared with W2 using t-SNE and UMAP.See the [report](Workshop_4/Workshop4.pdf). |
| [`Workshop_5/`](Workshop_5/) | Generative adversarial networks | `VanillaGAN` and `WGAN-GP` ($\lambda=10, n_{\text{critic}}=5$), evaluated with domain Fréchet distance and Inception Score. Adding vanilla GAN samples to the W2 classifier's training data improved accuracy by `0.32` percentage points over the 25-epoch real-only run.See the [report](Workshop_5/Workshop5.pdf). |
| [`Workshop_6/`](Workshop_6/) | Medical image classification | We adapt the W2 `SmallCNN` to PneumoniaMNIST and compare frozen feature reuse, fine-tuning, and training from scratch. The class-weighted model selected on validation reached `0.9770` test ROC-AUC. See the [report](Workshop_6/Workshop6.pdf). |
