# Machine Learning

This repository contains the information and retrievals of the workshops from the Machine Learning class.

## Team Members
- Jhojan Stiven Aragón Ramírez - 20221020060
- Santiago Reyes Gómez - 20221020098
- Juan Andrés Jiménez Palomino - 20221020087

## Problem's Domain
The domain where this implementations of AI solutions revolves around is Computer Vision oriented to Image Classification.

Across the different workshops, there will be different approaches to solve such problems, each folder of the repository contains the necessary information to fully understand the requirements, the approach of the solution and its corresponding conclussion analysis.

---

## Workshops Structure

### [Workshop 1: Data Preparation & Supervised Learning Base Models](Workshop_1/README.md)
- **Domain:** Image classification to the dataset of Fashion-MNIST. This dataset is composed of 10 classes, with 70000 monocromatic images of 28x28 pixels.
- **Reach:**
  - Download and verification of cryptographic integrity (hashes MD5 and SHA-256).
  - Exploratory Data Analysis (EDA): Luminic intensity stats, class balance (*Imbalance Ratio* = 1.0), avg images and 2D projections using PCA.
  - Processing Pipeline: Reproducible stratified partition (70% train / 15% val / 15% test en `data/splits.json`), z-score standarization (`StandardScaler`), augmented data in training (horizontal flip and traslation) and dimensionality reduction using PCA (95% cumulative variance).
  - Supervised Learning Base Models: Multiclass Logistic Regression, Random Forest & HistGradientBoosting.
  - Performance Evaluation: Accuracy, Precision, F1-Score & One vs. All Confusion Matrixes. To evaluate general performance and per class.
- **To seek complete reproductibility:** Go to [Workshop_1/README.md](Workshop_1/README.md).