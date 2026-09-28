# Workshop 1: Data Preparation & Supervised Learning Base Models

## 1. Dataset Description (Fashion-MNIST)

Fashion-MNIST is a dataset devolped by Zalando Research composed of 70,000 monocromatic images in a integer format of 8 bits (`uint8`), with a uniform resolution of 28x28 pixels (784 variables per sample). The dataset is divided in 10 categories of clothing products. It is originally distributed with 60,000 samples for training and 10,000 samples for testing.

### Category Mapping
| Label | Class |
| :---: | :--- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 2. Adquisition and Cryptographic Integrity Verification

The files are distributes compressed in a .gz file format. They are available in the official Zalando Research repository: [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).

### Official Download Links
- `train-images-idx3-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
- `train-labels-idx1-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
- `t10k-images-idx3-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
- `t10k-labels-idx1-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

### Check Sum & File size
| File | Size on Disk | Official Hash MD5 | Official Hash SHA-256 |
| :--- | :--- | :--- | :--- |
| `train-images-idx3-ubyte.gz` | 26,421,880 bytes (~25.2 MB) | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | 29,515 bytes (~28.8 KB) | `25c81989df183df01b3e8a0aad5dffbe` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | 4,422,102 bytes (~4.2 MB) | `bef4ecab320f06d8554ea6380940ec79` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | 5,148 bytes (~5.0 KB) | `bb300cfdad3c16e7a12a480ee83cd310` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |

To verify the integrity on the downloaded files:
```bash
python dataset_info.py
```

---

## 3. Repository Structure

The directory is structured as follows:

```
Workshop_1/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Binary IDX load, reproductible division and augmentations
│   ├── features.py             # Z-score scaling and dimensionality reduction with PCA
│   └── baselines.py            # Training, testing and base models exportation
├── notebooks/
│   └── SupervisedLearning.ipynb # Notebook with the complete EDA, visualizations and analysi
├── data/
│   ├── raw/                    # Compressed original binary files .gz
│   └── splits.json             # Stratified indexes of train (49k), val (10.5k), test (10.5k)
├── checkpoints/
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   └── Hist_Gradient_Boosting.joblib
├── runs/
│   └── results.json            # Detailed metricsMétricas detalladas, reportes y matrices de confusión
├── dataset_info.py              # Script CLI para verificación de integridad y estadísticas
├── pyproject.toml              # Dependencies definition and Poetry configuration
├── poetry.lock                 # Dependencies blocking
└── README.md                   # Technical documentation of the workshop
```

---

## 4. Environment Installation and Configuration

The project requires **Python 3.11**. The dependencies can be installed using Poetry or in a virtual environment using `pip`:

### Option A: Using Poetry
```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Option B: Using venv y pip
```bash
# Create the venv
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib scikit-learn scipy joblib
```

---

## 5. Reproductibility guide

### Step 1: Data validation and integrity
Execute the cryptographic and structural validation:
```bash
python dataset_info.py
```

### Step 2: Pipeline execution and training base models
The script `src/baselines.py` automatice data loading, the reading or generation of `data/splits.json`, the standard scaling (`StandardScaler`), projection with PCA (95% of explained variance, reducing to 256 dimensions), the adjustment of the 3 classifiers and metrics calculus:
```bash
python -m src.baselines
```

### Step 3: Stratified Partition (Train / Val / Test)
Based on the 70,000 images, a reproductible division was registered on `data/splits.json`:
- **Training (70%):** 49,000 samples (4,900 by class).
- **Validation (15%):** 10,500 samples (1,050 by class).
- **Test (15%):** 10,500 samples (1,050 by class).

The imbalance ratio is exactly 1.0, which avoid any bias by category frequency.

### Step 4: Exploration in Jupyter Notebook
To visualize the exploratory data analysis (EDA), acumulative variance of components and confusion matrixes, refer to `notebooks/eda.ipynb`

### Step 5: Model Training
To train the different models, refer to the following files: `notebooks/MulticlassLogisticRegression`, `notebooks/RandomForest` and `notebooks/HistGradientBoosting`