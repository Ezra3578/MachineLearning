# Taller 1: Preparación de Datos y Modelos Base de Aprendizaje Supervisado

Repositorio para el primer taller práctico del curso de Aprendizaje Automático (Semestre 2026-III), Universidad Distrital Francisco José de Caldas.

## Integrantes
- Jhojan
- Santiago Reyes Gómez - 20221020098
- Juan Andrés Jiménez Palomino - 20221020087

---

## 1. Descripción del Conjunto de Datos (Fashion-MNIST)

Fashion-MNIST es un conjunto de datos desarrollado por Zalando Research compuesto por 70,000 imágenes monocromáticas en formato entero de 8 bits (`uint8`), con resolución uniforme de 28x28 píxeles (784 variables por muestra). El conjunto está dividido en 10 categorías de productos de vestuario y calzado, distribuido originalmente en 60,000 muestras para entrenamiento y 10,000 para prueba.

### Mapeo de Categorías
| Etiqueta | Clase Oficial (Inglés) | Descripción en Español |
| :---: | :--- | :--- |
| 0 | T-shirt/top | Camiseta / Top |
| 1 | Trouser | Pantalón |
| 2 | Pullover | Suéter / Jersey |
| 3 | Dress | Vestido |
| 4 | Coat | Abrigo |
| 5 | Sandal | Sandalia |
| 6 | Shirt | Camisa |
| 7 | Sneaker | Zapatilla deportiva |
| 8 | Bag | Bolso / Cartera |
| 9 | Ankle boot | Bota al tobillo |

---

## 2. Adquisición y Verificación de Integridad Criptográfica

Los archivos se distribuyen comprimidos mediante gzip en formato binario IDX. Se encuentran disponibles en el repositorio oficial de Zalando Research: [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist).

### Enlaces Oficiales de Descarga
- `train-images-idx3-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
- `train-labels-idx1-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
- `t10k-images-idx3-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
- `t10k-labels-idx1-ubyte.gz`: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

### Sumas de Comprobación y Tamaño de Archivos
| Archivo | Tamaño en Disco | Hash MD5 Oficial | Hash SHA-256 Oficial |
| :--- | :--- | :--- | :--- |
| `train-images-idx3-ubyte.gz` | 26,421,880 bytes (~25.2 MB) | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` |
| `train-labels-idx1-ubyte.gz` | 29,515 bytes (~28.8 KB) | `25c81989df183df01b3e8a0aad5dffbe` | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` |
| `t10k-images-idx3-ubyte.gz` | 4,422,102 bytes (~4.2 MB) | `bef4ecab320f06d8554ea6380940ec79` | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` |
| `t10k-labels-idx1-ubyte.gz` | 5,148 bytes (~5.0 KB) | `bb300cfdad3c16e7a12a480ee83cd310` | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` |

Para verificar la integridad de los archivos descargados:
```bash
python dataset_info.py
```

---

## 3. Estructura del Repositorio

El directorio de trabajo está estructurado según los lineamientos del taller:

```
Workshop_1/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carga de binarios IDX, división reproducible y aumentaciones
│   ├── features.py             # Escalado z-score y reducción de dimensionalidad con PCA
│   └── baselines.py            # Entrenamiento, evaluación y exportación de modelos base
├── notebooks/
│   └── SupervisedLearning.ipynb # Cuaderno interactivo con EDA, pipeline y modelos base
├── data/
│   ├── raw/                    # Archivos binarios comprimidos .gz originales
│   └── splits.json             # Índices estratificados de train (49k), val (10.5k), test (10.5k)
├── checkpoints/
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   └── Hist_Gradient_Boosting.joblib
├── runs/
│   └── results.json            # Métricas detalladas, reportes y matrices de confusión
├── dataset_info.py              # Script CLI para verificación de integridad y estadísticas
├── pyproject.toml              # Definición de dependencias y configuración Poetry
├── poetry.lock                 # Bloqueo de dependencias
└── README.md                   # Documentación técnica del taller
```

---

## 4. Instalación y Configuración del Entorno

El proyecto requiere **Python 3.11**. Las dependencias pueden instalarse utilizando Poetry o mediante un entorno virtual con `pip`:

### Opción A: Usando Poetry (Recomendado)
```bash
# Instalar dependencias
poetry install

# Activar el entorno virtual
poetry shell
```

### Opción B: Usando venv y pip
```bash
# Crear el entorno virtual
python -m venv venv

# Activar en Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activar en Linux/macOS
source venv/bin/activate

# Instalar dependencias
pip install numpy pandas matplotlib scikit-learn scipy joblib
```

---

## 5. Guía de Reproducción de Experimentos

### Paso 1: Validación de Datos e Integridad
Ejecutar la comprobación criptográfica y estructural:
```bash
python dataset_info.py
```

### Paso 2: Ejecución del Pipeline y Entrenamiento de Modelos Base
El script `src/baselines.py` automatiza la carga de datos, la lectura o generación de `data/splits.json`, el escalado estandarizado (`StandardScaler`), la proyección por PCA (95% de varianza explicada, reduciendo a 256 dimensiones), el ajuste de los 3 clasificadores y el cálculo de métricas:
```bash
python -m src.baselines
```

### Paso 3: Exploración Interactiva en Jupyter Notebook
Para visualizar el análisis exploratorio de datos (EDA), la varianza acumulada de componentes y las matrices de confusión:
```bash
jupyter notebook notebooks/SupervisedLearning.ipynb
```

---

## 6. Resultados y Discusión Técnica

### Partición Estratificada (Train / Val / Test)
A partir del universo de 70,000 imágenes, se implementó una división reproducible registrada en `data/splits.json`:
- **Entrenamiento (70%):** 49,000 muestras (4,900 por clase).
- **Validación (15%):** 10,500 muestras (1,050 por clase).
- **Prueba (15%):** 10,500 muestras (1,050 por clase).

La tasa de desbalance de clases (*Imbalance Ratio*) es exactamente **1.0**, lo que elimina cualquier sesgo por frecuencia de categorías.

### Rendimiento Comparativo de los Modelos Base
A continuación se resumen los resultados obtenidos sobre los conjuntos de validación y prueba (con fijación de semilla aleatoria `seed = 42`):

| Modelo | Tiempo de Entrenamiento | Exactitud (Val) | Macro F1 (Val) | Exactitud (Test) | Macro F1 (Test) | AUC-ROC Test (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 38.6 s | 85.15% | 0.8503 | 85.31% | 0.8519 | 0.9837 |
| **Random Forest (100 árboles)** | 12.5 s | 86.59% | 0.8637 | 85.90% | 0.8565 | 0.9866 |
| **HistGradientBoosting** | 31.2 s | **88.30%** | **0.8825** | **87.48%** | **0.8740** | **0.9899** |

### Análisis Morfológico de Confusión
- **Alta separabilidad en calzado y accesorios:** Las categorías *Trouser* (F1 = 0.966), *Sandal* (F1 = 0.941), *Bag* (F1 = 0.932), *Sneaker* (F1 = 0.940) y *Ankle boot* (F1 = 0.923) muestran un agrupamiento distante en el espacio de características debido a siluetas nítidas y fondos contrastantes.
- **Concentración de error en prendas superiores:** Más del 70% de los errores se concentran entre *Shirt*, *T-shirt/top*, *Pullover* y *Coat*. La clase *Shirt* (camisa) presenta el menor desempeño individual (F1 = 0.716), confundiéndose mutuamente con camisetas y abrigos debido a la similitud estructural de mangas y torso en baja resolución (28x28 píxeles).
- **Conclusión metodológica:** Los modelos lineales y ensambles tradicionales alcanzan un techo de precisión cercano al 88%. Superar esta cota requerirá modelos de Deep Learning basados en capas convolucionales (CNN) en los siguientes talleres, capaces de capturar correlaciones espaciales locales e invariancia a pequeñas traslaciones.
