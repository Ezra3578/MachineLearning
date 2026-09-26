# Aprendizaje Automático (Machine Learning)

Repositorio que reúne los talleres prácticos, código y documentación desarrollados durante el curso de Aprendizaje Automático en la Universidad Distrital Francisco José de Caldas.

## Integrantes
- Jhojan
- Santiago Reyes Gómez - 20221020098
- Juan Andrés Jiménez Palomino - 20221020087

## Dominio del Problema
Los desarrollos del curso se concentran en tareas de **Visión por Computador (Computer Vision)** orientadas a la clasificación de imágenes. A lo largo del semestre se abordan distintas metodologías y paradigmas de modelado, organizados de forma modular por taller.

---

## Estructura de Talleres

### [Taller 1: Preparación de Datos y Modelos Base de Aprendizaje Supervisado](Workshop_1/README.md)
- **Dominio:** Clasificación de imágenes sobre el conjunto de datos Fashion-MNIST (10 clases, 70,000 muestras monocromáticas de 28x28 píxeles).
- **Alcance:**
  - Descarga y verificación de integridad criptográfica (hashes MD5 y SHA-256).
  - Análisis Exploratorio de Datos (EDA): estadísticas de intensidad lumínica, balance de clases (*Imbalance Ratio* = 1.0), imágenes promedio y proyecciones 2D con PCA.
  - Pipeline de preprocesamiento: partición estratificada reproducible (70% train / 15% val / 15% test en `data/splits.json`), estandarización z-score (`StandardScaler`), aumento de datos en entrenamiento (volteo horizontal y traslación) y reducción de dimensionalidad con PCA (95% varianza acumulada).
  - Modelos base supervisados: Regresión Logística, Random Forest e HistGradientBoosting.
  - Evaluación integral: Exactitud, Macro F1, Matrices de confusión normalizadas y análisis de error sistemático en prendas superiores (*Shirt*, *T-shirt*, *Pullover*, *Coat*).
- **Documentación completa y reproducción:** Consulte [Workshop_1/README.md](Workshop_1/README.md).