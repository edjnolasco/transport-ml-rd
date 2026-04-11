# transport-ml-rd

[![CI](https://github.com/edjnolasco/transport-ml-rd/actions/workflows/ci.yml/badge.svg)](https://github.com/edjnolasco/transport-ml-rd/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange)
![Green AI](https://img.shields.io/badge/Green%20AI-Enabled-brightgreen)
![Lint](https://img.shields.io/badge/lint-ruff-blue)
![Tests](https://img.shields.io/badge/tests-xUnit%20%7C%20pytest-success)
![Coverage](https://raw.githubusercontent.com/edjnolasco/transport-ml-rd/main/coverage.svg)
![Colab](https://img.shields.io/badge/Colab-Notebook-orange)

Machine Learning aplicado al dominio del transporte utilizando Support Vector Machines (SVM) y análisis de eficiencia bajo el enfoque **Green AI**.

---

## 📌 Descripción

Este repositorio implementa un pipeline completo de machine learning dividido en dos fases:

- **Práctica 2:** modelo base con Support Vector Machines (SVM)
- **Práctica 3:** extensión hacia análisis de eficiencia computacional (Green AI)

El proyecto no solo evalúa desempeño predictivo, sino también el costo computacional asociado a cada configuración de modelo.

---

## 📓 Notebooks del proyecto

El proyecto incluye dos notebooks principales que representan la evolución del análisis:

### 🔹 Práctica 2 – Modelo Base (SVM)

Implementación inicial del pipeline de clasificación utilizando Support Vector Machines.

- Preprocesamiento de datos
- Entrenamiento con SVM (kernel RBF)
- Evaluación con métricas clásicas

👉 Ejecutar en Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w8aF5fVg4bdfoBsr6w0-9BQhnkPvYCa7?usp=sharing)

---

### 🔹 Práctica 3 – Green AI y Trade-off

Extensión del análisis hacia eficiencia computacional:
- Comparación entre múltiples modelos
- Estrategias de optimización
- Análisis de trade-off (F1 vs tiempo)
- Frontera de Pareto
- Detección automática del mejor modelo

👉 Ejecutar en Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18ifzCj4s-Kq-NQEJlLTLI25IR3rDIsfS?usp=sharing)

## 🎯 Objetivos

- Implementar modelos de clasificación supervisada (SVM y otros)
- Aplicar técnicas de preprocesamiento sobre datos estructurados
- Evaluar modelos mediante métricas estándar
- Analizar sobreajuste, interpretabilidad y coste computacional
- Evaluar el **trade-off entre desempeño y eficiencia (Green AI)**

---

## 🧠 Práctica 2 – Modelo Base (SVM)

### Algoritmo

**Support Vector Machine (SVM)**

Configuración principal:
- Kernel: RBF  
- Parámetro de regularización: C  
- Parámetro del kernel: gamma  

Se selecciona por:
- Maximización del margen de separación  
- Capacidad para modelar relaciones no lineales  
- Base teórica sólida  

---

## 🌱 Práctica 3 – Enfoque Green AI

Se extiende el análisis hacia eficiencia computacional, evaluando:
- tiempo de entrenamiento  
- número de features (`n_features`)  
- tamaño de muestra (`sample_fraction`)  
- impacto de estrategias de optimización  

### Estrategias evaluadas

- Regularización  
- Selección de variables (SelectKBest)  
- Reducción de profundidad en árboles  
- Reducción del tamaño de muestra  

---

## 📊 Análisis de Trade-off

Se construye una figura tipo paper que representa:
- F1-score ponderado vs tiempo de entrenamiento  
- Diferenciación por modelo (color)  
- Diferenciación por estrategia (marcador)  
- **Frontera de Pareto**  
- **Detección automática del mejor modelo global**

Los resultados se exportan automáticamente:

```text
reports/figures/
├── figure_1_tradeoff.png
├── figure_1_tradeoff.pdf
├── figure_1_tradeoff.svg
├── figure_1_tradeoff_metadata.json

```
---

🗺️ Normalización territorial

Problema detectado:
- 63 valores únicos ❌
- 32 unidades reales ✅

Solución:
- normalización de texto
- eliminación de tildes
- unificación semántica

---

⚙️ Pipeline

El flujo de trabajo implementado es:

- Carga de datos
- Limpieza y preprocesamiento
- Normalización territorial
- Ingeniería de características
- Entrenamiento de modelos
- Evaluación de métricas
- Análisis Green AI

---

📄 Documentación completa:

👉 Data Pipeline

📈 Métricas de Evaluación
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Matriz de confusión

---

🧪 Resultados

El pipeline genera automáticamente:
- métricas de entrenamiento y prueba
- reportes de clasificación
- matriz de confusión
- curva ROC
- tiempos de entrenamiento
- comparación entre configuraciones

---

⚠️ Análisis
- Sobreajuste
- Comparación entre métricas de entrenamiento y prueba.
- Interpretabilidad
- Limitaciones de modelos como SVM.
- Coste Computacional
- Evaluación del tiempo de entrenamiento como métrica clave en Green AI.

---

## 📊 Visualización del Trade-off (Green AI)

La figura muestra el compromiso entre desempeño (F1-score) y costo computacional (tiempo de entrenamiento), incorporando:
- diferenciación por modelo (color)
- diferenciación por estrategia (marcador)
- frontera de Pareto
- mejor modelo global

![Trade-off](reports/figures/figure_1_tradeoff.png)


## 📁 Estructura del Proyecto

```text
transport-ml-rd/
│
├── docs/
│   └── data_pipeline.md
│
├── notebooks/
│   ├── practica2_transporte_svm_colab.ipynb
│   └── practica3_transporte_greenai_colab.ipynb
│
├── reports/
│   └── figures/
│       ├── figure_1_tradeoff.png
│       ├── figure_1_tradeoff.pdf
│       └── figure_1_tradeoff.svg
│
├── src/
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
│
├── tests/
│   ├── test_smoke.py
│   ├── test_territorial_units.py
│   ├── test_dataset_integrity.py
│   └── test_plots.py
│
├── .github/
│   └── workflows/
│       └── ci.yaml
│
├── README.md
└── requirements.txt
```
## 🧠 Arquitectura del Proyecto

Separación clara entre:
- lógica → `src/`  
- experimentación → `notebooks/`  
- resultados → `reports/`  
- validación → `tests/`  

---

## ⚙️ Componentes

### 📓 Notebooks
- ejecución del pipeline  
- análisis  
- visualización  

### 🧩 `src/visualization`
- generación de figuras tipo paper  
- funciones reutilizables  
- backend compatible con CI  

### 📊 `reports/`
- figuras exportadas  
- consumo desde README  

### 🧪 `tests/`
- validación de pipeline  
- validación de datos (32 provincias)  
- pruebas de visualización  

### 🔁 CI/CD
- tests automáticos  
- coverage  
- artifacts  

---

## 🔄 Flujo del sistema

```text
Notebook
   ↓
Preprocesamiento
   ↓
Modelado
   ↓
Evaluación
   ↓
src.visualization.plots
   ↓
Exportación de figuras
   ↓
reports/figures
   ↓
README / CI
```

⚙️ Instalación

git clone https://github.com/your-username/transport-ml-rd.git
cd transport-ml-rd
pip install -r requirements.txt

---

▶️ Ejecución

python main.py

---

## ▶️ Ejecutar en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18ifzCj4s-Kq-NQEJlLTLI25IR3rDIsfS?usp=sharing)

---

📌 Conclusión

El proyecto demuestra que:
- el desempeño del modelo no debe evaluarse de forma aislada
- el coste computacional es un factor clave
- el análisis de trade-offs permite seleccionar modelos más eficientes
- la calidad de los datos (ej. normalización territorial) impacta directamente los resultados

---

👤 Autor

Edwin José Nolasco

---

## 📚 Referencias

- Cortes, C., & Vapnik, V. (1995). Support-vector networks.  
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.  
- Scikit-learn documentation: https://scikit-learn.org/
- Awad, M., & Khanna, R. (2015). Support vector machines for classification. In Efficient learning machines (pp. 39–66). Apress. https://doi.org/10.1007/978-1-4302-5990-9_3
- Cervantes, J., García-Lamont, F., Rodríguez-Mazahua, L., & López, A. (2020). A comprehensive survey on support vector machine classification: Applications, challenges and trends. Neurocomputing, 408, 189–215. https://doi.org/10.1016/j.neucom.2019.10.118
- Guido, R. (2024). An overview on the advancements of support vector machines in medical applications. Information, 15(4), 235. https://doi.org/10.3390/info15040235
- Khyathi, G., Prasad, K., & Reddy, K. (2025). Support vector machines: A literature review on their application in analyzing mass data for public health. Cureus, 17(1), e77169. https://doi.org/10.7759/cureus.77169 
- Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI. Communications of the ACM, 63(12), 54–63. https://doi.org/10.1145/3381831
- Tang, W. (2024). Application of support vector machine system introducing cluster-based kernel methods. Machine Learning with Applications, 15, 100525. https://doi.org/10.1016/j.mlwa.2024.100525

---

## English Version

### Description

This repository contains a classification model based on **Support Vector Machines (SVM)**, developed as part of a Machine Learning course.

The project focuses on a transport-related scenario, where the objective is to classify risk levels using structured data. A full machine learning pipeline is implemented, including preprocessing, training, and evaluation.

---

### Objectives

- Implement a supervised classification model using SVM  
- Apply preprocessing techniques to structured data  
- Evaluate model performance using standard metrics  
- Analyze overfitting, interpretability, and computational cost  
- Prepare the project for Green AI optimization  

---

### Algorithm

- Model: Support Vector Machine (SVM)  
- Kernel: RBF  
- Hyperparameters: C, gamma  

---

### Evaluation

Metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve  

---

### Future Work

- Dataset size variation  
- Feature reduction  
- Runtime analysis  
- Model optimization techniques  

---

### Notes

This repository is developed strictly for academic purposes.
