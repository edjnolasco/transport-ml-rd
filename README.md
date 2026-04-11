# transport-ml-rd

Machine Learning aplicado al dominio del transporte utilizando Support Vector Machines (SVM).

---

## 📌 Descripción

Este repositorio contiene la implementación de un algoritmo de clasificación basado en **Máquinas de Vectores de Soporte (SVM)**, desarrollado como parte de la asignatura **Algoritmos de Clasificación en Machine Learning**.

El proyecto aborda un caso representativo del sector transporte, donde se busca clasificar niveles de riesgo a partir de datos estructurados. La implementación incluye un pipeline completo de aprendizaje automático, abarcando desde el preprocesamiento hasta la evaluación del modelo.

---

## 🎯 Objetivos

- Implementar un modelo de clasificación supervisada utilizando SVM  
- Aplicar técnicas de preprocesamiento sobre datos estructurados  
- Evaluar el modelo mediante métricas estándar  
- Analizar aspectos como sobreajuste, interpretabilidad y coste computacional  
- Preparar la base para optimizaci贸n posterior (enfoque Green AI)  

---

## 🧠 Algoritmo

El modelo utilizado es:

**Support Vector Machine (SVM)**

Configuración principal:

- Kernel: RBF (Radial Basis Function)  
- Parámetro de regularización: C  
- Parámetro del kernel: gamma  

SVM se selecciona por:

- Su capacidad de maximizar el margen de separación entre clases  
- Su solidez teórica en problemas de clasificación  
- Su capacidad de modelar relaciones no lineales mediante funciones kernel  

---

## 📊 Dataset

El conjunto de datos representa un escenario del dominio del transporte e incluye variables como:

- Variables temporales (hora, d铆a)  
- Condiciones ambientales  
- Características de la vía  
- Información relacionada con vehículos  

La variable objetivo corresponde a una **clasificación binaria de riesgo**.

> Nota: El dataset se utiliza con fines académicos.

---

## ⚙️ Pipeline

El flujo de trabajo implementado es:

1. Carga de datos  
2. Preprocesamiento:
   - Imputación de valores faltantes  
   - Codificación de variables categóricas (One-Hot Encoding)  
   - Escalado de variables (StandardScaler)  
3. División en entrenamiento y prueba  
4. Entrenamiento del modelo (SVM)  
5. Evaluación del modelo  

---

## 📈 Métricas de Evaluación

El modelo se evalúa utilizando:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Matriz de confusi贸n  
- Curva ROC  

Los resultados se almacenan en:

- /reports/tables  
- /reports/figures  

---

## 🧪 Resultados

El pipeline genera automáticamente:

- Métricas de entrenamiento y prueba  
- Reportes de clasificaci贸n  
- Matriz de confusi贸n  
- Curva ROC  
- Perfil del dataset  
- Resumen de tiempos de ejecuci贸n  

---

## ⚠️ Análisis

Se abordan los siguientes aspectos:

### Sobreajuste
Comparación entre métricas de entrenamiento y prueba.

### Interpretabilidad
Limitaciones de SVM en la explicación directa de predicciones.

### Coste Computacional
Evaluación basada en el tiempo de entrenamiento e inferencia.

---

## 🌱 Trabajo Futuro (Práctica 3 - Green AI)

El proyecto está preparado para extenderse mediante:

- Variación del tamaño del dataset  
- Reducción del número de variables  
- Medición del tiempo de entrenamiento  
- Aplicación de técnicas como:
  - Selección de variables  
  - Reducción de dimensionalidad  
  - Optimización de hiperparámetros  

---

## 🛠️ Instalación

```bash
git clone https://github.com/your-username/transport-ml-rd.git
cd transport-ml-rd
pip install -r requirements.txt
```

---

## ▶️ Ejecución

```bash
python main.py
```

O con parámetros:

```bash
python src/svm_pipeline.py --kernel rbf --c 3.0 --gamma scale
```

---

## 📁 Estructura del Proyecto

```
transport-ml-rd/
│
├── .github/
├── data/
├── notebooks/
├── reports/
│   ├── tables/
│   └── figures/
├── src/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```

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

## 👤 Autor

**Edwin José Nolasco**

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
