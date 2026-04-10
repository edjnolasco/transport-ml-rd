# transport-ml-rd

Machine Learning aplicado al dominio del transporte utilizando Support Vector Machines (SVM).

---

## 📌 Descripción

Este repositorio contiene la implementación de un algoritmo de clasificación basado en **Máquinas de Vectores de Soporte (SVM)**, desarrollado como parte de la asignatura **Algoritmos de Clasificación en Machine Learning**.

El proyecto aborda un caso representativo del sector transporte, donde se busca clasificar niveles de riesgo a partir de datos estructurados. La implementación incluye un pipeline completo de aprendizaje automático, abarcando desde el preprocesamiento hasta la evaluación del modelo.

---

## 🎯 Objetivos

- Implementar un modelo de clasificaci贸n supervisada utilizando SVM  
- Aplicar t茅cnicas de preprocesamiento sobre datos estructurados  
- Evaluar el modelo mediante m茅tricas est谩ndar  
- Analizar aspectos como sobreajuste, interpretabilidad y coste computacional  
- Preparar la base para optimizaci贸n posterior (enfoque Green AI)  

---

## 🧠 Algoritmo

El modelo utilizado es:

**Support Vector Machine (SVM)**

Configuraci贸n principal:

- Kernel: RBF (Radial Basis Function)  
- Par谩metro de regularizaci贸n: C  
- Par谩metro del kernel: gamma  

SVM se selecciona por:

- Su capacidad de maximizar el margen de separaci贸n entre clases  
- Su solidez te贸rica en problemas de clasificaci贸n  
- Su capacidad de modelar relaciones no lineales mediante funciones kernel  

---

## 📊 Dataset

El conjunto de datos representa un escenario del dominio del transporte e incluye variables como:

- Variables temporales (hora, d铆a)  
- Condiciones ambientales  
- Caracter铆sticas de la v铆a  
- Informaci贸n relacionada con veh铆culos  

La variable objetivo corresponde a una **clasificaci贸n binaria de riesgo**.

> Nota: El dataset se utiliza con fines acad茅micos.

---

## ⚙️ Pipeline

El flujo de trabajo implementado es:

1. Carga de datos  
2. Preprocesamiento:
   - Imputaci贸n de valores faltantes  
   - Codificaci贸n de variables categ贸ricas (One-Hot Encoding)  
   - Escalado de variables (StandardScaler)  
3. Divisi贸n en entrenamiento y prueba  
4. Entrenamiento del modelo (SVM)  
5. Evaluaci贸n del modelo  

---

## 📈 Métricas de Evaluación

El modelo se eval煤a utilizando:

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

El pipeline genera autom谩ticamente:

- M茅tricas de entrenamiento y prueba  
- Reportes de clasificaci贸n  
- Matriz de confusi贸n  
- Curva ROC  
- Perfil del dataset  
- Resumen de tiempos de ejecuci贸n  

---

## ⚠️ Anólisis

Se abordan los siguientes aspectos:

### Sobreajuste
Comparaci贸n entre m茅tricas de entrenamiento y prueba.

### Interpretabilidad
Limitaciones de SVM en la explicaci贸n directa de predicciones.

### Coste Computacional
Evaluaci贸n basada en el tiempo de entrenamiento e inferencia.

---

## 🌱 Trabajo Futuro (Pr谩ctica 3 - Green AI)

El proyecto est谩 preparado para extenderse mediante:

- Variaci贸n del tama帽o del dataset  
- Reducci贸n del n煤mero de variables  
- Medici贸n del tiempo de entrenamiento  
- Aplicaci贸n de t茅cnicas como:
  - Selecci贸n de variables  
  - Reducci贸n de dimensionalidad  
  - Optimizaci贸n de hiperpar谩metros  

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

O con par谩metros:

```bash
python src/svm_pipeline.py --kernel rbf --c 3.0 --gamma scale
```

---

## 📁 Estructura del Proyecto

```
transport-ml-rd/
鈹?鈹溾攢鈹€ data/
鈹溾攢鈹€ reports/
鈹?  鈹溾攢鈹€ tables/
鈹?  鈹斺攢鈹€ figures/
鈹溾攢鈹€ src/
鈹溾攢鈹€ main.py
鈹溾攢鈹€ requirements.txt
鈹斺攢鈹€ README.md
```

---

## 📚 Referencias

- Cortes, C., & Vapnik, V. (1995). Support-vector networks.  
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.  
- Scikit-learn documentation: https://scikit-learn.org/  

---

## 👤 Autor

**Edwin Josó Nolasco**

---

## 馃嚭馃嚫 English Version

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
