## Detector de Fraude Crediticio

Este proyecto utiliza una red neuronal para detectar fraudes en transacciones de tarjetas de crédito. 
El modelo fue entrenado utilizando el conjunto de datos creditcard.csv de Kaggle.com, que contiene transacciones realizadas por titulares de tarjetas en Europa. 
El objetivo del proyecto es identificar patrones de fraude en un conjunto de datos desbalanceado y aplicar técnicas de preprocesamiento para mejorar la precisión del modelo.

### Contexto

Es crucial que las compañías de tarjetas de crédito puedan identificar transacciones fraudulentas para proteger a los consumidores y evitar que paguen por compras no realizadas. 
Este proyecto utiliza un conjunto de datos de transacciones con tarjetas de crédito, en el cual solo el 0.172% de las transacciones son fraudulentas.

### Dataset
- Fuente: Kaggle - Credit Card Fraud Detection
- Descripción: El conjunto de datos contiene transacciones realizadas en septiembre de 2013. Incluye 492 fraudes de un total de 284,807 transacciones.
- Características:
  - Las variables de entrada son transformadas con PCA (Análisis de Componentes Principales).
  - Tiempo: segundos transcurridos desde la primera transacción.
  - Monto: monto de la transacción.
  - Clase: variable de respuesta (1 para fraude, 0 para no fraude).
  - Debido al desbalance en las clases, se recomienda medir la precisión utilizando el área bajo la Precision-Recall Curve (AUPRC).

### Objetivo

El objetivo de este proyecto es desarrollar un modelo que detecte fraudes en las transacciones utilizando una red neuronal. 
Además, se compararán los resultados de la red neuronal con otros clasificadores para evaluar su desempeño.

### Descripción del Proyecto

#### Pasos para Resolver el Problema
1.  Exploración y Preparación de Datos:
    - Carga y análisis exploratorio: Importación del dataset y análisis inicial de sus características, distribución y valores faltantes
    - Tratamiento de valores atípicos: Identificación y manejo de outliers mediante técnicas estadísticas como z-score
    - Preprocesamiento: Normalización/estandarización de variables numéricas para mejorar el rendimiento de los algoritmos
2. Manejo del Desbalance de Clases:
    - Implementación de técnicas de submuestreo (NearMiss) y sobremuestreo (SMOTE) para equilibrar la distribución entre transacciones fraudulentas y legítimas
    - Evaluación del impacto de estas técnicas en el rendimiento de los modelos
3. Selección de Características:
    - Análisis de importancia de variables mediante información mutua y otros métodos
    - Eliminación de características redundantes o poco relevantes para mejorar la eficiencia
4. Modelado y Evaluación:
    - Modelos tradicionales: Implementación de Random Forest y Regresión Logística
    - Redes neuronales: Desarrollo de arquitecturas con capas densas y dropout para prevenir sobreajuste
    - Validación cruzada: Uso de StratifiedKFold para evaluar la robustez de los modelos
    - Métricas de evaluación: Análisis mediante matrices de confusión, classification report y curvas ROC para evaluar el rendimiento con énfasis en la detección de fraudes
5. Optimización y Ajuste
    - Refinamiento de hiperparámetros para maximizar métricas relevantes (precisión, recall, F1-score)
    - Comparación sistemática entre diferentes enfoques para seleccionar el modelo final
6. Conclusión
    - Posibles mejoras finales

### Cómo usarlo

1. Clona este repositorio en tu máquina local:
    - git clone https://github.com/Marcelo-Huichulef/fraud-detection-using-deep-learning.git
2. Navega al directorio del proyecto:
    - cd fraud-detection-using-deep-learning
3. Instala las dependencias necesarias:
    - pip install -r requirements.txt
4. Abre el archivo Detector_Fraude_Crediticio.ipynb en Jupyter Notebook:
    - jupyter notebook Detector_Fraude_Crediticio.ipynb
Dentro del notebook, sigue las instrucciones para cargar los datos, preprocesarlos, entrenar los modelos y evaluar su desempeño.

### Dependencias

Este proyecto requiere las siguientes librerías:

- Pandas
- NumPy
- SciPy (incluyendo funciones estadísticas y zscore)
- Matplotlib
- Seaborn
- Scikit-learn (para preprocesamiento, validación, modelado y evaluación, incluyendo StandardScaler, train_test_split, RandomForestClassifier, LogisticRegression, entre otros)
- Imbalanced-learn (por ejemplo, SMOTE y NearMiss para balancear la distribución de clases)
- TensorFlow y Keras (para la construcción y entrenamiento de redes neuronales)

Puedes instalar todas las dependencias con:
- pip install -r requirements.txt

### Resultados

El modelo de red neuronal es capaz de detectar transacciones fraudulentas con alta precisión, a pesar de la naturaleza desbalanceada de los datos. 
Las métricas de evaluación mostraron que la red neuronal superó a otros clasificadores en términos de precisión y capacidad para identificar fraudes reales.
