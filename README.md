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
1. Preprocesamiento de los Datos:
    - Análisis exploratorio de los datos y visualización de su distribución.
    - Uso del algoritmo NearMiss para crear un subconjunto balanceado de datos con una proporción 50/50 entre fraudes y no fraudes.
2. Entrenamiento del Modelo:
    - Se entrena una red neuronal profunda para detectar fraudes en las transacciones.
    - Se comparan los resultados con otros clasificadores como el árbol de decisiones y el modelo de regresión logística.
3. Evaluación del Modelo:
    - El rendimiento del modelo se mide utilizando el AUPRC (Área bajo la curva Precision-Recall), debido al desbalance de clases.
4. Resultados:
    - Comparación de la precisión de la red neuronal con otros clasificadores.
    - Visualización de las métricas de evaluación y las curvas ROC y Precision-Recall.

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

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- imbalanced-learn (para NearMiss)

Puedes instalar todas las dependencias con:
- pip install -r requirements.txt

### Resultados

El modelo de red neuronal es capaz de detectar transacciones fraudulentas con alta precisión, a pesar de la naturaleza desbalanceada de los datos. 
Las métricas de evaluación mostraron que la red neuronal superó a otros clasificadores en términos de precisión y capacidad para identificar fraudes reales.
