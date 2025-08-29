    #1.1 Usando la libreria de drive

#Importamos las librerias que usaremos en este practica

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

#Importamos la libreria de drive

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Cople el archivo a su Gdrive desde: https://raw.githubusercontent.com/armandoordonez/eda_couse/main/data/diabetes-NAN.csv
     
     #1.2 Cargando el archivo a google colab
     
# Modificamos la ruta de acuerdo a su Gdrive
#Usamos pd (la libreria de pandas, que fue definida en la celda donde importamos las librerias) para leer el dataset y asignarlo a df

df = pd.read_csv('/content/drive/MyDrive/datasets/diabetes-NAN.csv',    #Ruta completa a la ubicación del archivo de interés.

     na_values='?')     

#Como el archivo contiene algunos caracteres ?, con este comando pandas los interpreta como nan (not a number).
   
# 2. Observaciones basicas del dataset
# Tamaño del dataset

df.shape   

#información general -- Podemos observar el nombre de las columnas, si tienen nulos y cual es su tipo de dato

df.info()

# Vista preliminar Observamos los primeros y ultimos 5 datos del data set

df.head()
df.tail()

# 3.1 Clasificación con datos nulos

#Escojamos nuestras variables predictoras
predictores = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#Escojamos que es lo que queremos encontrar
objetivo= 'Outcome'

#Recordemos que en nuestra variable Y va lo que queremos encontrar/predecir.
#Mientras en la vairable X van las variables que usaremos para encontrar a Y
y= df[objetivo].values
X= df[predictores].values

from sklearn.model_selection import train_test_split
# Crear conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#la a regresión logisitica no puede lidear con valores nulos


# Se importa el modelo
from sklearn.linear_model import LogisticRegression

#Se crea el modelo
log_reg = LogisticRegression(solver='lbfgs',max_iter=100)
#se entrena el modelo
log_reg.fit(X_train, y_train)

# LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, 
# you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. 
#Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. 
# See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: 
# https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
# Usamos otro modelo

from sklearn.ensemble import (HistGradientBoostingClassifier)

hgbm = HistGradientBoostingClassifier(random_state=42)
hgbm.fit(X_train, y_train)
hgbm.score(X_test, y_test)

# 3.2 Clasificación sin datos nulos
#veamos cuantos nan tenemos en columnas

df.isna().sum()

#veamos cuantos nan tenemos en total
df.isna().sum().sum()

# Podemos ver, que solo 42 de las 768 observaciones tienen valores nulos, así que en este caso borraremos esos datos
df_clean=df.dropna()
df_clean.isna().sum()

# Ahora que tenemos un dataset limpio, repetimos el proceso

#Escojamos nuestras variables predictoras
predictores = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#Escojamos que es lo que queremos encontrar
objetivo= 'Outcome'

#Recordemos que en nuestra variable Y va lo que queremos encontrar/predecir.
#Mientras en la vairable X van las variables que usaremos para encontrar a Y
y= df_clean[objetivo].values
X= df_clean[predictores].values

# Crear conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#Se crea el modelo
log_reg = LogisticRegression(solver='lbfgs',max_iter=1000)
#se entrena el modelo
log_reg.fit(X_train, y_train)

LogisticRegression(max_iter=1000)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

log_reg.score(X_test, y_test)

# Se puede observar que ya obtuvimos un mejor resultado sin nulos usando regresión logistica.
# Dicho esto, usaremos el hgbm de nuevo para una mejor comparación

hgbm = HistGradientBoostingClassifier(random_state=42)
hgbm.fit(X_train, y_train)
hgbm.score(X_test, y_test)

# Como se puede ver, este modelo tambien presenta una mejora en su score.

# Ejercicio: Utiliza otros modelos que soporten nulos para la predicción y evalua su score
     

