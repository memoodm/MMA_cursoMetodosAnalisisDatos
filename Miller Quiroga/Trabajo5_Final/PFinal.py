#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:21:36 2020

@author: milleralexanderquirogacampos
"""
from scipy import stats
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import os #sistema op
from pandas_profiling import ProfileReport
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


os.chdir('/Users/milleralexanderquirogacampos/Documents/Documentos - MacBook Pro de Miller/MaestriaMatematicas/3 - Metodos de Analisis de datos/LuzStellaGomez/Clase7')
cwd = os.getcwd() # asigan variable swd al directorio
csv_file = "Winequality.csv"

#df = pd.read_csv(r'/Users/milleralexanderquirogacampos/Documents/Documentos - MacBook Pro de Miller/MaestriaMatematicas/3 - Metodos de Analisis de datos/LuzStellaGomez/Clase7',sep=';')

df = pd.read_csv(csv_file, sep=';')

pd.read_csv('Winequality.csv',sep=';', header=0)
hh = df.describe()

df.hist()
plt.show()
print(df.groupby("density").size())

profile = ProfileReport(df, title='Pandas Profiling Report', explorative = True)
profile.to_file(output_file="output.html")

print(df.shape)
df.head()

#Me indica si hay valores faltantes en la data
df.isnull().any()

#observo las variables del dataset
df.columns

#de acuerto a la estadística observo en un diagrama de caja un valor atípico de la variable total sulfur dioxide
sns.boxplot(df['total sulfur dioxide'])

#Se ha utilizado la función zscore() definida en la biblioteca de SciPy y se ha establecido el umbral=3. Después de eliminar los valores atípicos
z = np.abs(stats.zscore(df))
df2 = df[(z < 3).all(axis=1)]
df2.shape


#grafico de correlación
plt.subplots(figsize=(15, 10))
sns.heatmap(df2.corr(), annot = True, cmap = 'coolwarm')

#se comprueba el desequilibrio de las clases
df2['quality'].value_counts()

# Se Define las caracteristicas de x
X = np.asarray(df2.iloc[:,:-1])
# Se Define el objetivo y
y = np.asarray(df2['quality'])

#Estandarización del dataset
X = preprocessing.StandardScaler().fit(X).transform(X)



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
print ("Train set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)


smt = SMOTE(random_state=0) #Fijar la Semilla
data_X,data_y = smt.fit_sample(X_train, y_train)

sns.countplot(x='Calidad',data = data_y)
plt.title('Conteo vs Calidad')
plt.show()


#División del tren y equipos de prueba
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)


# Número de k de 1 a 101
k_range = range(1, 101)
k_scores = []

# Calcular la puntuación de validación cruzada para cada número k de 1 a 101
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #Validación cruzada de 10 veces con una puntuación de "precisión". 
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Precisión de la trama para cada número k entre 1 y 101
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.grid()
#plt.show()


# Entrenar el modelo y predecir para k = 39
knn = KNeighborsClassifier(n_neighbors = 39)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# Informe de clasificación para el conjunto de pruebas
print(metrics.classification_report(y_test, y_pred, digits=3, zero_division = 1))

# Calcular la puntuación del CV con la puntuación de "precisión" y 10 veces más
accuracy = cross_val_score(knn, X, y, scoring = 'accuracy',cv=10)
print('cross validation score',accuracy.mean())

# Calcular la puntuación del CV con la puntuación de 'roc_auc_ovr' y 10 pliegues
accuracy = cross_val_score(knn, X, y, scoring = 'roc_auc_ovr',cv=10)
print('cross validation score with roc_auc',accuracy.mean())

# Calcula la puntuación de roc_auc con el parámetro de multiclase
print('roc_auc_score',roc_auc_score(y_test,knn.predict_proba(X_test), multi_class='ovr'))


"""
Cuando miro el informe de clasificación veo inmediatamente 
que las clases 4 y 8 no se han tenido en cuenta en el 
entrenamiento porque sus resultados de recuerdo son cero. 
Esto significa que, de todos los miembros de la clase 4 y 8, 
no predijo ninguno de ellos correctamente. Por lo tanto, 
no sería un buen modelo para nuestro conjunto de datos.
"""

#FINAL FINAL ---- NO DA MÁS

