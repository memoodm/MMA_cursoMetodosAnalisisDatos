# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:18:26 2020

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:56:07 2020

@author: Diogenes Barreto
"""

import pandas as pd # Libreria para analisis de Datos#
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split# divide en DF en subconjuntos de entrenamientos aleatorio
from sklearn.utils import resample

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns # graficas y estadisticas , maquillaje#
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)


os.chdir("C:/Users/LENOVO/Documents/Clase13/")

cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
text_file='Winequality.csv' #asignacion dle nombre del archivo a una bandeja #

df= pd.read_csv('Winequality.csv', sep= ";")


print(df.head())#imprime las primeras 10#
print(df.tail()) #imprime las primeras 10#

print(df.shape)
df.head()

df.isnull().any()

df.describe() #estadistica descriptiva de la variables

from pandas_profiling import ProfileReport # Reportes de variables
ProfileReport(df)
profile = ProfileReport(df, title="Pandas Profiling Report",explorative=True)
profile = ProfileReport(df)
profile.to_file('profile_report.html')

sns.boxplot(df['total sulfur dioxide'])# Distribucion de Variables

pd.crosstab(df.pH,df.quality).plot(kind='bar')
df.plot.scatter(x='citric acid', y='quality')
plt.title('Purchase Frequency for duration ')
plt.xlabel('ph')
plt.ylabel('Frequency of Quality')
plt.savefig('purchase_fre_cons_price_idx')

plt.subplots(figsize=(15, 10))
df.columns
df.corr#Matriz de Correlacion
sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")#Matriz de correlacion
plt.show()

from scipy import stats # normalizacion de valores , eleimino datos atipicos retirados de la media
z = np.abs(stats.zscore(df))
df2= df[(z < 3).all(axis=1)]
df2.shape

df2.describe() #estadistica descriptiva de la variables
plt.subplots(figsize=(15, 10))
df2.columns
df2.corr#Matriz de Correlacion
sns.heatmap(df2.corr(), annot = True, cmap = "coolwarm")#Matriz de correlacion
plt.show()

#Procesamiento de Data

df2["quality"].value_counts()#Cantidad de vector Calidad (Cluster)

#Definir las características y el objetivo

X = np.asarray(df2.iloc[:,:-1]) # Define Caractristica del Vector X
# Define target y
y = np.asarray(df2["quality"]) #Define el objetivo Vector y

# Estandarizar el conjunto de datos

X = preprocessing.StandardScaler().fit(X).transform(X)

#Train and test sets split
#Entrenamiento y prueba -divison-#Divider el grupo de entrenamiento 70%y gruppo de prueba 30% balanceo

#X = df.loc[:,df.columns != 'quality'] #Vector de Vectores, Columna var Pred#
#y = df.loc[:,df.columns == 'quality']# Vector Respuesta


from imblearn.over_sampling import SMOTE
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
print ("Train set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)


smt = SMOTE(random_state=0) #Fija la Semilla
data_X,data_y=smt.fit_sample(X_train, y_train)

sns.countplot(x='quality',data=data_y)
plt.title('Count vs quality ')
plt.show()


#Validacion y seleccion del Modelo, 
#En esta parte entrena el algoritmos de clasificación para encontrar el mejor  conjunto de datos que usé

#K-Nearest Neighbors

#Gráfica de precisión de KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# Number of k from 1 to 35
k_range = range(1, 35)

k_scores = []

# Calcular la puntuación de validación cruzada para cada número k de 1 a 35
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k) 
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy") 
    k_scores.append(scores.mean())

# Grafica de Precision para cada Valor de K entre 1  y 35
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()

#Entrenamiento del modelo KKN y prediccion para K=23.(Tomomaos el valor de k=23 , se estabuliza comportamiento de K)

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)
print(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Informe de clasificación del conjunto de pruebas

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits=3, zero_division = 1))


# Calcule la puntuación del CV con la puntuación de "precisión" y 10 veces más

accuracy = cross_val_score(knn, X, y, scoring = 'accuracy',cv=10)
print('cross validation score',accuracy.mean())

# Calcular la puntuación del CV con la puntuación de 'roc_auc_ovr' y 10 pliegues

accuracy = cross_val_score(knn, X, y, scoring = 'roc_auc_ovr',cv=10)
print('cross validation score with roc_auc',accuracy.mean())

# Calcular la puntuación de roc_auc con el parámetro de multiclase
print('roc_auc_score',roc_auc_score(y_test,knn.predict_proba(X_test), multi_class='ovr'))


#Prueba:
    
    
datos= {'fixed acidity': [8500],'volatile acidity':[4700],'citric acid':[5800],'residual sugar':[7400],
        'chlorides':[6200],'free sulfur dioxide':[7300],'total sulfur dioxide':[5600],'density':[0.9982],
                'pH':[2],'sulphates':[5],'alcohol':[10.8]}

X_test = pd.DataFrame(datos,columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol'])
  
y_pred = knn.predict(X_test)
print(y_pred)
   
    











