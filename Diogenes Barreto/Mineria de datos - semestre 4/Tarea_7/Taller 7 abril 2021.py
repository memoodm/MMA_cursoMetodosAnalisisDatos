# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:03:35 2021

@author: LENOVO
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:33:00 2021

@author: LENOVO
"""

#Support Vector Regression


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
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)
from countryinfo import CountryInfo

os.chdir("C:/Users/LENOVO/Documents/Clase8/")

#Region
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv1_file='Position_Salaries.csv' #asignacion dle nombre del archivo a una bandeja #


dataset =pd.read_csv(csv1_file)

X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

# se hace un reshape.para cambiar las dimensiones de un arreglo a multiples dimensiones

y=y.reshape(len(y),1)
print(y)

# estandarizamos los datos.

from sklearn.preprocessing import StandardScaler


sc_X = StandardScaler()
sc_y = StandardScaler()

X= sc_X.fit_transform(X)
y= sc_y.fit_transform(y)
print(X)
print(y)

X.std()
y.std()

#crear vector suppor vector regration

from sklearn.svm import SVR # Clase necsita algunos parametros,Kernel

regressor = SVR(kernel="rbf")
regressor.fit(X,y.ravel())

print(regressor)

# Reescala un nivel de 6.5
sc_X.transform([[6.5]])

# predice el salario rescalado del nivel reescalado
regressor.predict(sc_X.transform([[6.5]]))

# Prediccion de un nuevo resultado usando transformacion inversa
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualizando los resultados

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') 
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue', label ='rbf')
plt.title('(SVR)')
plt.xlabel('Nivel o Cargo Laboral')
plt.ylabel('Salario')
plt.show()

regressor = SVR(kernel="rbf")
regressor.fit(X,y.ravel())

#%%% Polinomial
regressor = SVR(kernel="poly")
regressor.fit(X,y.ravel())
print(regressor)

# Reescala un nivel de 6.5
sc_X.transform([[6.5]])

# predice el salario rescalado del nivel reescalado
regressor.predict(sc_X.transform([[6.5]]))

# Prediccion de un nuevo resultado usando transformacion inversa
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualizando los resultados

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') 
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue', label ='poly')
plt.title('(POLY)')
plt.xlabel('Nivel o Cargo Laboral')
plt.ylabel('Salario')
plt.show()



