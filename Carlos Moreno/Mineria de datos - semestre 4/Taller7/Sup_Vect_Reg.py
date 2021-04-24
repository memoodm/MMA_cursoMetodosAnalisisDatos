#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:03:04 2021

@author: milleralexanderquirogacampos
"""

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import os

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn import preprocessing # procesamiento de datos
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.svm import SVR

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%% importamos archivos de excel y csv y los convertimos en df


os.chdir('/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase6/')
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo

datos = pd.read_csv('Position_Salaries.csv')

datos.describe()

print(datos.head(5))
print(datos.tail(5))

print(datos.columns)


#Suport Vector Regresion

#%%

X = datos.iloc[:,1:-1].values
y = datos.iloc[:,-1].values

X
y

#Cambiar las dimensiones de y de una dimensión a 2 dimensiones
y = y.reshape(len(y),1)

y

#Estandarizamos las variables
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_X.fit_transform(y)

X
y


regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#Si hay errores
regressor.fit(X,y.ravel())

#finalizamos todo.., ahora hacemos la predicción


#supongamos que hay un argumento de 6,5 millones
regressor.predict([[6.5]])
#realizamos la transformación inversa. - 
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

#Grafica de Dispersión
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('uju')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.grid()
plt.show()

#%% - Regresión polinomial

X_p = datos.iloc[:,1:-1].values
y_p = datos.iloc[:,-1].values

X_p
y_p

#Cambiar las dimensiones de y de una dimensión a 2 dimensiones
y_p = y_p.reshape(len(y_p),1)

y_p

#Graficar los datos x_p y y_p
plt.scatter(X_p, y_p)
plt.grid()
plt.show()

from sklearn.model_selection import train_test_split

#separo los datos de train en entrenamiento y prueba para probar los algoritmos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size = 0.2)

#Se agregan las características polinomiales de
from sklearn.preprocessing import PolynomialFeatures

#Se define el grado del polinomio, se puede empezar desde 2 y luego subir hasta ver el ajuste
#polig_reg = PolynomialFeatures(degree = 2)
polig_reg = PolynomialFeatures(degree = 2)


#se transforma las características existentes en características de mayor grado
X_train_poli = polig_reg.fit_transform(X_train_p)
X_test_poli  = polig_reg.fit_transform(X_test_p)

from sklearn import datasets, linear_model
#Defino el algoritmo a utilizar
pr = linear_model.LinearRegression()

#Entrenamos el modelo
pr.fit(X_train_poli, y_train_p)

#realizo una predicción
Y_pred_pr = pr.predict(X_test_poli)

#Graficamos los datos junto con modelo
plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_pr, color = 'red', linewidth=3)
plt.grid()
plt.show()


#Calculamos los coeficientes del polinomio
print()
print('Datos del Modelo REGRESIÓN POLINOMIAL')
print()

print('Valor de la pendiente o del cofeiciente a :')
print(pr.coef_)

print('Valor de la intesección o del cofeiciente b :')
print(pr.intercept_)

#Calculamos la precisión del algoritmo - r^2

print('Precision del modelo')
print(pr.score(X_train_poli, y_train_p)*100)


#Se obtiene una precisión del 93.74%, esto indica que este modelo se ajusta para el 




