# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:19:19 2020

@author: Julian
"""

import pandas as pd
import numpy as np
import sklearn.linear_model as LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd

Q =  np.array([8500,4700,5800,7400,6200,7300,5600])
P = np.array([2,5,3,2,5,3,4])
A = np.array([2800,200,400,500,3200,1800,900])

X_multiple = pd.DataFrame({"P":P,"A": A})

print(X_multiple.describe())

y_multiple = Q

from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2)

#Defino el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

#Entreno el modelo
lr_multiple.fit(X_train, y_train)


#Realizo una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

print('DATOS DEL MODELO REGRESIÓN LINEAL MULTIPLE')
print()
print('Valor de las pendientes o coeficientes "a":')
print(lr_multiple.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr_multiple.intercept_)

print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))

X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.OLS(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())







