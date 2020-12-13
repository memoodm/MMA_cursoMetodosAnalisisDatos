# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 02:56:56 2020

@author: Harold Ricardo
"""
# Import libraries
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn import metrics

os.chdir('C:/Users/Harold Ricardo/ModuloAnalisDeDatos/clase5/') # Asignacion de ruta de trabajo
cwd=os.getcwd()                  # Asigna al a variable cwd el directorio de trabajo
excel_file="price.csv" 

data =pd.read_csv(excel_file)

data

# Variables independientes
X = data.drop(['quantity_sold'], axis = 1)

# Variable dependiente
y = data['quantity_sold']

# Crear el modelo
lr = LinearRegression()

# Ajustar el modelo
lr.fit(X, y)

# Hacer predicciones
pred = lr.predict(X)

print('Intercepto:', lr.intercept_)

coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coeficientes'])  
print(coeff_df)

print('Error Medio Absoluto:', metrics.mean_absolute_error(y, pred))  
print('Error medio cuadrático:', metrics.mean_squared_error(y, pred))  
print('Raiz del error medio cuadrático:', np.sqrt(metrics.mean_squared_error(y, pred)))
print('R2:', np.sqrt(metrics.r2_score(y, pred)))

