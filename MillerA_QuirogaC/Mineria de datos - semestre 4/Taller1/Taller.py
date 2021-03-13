#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 06:13:40 2021
Métodos Avanzados de Minería de Datos y Machine Learning
Doc: Luz Stella Gómez Fajardo Ph.D
Taller 1
Universidad Sergio Arboleda
@author: milleralexanderquirogacampos
"""

#importación de librerías
from scipy import stats
from sklearn import metrics
from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os #sistema op
from sklearn import preprocessing

os.chdir('/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase1')
cwd=os.getcwd() # asigan variable swd al directorio
excel_file = "Colombia-Feb21.xlsx" #asigno archivo

df = pd.read_excel(excel_file, header= 0)
#df = pd.read_excel(excel_file, sep=';')

#Elimino la fila con el keyword drop (se elimina Organization Name URL)
df = df.drop(['Organization Name URL'], axis=1)

#Remplazo las palabras con caracteres especiales
df= df.replace({"BogotÃ¡, Distrito Especial, Colombia":"Bogotá","MedellÃ­n, Antioquia, Colombia":"Medellín", "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén"})

#elimino la columna index para que en las gráficas no me aparezca el numero de la fila
df = pd.read_excel(excel_file, index_col=0)

#una forma muy conveniente, filtrar los valores de una estructura de 
#datos pandas para dejar solo aquellos no nulos.
df0=df.dropna()

df0["Last Funding Amount"].plot(kind="hist")
plt.show()
df0["Last Funding Amount"].plot(kind="bar")
plt.show()

df0.iloc[:, 8]


sorted_by_gross = df0.sort_values(['Last Funding Amount'], ascending=False)
print(sorted_by_gross.head(10))


sorted_by_gross['Last Funding Amount'].head(10).plot(kind="pie")
plt.show()




