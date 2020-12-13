# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:23:02 2020

@author: Rectoria
"""

import pandas as pd #importamos librerías para analisis de datos
import numpy as np #librería para análisis numérico
import matplotlib.pyplot as plt #Librería de visualización
import seaborn as sns #graficas más bonitas y estadística
import os #Librería para el sistema operativo
from sklearn import preprocessing # procesamiento de datos
from pandas_profiling import ProfileReport

plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#ruta del archivo donde esta alojado
os.chdir ("D:\Desktop\DOCUEMNTOSMAESTRIA")
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo
excel_file = "Tarea4.xlsx" # Asignación del nombre del archivo a una bandera

df = pd.read_excel(excel_file)


import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.formula.api as smf


#Cargamos la data
#df2 = sm.datasets.get_rdataset("QuantySold", "Price", "Advertising").data #Advertising

vars = ["QuantySold", "Price", "Advertising"]

df = df[vars]
df[-3:]

df = df.dropna()
df[-7:]

y, X = dmatrices('QuantySold ~ Price + Advertising', data=df, return_type='dataframe')

y[:3]
X[:3]


mod = sm.OLS(y, X)    # Describe model

res = mod.fit()       # Fit model

print(res.summary())   # Summarize model

res.params

res.rsquared

sm.stats.linear_rainbow(res)


sm.graphics.plot_partregress('QuantySold', 'Price', 'Advertising', data = df, obs_labels=False)
