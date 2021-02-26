# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:03:28 2020

@author: LENOVO
"""
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

import pandas as pd # Libreria para analisis de Datos#
import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns # graficas y estadisticas , maquillaje#
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

os.chdir("C:/Users/LENOVO/Documents/case11/")

cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
excel_file="dataventa.xlsx" #asignacion dle nombre del archivo a una bandeja #

df=pd.read_excel(excel_file)


# Load data/Carga Las variables
#dat = sm.datasets.get_rdataset("data").data

vars = ['QuantySold', 'Price', 'Advertising']

df = df[vars]

df[-7:]
df = df.dropna()
df[-7:]

#Calculo de Minimos Cuadrados
y, X = dmatrices('QuantySold ~ Price + Advertising', data=df, return_type='dataframe')

y[:7]
X[:7]

mod = sm.OLS(y, X)    # Describe model
res = mod.fit()       # Fit model
print(res.summary())   # Summarize model

#parametros de la regresion
res.params
#coeficiente de determinacion:
res.rsquared

sm.stats.linear_rainbow(res)
print(sm.stats.linear_rainbow.__doc__)

#Grafica de Correlacion
sm.graphics.plot_partregress('QuantySold', 'Price', 'Advertising',data=df, obs_labels=False)



