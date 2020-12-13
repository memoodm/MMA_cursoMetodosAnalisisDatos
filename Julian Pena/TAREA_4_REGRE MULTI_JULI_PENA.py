# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 07:07:05 2020

@author: Julian
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
Q =  np.array([8500,4700,5800,7400,6200,7300,5600])
P = np.array([2,5,3,2,5,3,4])
A = np.array([2800,200,400,500,3200,1800,900])



df=pd.DataFrame({"QuantySold":Q,"Price":P,"Advertising": A})


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
