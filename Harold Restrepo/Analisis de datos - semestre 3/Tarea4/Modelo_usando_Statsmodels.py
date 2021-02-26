# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 03:44:14 2020

@author: Harold Ricardo
"""
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


os.chdir('C:/Users/Harold Ricardo/ModuloAnalisDeDatos/clase5/') # Asignacion de ruta de trabajo
cwd=os.getcwd()                  # Asigna al a variable cwd el directorio de trabajo
excel_file="price.csv" 

data =pd.read_csv(excel_file)

#Variables independientes
X = data.drop(['quantity_sold'], axis = 1)

# Variable dependiente
y = data['quantity_sold']

# Creación del modelo 
X = sm.add_constant(X, prepend=True)
modelo = sm.OLS(endog=y, exog=X,)
modelo = modelo.fit()
print(modelo.summary())
