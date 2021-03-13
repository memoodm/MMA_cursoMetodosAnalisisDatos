#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 22:27:18 2021

@author: milleralexanderquirogacampos
"""

####        IMPORTAR LIBRERÍAS QUE SE VAN A UTILIZAR PARA EL PROGRAMA


import pandas as pd # Libreria para analisis de Datos#
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)

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

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)

#%%         LECTURA DE LA DATA

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase3")
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#

#           PAISES LATINOAMERICANOS
c1 ='ColombiaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c2 ='ChileCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c3 ='BrazilCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c4 ='ArgentinaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c5 ='MexicoCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c6 ='UruguayCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #

co = pd.read_csv(c1, index_col= 0)
ch = pd.read_csv(c2, index_col= 0)
br = pd.read_csv(c3, index_col= 0)
ar = pd.read_csv(c4, index_col= 0)
mx = pd.read_csv(c5, index_col= 0)
ur = pd.read_csv(c6, index_col= 0)

#           Paises Europa, Asia y NortAmerica

c7 = 'SpainCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c8 = 'GermanyCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c9 = 'SwitzerlandCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c10 = 'IsraelCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
c11 = 'USACB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #

es = pd.read_csv(c7, index_col= 0)
ge = pd.read_csv(c8, index_col= 0)
sw = pd.read_csv(c9, index_col= 0)
Is = pd.read_csv(c10, index_col= 0)
us = pd.read_csv(c11, index_col= 0)

#%%         LIMPIEZA DE LA DATA DE NULL Y NAN

#ARMA UNA LISTA DE LOS 11 ARCHIVOS
datos = [co, ch, br, ar, mx, ur, es, ge, sw, Is, us]

#VERIFICA SI HAY DATOS NULOS EN CADA VARIABLE DE CADA ARCHIVO
for i in datos:
    print(i.isnull().sum())

#DIMENSIÓN DE CADA VARIABLE DE LAS DATA
for i in datos:
   print(i.shape)
   
#IMPRIME LOS 11 PRIMEROS INVERSIONES DE CADA DATA FRAME
for i in datos:
    i["Last Funding Amount"].head(11).plot(kind="barh")

for k in datos:
    k["Last Funding Amount"].plot(kind="hist")
    plt.grid(True)
    plt.show

for j in datos:
    pd.crosstab(j["Organization Name"].head(10),i["Last Funding Amount"].head(10)).plot(kind='bar')

#%%         UNIMOS TODAS LAS BASES DE DATOS A UNA SOLA

df = pd.concat([co, ch, br, ar, mx, ur, es, ge, sw, Is, us])
print(df.shape)

#%% - - - - Eliminar variables que no aportan nada
df.drop(['Organization Name URL'], axis=1)

#%%

df.info()

#identificación de variables (distribución de probabilidad) - Variables numéricas
df.hist();#identificación de variables (distribución de probabilidad)

df["Number of Lead Investments"].value_counts(normalize=True)
df["Number of Lead Investments"].unique()

df["Closed Date"].plot(kind ="bar")

df["Closed Date"].value_counts(normalize=True)

df.shape

#%%
#Grafica de barras
#df.bar();
#que hay en la variable
df['Closed Date'].dtypes
#identificación de variable
df['Closed Date'].unique

#%%

df1 = df['Headquarters Location'].str.split(",", n = 3, expand = True) 

df["Ciudad"]= df1[0]
df["Departamento"]= df1[1]
df["Pais"]= df1[2]

#%%
df = df.drop(['Organization Name URL'],axis=1)
df.columns

#%%

#Organiza las variables en forma ascendente
df_sorted_by_cb_rank = df.sort_values(["CB Rank (Company)"], ascending = False)
df_sorted_by_Number_of_Employees = df.sort_values(["Number of Employees"], ascending = True)
df_sorted_by_Estimated_Revenue_Range = df.sort_values(["Estimated Revenue Range"], ascending = True)
df_sorted_by_Number_of_Articles = df.sort_values(["Number of Articles"], ascending = True)
df_sorted_by_Investment_Stage = df.sort_values(["Investment Stage"], ascending = True)
df_sorted_by_Number_of_Funding_Rounds = df.sort_values(["Number of Funding Rounds"], ascending = True)
df_sorted_by_Last_Funding_Amount = df.sort_values(["Last Funding Amount"], ascending = True)
df_sorted_by_Last_Funding_Type = df.sort_values(["Last Funding Type"], ascending = True)
df_sorted_by_Last_Equity_Funding_Amount = df.sort_values(["Last Equity Funding Amount"], ascending = True)
df_sorted_by_Number_of_Investments = df.sort_values(["Number of Investments"], ascending = False)


#%%
############# GRAFICAS
pd.crosstab(df_sorted_by_Last_Equity_Funding_Amount["Last Equity Funding Amount"].head(10),df_sorted_by_Last_Equity_Funding_Amount["Last Funding Type"].head(10)).plot(kind="barh")
plt.xlabel('Last Equity Funding Amount')
plt.ylabel('Last Funding Type')
plt.title('Comparación de dos variables')
plt.show()


pd.crosstab(df_sorted_by_Number_of_Articles["Number of Articles"].head(50),df_sorted_by_Number_of_Articles["Number of Funding Rounds"].head(50)).plot(kind="barh")
plt.title('Number of Articles vs Number of Funding Rounds ')
plt.xlabel("Number of Articles")
plt.ylabel("Number of Funding Rounds")
plt.savefig('')








