# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:51:01 2021

@author: LENOVO
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:33:00 2021

@author: LENOVO
"""
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

os.chdir("C:/Users/LENOVO/Documents/Clase5/")

#Region
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv_file='ColombiaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
excel_file='Top100Startups- Colombia.xlsx' #asignacion dle nombre del archivo a una bandeja #

df=pd.read_csv(csv_file)
top100= pd.read_excel(excel_file, index_col=0)

#realizo el marge de las dos datas ne un DF
df_marge = df.join(top100, rsuffix='_right')
print(df_marge)
df_marge.shape

#asignacion de Mayusaculas a la lista de las variables
df_marge["Organization Name"]=df_marge["Organization Name"].str.upper()
df_marge['Organization']=df_marge['Organization'].str.upper()

#Realizo la inteccion en las Columnas para saber cual estan entan en la misma data 
intersec=set(df_marge['Organization Name']).intersection(set(df_marge['Organization']))
len(intersec)
for elemento in intersec:
    print(elemento)

df_marge ["same Organization"]=0

for i in intersec:
   df_marge.loc[df_marge['Organization Name'] == i, ["same Organization"]] = 1


# if set(df_marge['Organization Name'])==set(df_marge['Organization']):
#    set(df_marge["same Organization"])==1
# else:
#    set(df_marge["same Organization"])==0
