#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:26:34 2021
Técnicas Avanzadas de Minería de 
Datos y Machine Learning

            Taller 3

Miller Alexander Quiroga Campos
@author: milleralexanderquirogacampos
"""

#Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import preprocessing # procesamiento de datos
from pandas_profiling import ProfileReport

from sklearn.linear_model import LinearRegression #Regresión Lineal
from sklearn.linear_model import LogisticRegression #Regresión Logistica

#%% importamos archivos de excel y csv y los convertimos en DataFrame

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Taller 3/")
cwd = os.getcwd()

#DataFrame Top100Startups
b = pd.read_excel('Top100Startups- Colombia.xlsx')#, index_col = 0)
#DataFrame ColombiaCB
a = pd.read_csv('ColombiaCB-5March21.csv')#, index_col = 0)

print(b.head(5))
print(b.tail(5))

print(a.head(5))
print(a.tail(5))

#Estadística de las variables cuantitativas
a.describe()

#%%

a1 = a.join(b, rsuffix='_right')

a1['FLag'] = 0
a1.shape

set(a1['Organization Name']).intersection(set(a1['Co']))
Int = set(a1['Organization Name']).intersection(set(a1['Co']))
#aja = set(a1['Organization Name']).intersection(set(a1['Co']))

set(a1['Contact Email']).intersection(set(a1['E-m']))

for j in Int:
    a1.loc[a1['Organization Name'] == j, ['Flag']] = 1

#%%

#### como hay discriminación de caracteres



a1["Organization Name"] = a1["Organization Name"].str.upper()
a1["Co"] = a1["Co"].str.upper()

set(a1['Organization Name']).intersection(set(a1['Co']))
Int = set(a1['Organization Name']).intersection(set(a1['Co']))

for i in Int:
    a1.loc[a1['Organization Name'] == i, ['Flag']] = 1


c =  a1.to_csv('/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/c.csv', index = False)


#%%

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase 5")
cwd = os.getcwd()

d = pd.read_excel('Empresas Unicorn - Contactos.xlsx')#, index_col = 0)
#DataFrame ColombiaCB
a

d1 = a.join(d, rsuffix='_right')

d1['FLag'] = 0

d1["Organization Name"] = d1["Organization Name"].str.upper()
d1["Name"] = d1["Name"].str.upper()

set(d1['Organization Name']).intersection(set(d1['Name']))
Int2 = set(d1['Organization Name']).intersection(set(d1['Name']))

d1['FLag2'] = 0

for k in Int2:
    d1.loc[d1['Organization Name'] == k, ['Flag2']] = 1
    
#%%

e1 = b.join(d, rsuffix='_right')

e1['FLag3'] = 0
    
e1["Co"] = e1["Co"].str.upper()
e1["Name"] = e1["Name"].str.upper()


set(e1['Co']).intersection(set(e1['Name']))
Int3 = set(e1['Co']).intersection(set(e1['Name']))

#%%

set(e1['Co']).intersection(set(e1['Name'])).intersection(set(a1['Organization Name']))
Int4 = set(e1['Co']).intersection(set(e1['Name'])).intersection(set(a1['Organization Name']))

F = a.join(b, rsuffix='_right')
F = F.join(d, rsuffix='_right')

F['FLag3'] = 0
    
F["Organization Name"] = F["Organization Name"].str.upper()
F["Co"] = F["Co"].str.upper()
F["Name"] = F["Name"].str.upper()

F['Intersection'] = 0

#data es el dataframe, values la variable a comparar
def Cotejar(data, values):
    if data in values:
        return 1
    else:
        return 0

a['cotejar_Or'] = a['Organization Name'].apply(Cotejar, values = d['Name'].tolist())

#%%     REGRESIÓN LÓGISTICA - 
#       CARACTERISTICAS COMUNES ENTRE LAS 12 COMPAÑIAS DE INTERSECCIÓN



