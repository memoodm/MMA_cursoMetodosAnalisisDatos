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

#%% importamos archivos de excel y csv y los convertimos en DataFrame

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Taller 3/")
cwd = os.getcwd()

b = 'Top100Startups- Colombia.xlsx'
a = 'ColombiaCB-5March21.csv'

b = pd.read_excel(b)#, index_col = 0)
a = pd.read_csv(a)#, index_col = 0)

print(b.head(15))
print(a.head(15))

print(b.tail(15))
print(a.tail(15))


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
z = set(a1['Organization Name']).intersection(set(a1['Co']))

for j in z:
    a1.loc[a1['Organization Name'] == j, ['Flag']] = 1

