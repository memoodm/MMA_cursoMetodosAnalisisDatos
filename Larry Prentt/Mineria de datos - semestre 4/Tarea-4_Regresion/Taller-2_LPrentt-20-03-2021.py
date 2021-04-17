# -*- coding: utf-8 -*-
"""
Created on Saturday Mar 20 11:00:00 2021

@author: larry Prentt
"""
#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.chdir("D:/4to_Semestre/Mineria de datos/python/Tercer Archivo")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

# C_csv_file = 'ColombiaCB-5March21.csv'
# 1 = Colombia

df=pd.read_csv('ColombiaCB-5March21.csv')
df["Organization Name"]=df["Organization Name"].str.lower()

#df=pd.read_excel(xls_file, header=0,sep=';', index_col=0)
xls_file = 'Top100Startups- Colombia.xlsx'
dftop=pd.read_excel(xls_file)
dftop["Organization"]=dftop["Organization"].str.lower()

xls_file2 = 'Empresas Unicorn - Contactos.xlsx'
dfunicorn = pd.read_excel(xls_file2)
dfunicorn["Name"]=dfunicorn["Name"].str.lower()

# Ojo No usar df1 = df

df1 = df.copy()
df2 = df.copy()

################################################################################
################## CrunchBase vs Top 100
df["y"]=0
df["Flag"]=0

intersect=set(df['Organization Name']).intersection(set(dftop['Organization']))
len(intersect)

######## Forma 1
k = df.shape[1]
for i in range(df.shape[0]):
    for j in range(dftop.shape[0]):
        if df["Organization Name"][i]==dftop["Organization"][j]:
            df.iloc[i:i+1, k-2:k-1] = 1
            break
        
###### Otra Forma
for j in intersect:
    df.loc[df['Organization Name'] == j, ['Flag']] = 1


# df["Organization Name"]=df["Organization Name"].str.upper()

##############################################################################
################## CrunchBase vs Unicorn

df1["y"]=0
df1["Flag"]=0

intersect2=set(df1['Organization Name']).intersection(set(dfunicorn['Name']))
len(intersect2)

######## Forma 1
k = df1.shape[1]
for i in range(df1.shape[0]):
    for j in range(dfunicorn.shape[0]):
        if df1["Organization Name"][i]==dfunicorn["Name"][j]:
            df1.iloc[i:i+1, k-2:k-1] = 1
            break
        
###### Otra Forma
for t in intersect2:
    df1.loc[df1['Organization Name'] == t, ['Flag']] = 1

#############################################################################
################################################################################
################## CrunchBase vs Unicorn vs Top 100

# df2 = df2.drop(['y'], axis=1) # elimina columna Organization Name URL 
df2["y"]=0
df2["Flag"]=0

intersect3=set(df['Organization Name']).intersection(set(dftop['Organization'])).intersection(set(dfunicorn['Name']))
len(intersect3)

# conversion del conjunto intersect3 a numpy.array
int3=np.array(list(intersect3))

k = df2.shape[1]
for i in range(df2.shape[0]):
    for j in range(len(intersect3)):
        if df2["Organization Name"][i]==int3[j]:
            df2.iloc[i:i+1, k-2:k-1] = 1
            break

for l in intersect3:
    df2.loc[df2['Organization Name'] == l, ['Flag']] = 1


# df["Organization Name"]=df["Organization Name"].str.upper()

##############################################################################
