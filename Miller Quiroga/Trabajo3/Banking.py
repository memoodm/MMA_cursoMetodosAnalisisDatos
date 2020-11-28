#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:44:21 2020

@author: milleralexanderquirogacampos
"""
###########         LIBRERÍAS 

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
os.chdir('/Users/milleralexanderquirogacampos/Documents/Documentos - MacBook Pro de Miller/MaestriaMatematicas/3 - Metodos de Analisis de datos/LuzStellaGomez/Clase4')
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo
text_file = "Banking.csv" # Asignación del nombre del archivo a una bandera

#sklearn = metodologías de modelos para regresión logistica, neuronal,..., 
from sklearn.linear_model import LinearRegression #Regresión Lineal
from sklearn.linear_model import LogisticRegression #Regresión Logistica

df = pd.read_csv(text_file)

print(df.head(15))
print(df.tail(15))


ProfileReport(df)
# reporte = ProfileReport(Banking).to_html("report.html")
# reporte.to_scv(r'report.txt')

report = ProfileReport(df)
report.to_file('profile_report.html')

df['education'].unique()#Permite obtener categorías únicas
df['age'].unique()#Permite obtener categorías únicas
df['marital'].unique()#Permite obtener categorías únicas
df['contact'].unique()#Permite obtener categorías únicas
df['day_of_week'].unique()#Permite obtener categorías únicas
df['poutcome'].unique()#Permite obtener categorías únicas
df['y'].unique()#Permite obtener categorías únicas


df['education'] = np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education'] = np.where(df['education'] =='basic.4y', 'Basic', df['education'])
df['education'] = np.where(df['education'] =='basic.6y', 'Basic', df['education'])

df['y'].unique()#Permite obtener categorías únicas


count_no_sub = len(df[df['y']==0])
count_sub = len(df[df['y']==1])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

# sns.countplot(x ='y',Banking = Banking, palette = 'Blues')
# sns.countplot(x='job',Banking = Banking, palette='Blues')
# sns.countplot(y="marital", Banking = Banking, palette='GnBu_d')
# #palette='cubehelix'
# sns.countplot(x='job',Banking = Banking, palette='cubehelix')

pd.crosstab(df.age,df.y).plot(kind='barh')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Edad')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_Edad')
sns.countplot(x = "Edad", df = df, palette = "Blues")
sns.countplot(y = "Edad", df = df, palette = "GnBu_d")


"""
%matplotlib qt
"""

sns.countplot(x='job',df=df, palette='Blues')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Y')
plt.ylabel('Frequency of Y = 1 or 0')
#plt.savefig('purchase_fre_Edad')
plt.show()

sns.countplot(y="job", df=df, palette='GnBu_d')
sns.countplot(y="job", df=df, palette='cubehelix')
