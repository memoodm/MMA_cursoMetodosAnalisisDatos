# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:56:07 2020

@author: Diogenes Barreto
"""

import pandas as pd # Libreria para analisis de Datos#
import numpy as np
from sklearn import preprocessing # libreria para porcesar datos#
import matplotlib.pyplot as plt # libreria para graficar- vizualizacion de imagenes#
#from pandas_profiling import ProfileReport

from sklearn.linear_model import LinearRegression #Regresión Lineal
from sklearn.linear_model import LogisticRegression #Regresión Logistica

import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns # graficas y estadisticas , maquillaje#
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

os.chdir("C:/Users/LENOVO/Documents/Clase10/")

cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
text_file="Banking.csv" #asignacion dle nombre del archivo a una bandeja #



data=pd.read_csv(text_file)

print(data.head(10))#imprime las primeras 10#
print(data.tail(10)) #imprime las primeras 10#

from pandas_profiling import ProfileReport
ProfileReport(data)
profile = ProfileReport(data, title="Pandas Profiling Report",explorative=True)
#reporte = ProfileReport(data).to_html("report.html")
#profile.to_scv("Pandas Profiling Report.txt")

profile = ProfileReport(data)
profile.to_file('profile_report.html')



data["education"].unique()# Permite Unir las categorias en una sola#

data['education'] = np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education'] = np.where(data['education'] =='basic.4y', 'Basic', data['education'])
data['education'] = np.where(data['education'] =='basic.6y', 'Basic', data['education'])

data["education"].unique()# Permite Unir las categorias en una sola#

count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

sns.countplot(x='y',data=data, palette='Blues')
sns.countplot(x='job',data=data, palette='Blues')
sns.countplot(y='marital', data=data, palette='GnBu_d')
#palette='cubehelix'
sns.countplot(x='job',data=data, palette='cubehelix')



pd.crosstab(data.duration,data.y).plot(kind='bar')
data.plot.scatter(x='duration', y='y')
plt.title('Purchase Frequency for duration ')
plt.xlabel('duration')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_cons_price_idx')

#plt.xlabel('cons_price_idx')
#plt.ylabel('Frequency of Purchase')


#Dimension del data Frame#
data.shape


sorted_by_gross = data.sort_values(['duration'], ascending=False)
print(sorted_by_gross.head(10))

 #grafico de barras horizontales#
sorted_by_gross['duration'].head(10).plot(kind="bar")
plt.xlabel("age")
plt.ylabel('duration')
plt.title("")
plt.show()


data["duration"].plot(kind="hist")
plt.show()

#descripcion de medias#
data.describe()
#data["duration"].mean()


#movies['Gross Earnings'].mean()

# analisis de correlacion#



