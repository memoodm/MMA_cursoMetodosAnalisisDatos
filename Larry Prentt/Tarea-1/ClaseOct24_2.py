# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 08:20:18 2020

@author: lprentt
"""
import pandas as pd # Importando libreria pandas para analisis de datos
import numpy as np  # Importando libreria numpy  para analisis numerico
import matplotlib.pyplot as plt # Para Visualizacion de datos
import os # libreria para uso de comandos DOS - Sistema operativo

os.chdir("D:/Analisis-Datos/python")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

excel_file = 'Movies.xls'  # objeto string con el nombre del archivo a revisar # Es una bandera

movies1=pd.read_excel(excel_file, sheet_name=0, index_col=0) # metodo read lee el excel como un dataframe. Por default
# lee la primera hoja

# Exploraciones sobre el dataframe movies

print(movies1.head(15))
print(movies1.tail(15))

movies2=pd.read_excel(excel_file, sheet_name=1, index_col=0)
movies3=pd.read_excel(excel_file, sheet_name=2, index_col=0)

movies = pd.concat([movies1, movies2, movies3])
# movies.fillna(0, inplace=True) # Asigna cero a todos los datos del dataframe que fueron cargados como NaN
movies.shape  # imprime las dimensiones del dataframe

# organizando df por ingreso gross

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)

print(sorted_by_gross.head(10))

sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")
plt.show()

sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar")  # grafico de barras vertical
plt.show()

movies["IMDB Score"].plot(kind="hist")
plt.show()

movies.describe() # estadistica descriptiva
movies["IMDB Score"].mean()  # mean del score
movies["Gross Earnings"].mean() # mean del Gross Earnings
movies["Gross Earnings"].corr(movies["Budget"]) # correlacion entre 2 variables

movies["Net_Earnings"]= (movies["Gross Earnings"] - movies["Budget"])/1000000

# Tarea 1
sorted_by_net_earnings = movies.sort_values(['Net_Earnings'], ascending=False)
sorted_by_net_earnings['Net_Earnings'].head(5).plot(kind="barh")
plt.ylabel("Movies's Title")
plt.xlabel("Net Earnings MMUSD")
plt.title("IMDB Score - Top Five")
plt.show()

#Tarea 2
tabla1 = pd.pivot_table(movies, values=["Net_Earnings"], index=["Year"],  aggfunc=np.sum)
tabla1.plot(kind="bar")
plt.xlabel('Year')
plt.ylabel('Net_Earnings MMUSD')
plt.title('Net_Earnings Vs Años')
plt.show()

#Tarea 3  # pendiente arreglar grafica
tabla2 = pd.pivot_table(movies, 'Net_Earnings','Country','Language', aggfunc=np.sum )
#tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country')
plt.ylabel('Net_Earnings MMUSD')
plt.title('Net_Earnings - Country - Language')
#plt.legend(loc='upper center', shadow=True, fontsize='x-small')
plt.legend(loc='upper left', shadow=True, fontsize='xx-small')
plt.show()

#Tarea 4
tabla3 = pd.pivot_table(movies, values=["Language"], index=["Country"], aggfunc="count")
tabla3.plot(kind="bar", stacked = 'True',alpha = 0.8 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country')
plt.ylabel('Languaje (Counts)')
plt.title('Country - Language')
plt.show()


