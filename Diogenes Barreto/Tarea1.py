# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 19:56:07 2020

@author: Diogenes Barreto
"""

import pandas as pd # Libreria para analisis de Datos#
import numpy as np
import matplotlib.pyplot as plt # libreria para graficar- vizualizacion de imagenes#
import os # Operative System #
#import datetime
#import matplotlib.dates as mdates
#from matplotlib.transforms import Transform
#from matplotlib.ticker import (
   # AutoLocator, AutoMinorLocator)


os.chdir("C:/Users/LENOVO/Documents/clase8/")

cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
excel_file="movies.xls" #asignacion dle nombre del archivo a una bandeja #

movies1=pd.read_excel(excel_file)

print(movies1.head(10))#imprime las primeras 10#
print(movies1.tail(10)) #imprime las primeras 10#

movies1=pd.read_excel(excel_file, sheet_name=0, index_col=0)
movies2=pd.read_excel(excel_file, sheet_name=1, index_col=0)
movies3=pd.read_excel(excel_file, sheet_name=2, index_col=0)

print(movies1,movies2,movies2)

# Conquetanar #

movies= pd.concat([movies1,movies2,movies3])
movies.fillna(0, inplace=True) # coloca los espacio NaN como Ceros#
movies.shape

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)
print(sorted_by_gross.head(10))

 #grafico de barras horizontales#
sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar")
plt.xlabel("Ranking")
plt.ylabel('Gross Earnings')
plt.title("IMDB Score<11")
plt.show()


movies["IMDB Score"].plot(kind="hist")
plt.show()

#descripcion de medias#
movies.describe()
movies["IMDB Score"].mean()
movies['Gross Earnings'].mean()

# analisis de correlacion#

movies['Gross Earnings'].corr(movies["Budget"]) # Baja correlacion, dinero invertdo no implica ganacia#

movies["netEarnings"]= movies['Gross Earnings'] - movies['Budget']
print(movies["netEarnings"])

sorted_by_netEarnings = movies.sort_values(["netEarnings"], ascending=False)
sorted_by_netEarnings["netEarnings"].head(10).plot(kind="bar")
plt.xlabel("Ranking")
plt.ylabel("netEarnings")
plt.title("IMDB Score<11")
plt.show()

#Punto 2 Grafico Gnacias Vs Años#

tabla1 = pd.pivot_table(movies, values=["netEarnings"], index=["Year"],  aggfunc=np.sum)
tabla1.plot(kind="bar")
plt.xlabel('Year')
plt.ylabel('netEarnings')
plt.title('Ganancias Vs Años')
plt.show()

#Punto 3 Grafico ganacias netas vs idioma -Pais#


tabla2 = pd.pivot_table(movies, 'netEarnings','Country','Language', aggfunc=np.sum )
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country - Language')
plt.ylabel('netEarnings')
plt.title('netEarnings - Language - Country')
plt.legend(loc='upper left', shadow=True, fontsize='xx-small')
plt.show()


#Punto 3 opcion adicional#

tabla2 = pd.pivot_table(movies, values=["netEarnings"], index=["Language", "Country"], aggfunc=np.sum)
tabla2.fillna(0, inplace=True)
fig, ax = plt.subplots(constrained_layout=True,)
ax2 = ax.secondary_xaxis('top')
#tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
ax1= tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
#ax2= tabla2.plot(kind="bar", stacked = 'True', secax=True, alpha = 1.0 ,width = 1.0, figsize=(9,4))             
#plt.xlabel('Country - Language')

ax1.set_xlabel("lenguage")
ax2.set_xlabel("Country")
#ax1.legend(loc="upper left")#
#ax2.legend(loc="upper right")
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Country - Language')
plt.show()


# Punto 4 Grafico de idioma y Pais#

tabla3 = pd.pivot_table(movies, values=["Language"], index=["Country"], aggfunc="count")
tabla3.plot(kind="bar", stacked = 'True',alpha = 0.8 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country')
plt.ylabel('Languaje')
plt.title('Language-Country')
plt.show()
