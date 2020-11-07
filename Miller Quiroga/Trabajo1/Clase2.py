#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 08:13:11 2020

@author: milleralexanderquirogacampos
"""

import pandas as pd #importamos librerías para analisis de datos
import numpy as np #librería para análisis numérico
import matplotlib.pyplot as plt #Librería de visualización
import os #Librería para el sistema operativo


#ruta del archivo donde esta alojado
os.chdir('/Users/milleralexanderquirogacampos/Documents/Documentos - MacBook Pro de Miller/Maestria MatApl/3 - Metodos de Analisis de datos/LuzStellaGomez/Clase2')
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo
excel_file="Movies.xls" # Asignación del nombre del archivo a una bandera

movies1 = pd.read_excel(excel_file)

print(movies1.head(15))
print(movies1.tail(15))

#read_excel(excel_file, sheet_name=0, index_col=0)

#hace un df de cada hoja de excel
#movies1 = pd.read_excel(excel_file, sheet_name=0)
movies1 = pd.read_excel(excel_file, sheet_name=0, index_col = 0)
movies2 = pd.read_excel(excel_file, sheet_name=1, index_col = 0)
movies3 = pd.read_excel(excel_file, sheet_name=2, index_col = 0)

#Concatenar
movies = pd.concat([movies1,movies2,movies3])
#Coloca los espacios en blanco o fna como 0
movies.fillna(0, inplace=True)

movies.shape

#Organización de peliculas de acuerdo a la variable que se desea

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)

print(sorted_by_gross.head(10))

#grafica diagrama de barras horizontal
#sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")
sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar")

#sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar", legend = "Reverse")


#Histograma con el score de las peliculas
movies["IMDB Score"].plot(kind="hist")
#plt.hist(movies['IMDB Score'], bins=18)


#se realiza una estadistica del dataframe
movies.describe()

movies["IMDB Score"].mean()
movies['Gross Earnings'].mean()
movies['Gross Earnings'].corr(movies["Budget"])



#Genera una nueva columna en el df llamada netEarnings
movies["netEarnings"] = movies["Gross Earnings"] - movies["Budget"]


sorted_by_netEarnings = movies.sort_values(["netEarnings"],ascending=False)
sorted_by_netEarnings["netEarnings"].head(10).plot(kind="bar")
#sorted_by_netEarnings["netEarnings"].head(10).plot(kind="barh")
plt.xlabel('Posición')
plt.ylabel('NetEarnings')
plt.title('Movies IMDB')
plt.show()



"""
#Tarea
Punto 1.

movies1 = pd.read_excel(excel_file, sheet_name=0, index_col = 0)
movies2 = pd.read_excel(excel_file, sheet_name=1, index_col = 0)
movies3 = pd.read_excel(excel_file, sheet_name=2, index_col = 0)



sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar")
plt.xlabel('Posición')
plt.ylabel('Gross Earnings')
plt.title('Movies IMDB')
plt.show()


"""
#Tarea 
#Punto 2. 


tabla1 = pd.pivot_table(movies, values=["netEarnings"], index=["Year"], aggfunc=np.sum)
tabla1.plot(kind="bar")
plt.xlabel('Year')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Year')
plt.show()

#Punto 3. 

#tabla2 = pd.pivot_table(movies, values=["netEarnings"], index=["Country","Language"], aggfunc=np.sum)
tabla2 = pd.pivot_table(movies, 'netEarnings','Country','Language', aggfunc=np.sum )
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country - Language')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Country - Language')
plt.show()

#Punto 4. 

tabla3 = pd.pivot_table(movies, values=["Language"], index=["Country"], aggfunc="count")
tabla3.plot(kind="bar", stacked = 'True',alpha = 0.8 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country')
plt.ylabel('Languaje')
plt.title('Country - Language')
plt.show()


plt.figure(num='Movies IMDB')
sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")
plt.xlabel('Posición')
plt.ylabel('Gross Earnings')
plt.title('Movies IMDB')
plt.show()

sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")



import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Definimos una lista con paises como string
paises = ['Estados Unidos', 'España', 'Mexico', 'Rusia', 'Japon']
#Definimos una lista con ventas como entero
ventas = [25, 32, 34, 20, 25]

fig, ax = plt.subplots()
#Colocamos una etiqueta en el eje Y
ax.set_ylabel('Ventas')
#Colocamos una etiqueta en el eje X
ax.set_title('Cantidad de Ventas por Pais')
#Creamos la grafica de barras utilizando 'paises' como eje X y 'ventas' como eje y.
plt.bar(paises, ventas)
plt.savefig('barras_simple.png')
#Finalmente mostramos la grafica con el metodo show()
plt.show()




plt.show()

x_values = movies.Budget.unique()
y_values = movies.Title.value_counts().tolist()
plt.figure()
plt.bar(x_values, y_values)          #El gráfico
plt.title('Pedidos de postres')      #El título
ax = plt.subplot()                   #Axis
ax.set_xticks(x_values)             #Eje x
ax.set_xticklabels(x_values)        #Etiquetas del eje x
ax.set_xlabel('Petición de postre')  #Nombre del eje x
ax.set_ylabel('Volumen de peticiones')  #Nombre del eje y



Pivote1 = pd.pivot_table(movies, values=["netEarnings"], index=["Year"], aggfunc=np.sum)
Pivote1.plot(kind="barh")
plt.xlabel('Year')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Year')
plt.show()

Pivote2 = pd.pivot_table(movies, 'netEarnings','Country','Language', aggfunc=np.sum )
Pivote2.fillna(0, inplace=True)
Pivote2.plot(kind="bar")
plt.xlabel('Country - Language')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Country - Language')
plt.show()


