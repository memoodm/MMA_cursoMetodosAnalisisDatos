# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:37:04 2020

@author: Julian
"""

#PUNTO 1  MOSTRAR NET PROFIT CON EL NOMBRE DE LA PELICULA EN LA MISMA GRAFICA

import pandas as pd 
import numpy as np
 #graficar en python / visualizacion
import os # operative system
import matplotlib.pyplot as plt


os.chdir('C:/Users/Julian/Documents/MMA/PYTHONLUZ')
cwd=os.getcwd() # asigan variable swd al directorio
excel_file= "Movies.xls" #asigno archivo

movies1 = pd.read_excel(excel_file)

#Exploraciones dobre el df

print(movies1.head(15))
print(movies1.tail(15))

movies2 = pd.read_excel(excel_file, sheet_name=1)
movies3 = pd.read_excel(excel_file,sheet_name=2)

#concatenar

movies = pd.concat([movies1, movies2, movies3])
movies.shape

#organizar el df

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)
print(sorted_by_gross.head(10))

#sorted_by_gross['Gross Earnings'].head(10).plot(kind="bar")

#histograma

movies["IMDB Score"].plot(kind="hist")
plt.show() #MAYORIA DE SCORES SALEN ENTRE 6 Y 8

movies.describe()

movies["IMDB Score"].mean()

movies["IMDB Score"].mean()
#correlacion bajita, no tiene que ver el presupuesto con las ganancias
movies["Gross Earnings"].corr(movies["Budget"])
movies["netProfit"]= movies["Gross Earnings"] - movies["Budget"]
movies["netProfit"]

sorted_by_netProfit= movies.sort_values(["netProfit"], ascending = False)
#sorted_by_netProfit["netProfit"].head(5).plot(kind="barh")

#sorted_by_netProfit[['Title', 'netProfit']].head(10).plot(kind='barh')

#TAREA 1 MOSTRAR NET PROFIT CON EL NOMBRE DE LA PELICULA

plt.xlabel('netProfit')
plt.ylabel('Title')
plt.plot(sorted_by_netProfit['netProfit'].head(10),sorted_by_netProfit['Title'].head(10))


#TAREA 2TABLA DINAMICA

#POR AñO
Tabla= pd.pivot_table(sorted_by_netProfit,index=['Year'], values= ['netProfit'] )

#PAIS, IDIOMA GANANCIA

Tabla1= pd.pivot_table(sorted_by_netProfit,index=['Country', 'Language'] , values = ['netProfit'])

#PAIS, IDIOMA

Tabla2= pd.pivot_table(sorted_by_netProfit,index=['Country', 'Language'])





