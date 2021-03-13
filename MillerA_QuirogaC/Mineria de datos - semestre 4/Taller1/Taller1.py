#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 06:13:40 2021
Métodos Avanzados de Minería de Datos y Machine Learning
Doc: Luz Stella Gómez Fajardo Ph.D
Taller 1
Universidad Sergio Arboleda
@author: milleralexanderquirogacampos

TALLER 1.
1.	Top 20 de ciudades por concentración de empresas 
2.	Distribución de ciudades por capital levantado 
3.	Distribución de ciudades por capital levantado por año 
4.	Distribución de ciudades por capital levantado por sector 
5.	Sectores con mayor inversión equity
6.	15 rondas de financiación más grandes por año
"""

#importación de librerías
from scipy import stats
from sklearn import metrics
from scipy import stats
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os #sistema op
from sklearn import preprocessing
from pandas_profiling import ProfileReport

os.chdir('/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Clase1')
cwd=os.getcwd() # asigan variable swd al directorio
excel_file = "Colombia-Feb21.xlsx" #asigno archivo

df = pd.read_excel(excel_file, header= 0)

#elimino la columna index para que en las gráficas no me aparezca el numero de la fila
#df = pd.read_excel(excel_file, index_col=0)

#Remplazo las palabras con caracteres especiales
df = df.replace({"BogotÃ¡, Distrito Especial, Colombia":"Bogotá",
                   "MedellÃ­n, Antioquia, Colombia":"Medellín", 
                   "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén", 
                   "BogotÃ¡, Distrito Especial, Colombia":"Bogotá",
                   "MedellÃ­n, Antioquia, Colombia":"Medellín", 
                   "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén"})

#Elimino la fila con el keyword drop (se elimina Organization Name URL)
df = df.drop(['Organization Name URL'], axis=1)

#una forma muy conveniente, filtrar los valores de una estructura de 
#datos pandas para dejar solo aquellos no nulos.
df0 = df.dropna()

#Observa las variables del df
df0.columns

#imprime los primeros 15 y los últimos 15 registros del dataset
print(df0.head(15))
print(df0.tail(15))

#Imprime la primera columna del df
df0.iloc[:,1]

#Estadística de las variables cuantitativas
Statics = df0.describe()

#%%#########        TALLER 1    ##############
"""
1.	Top 20 de ciudades por concentración de empresas 
2.	Distribución de ciudades por capital levantado 
3.	Distribución de ciudades por capital levantado por año 
4.	Distribución de ciudades por capital levantado por sector 
5.	Sectores con mayor inversión equity 
6.	15 rondas de financiación más grandes por año
"""
###########     SOLUCIÓN        ##############

#########   1.	Top 20 de ciudades por concentración de empresas 

plot = (100 * df0['Headquarters Location'].value_counts() / len(df0['Headquarters Location'])).plot(
kind='bar', title='Localización de Sedes %')

plot = df0['Headquarters Location'].value_counts().plot(kind='pie', autopct='%.2f', 
                                            figsize=(6, 6),
                                            title='Ciudades')

#calcula el numero de empresas por ciudad
City = pd.crosstab(index=df0["Headquarters Location"],columns=df0["Organization Name"], margins=True)
City.sort_values(["All"], ascending = False)
pd.value_counts(df0['Headquarters Location'])

#%%
#########   2.	Distribución de ciudades por capital levantado

pd.crosstab(index=df0["Headquarters Location"],columns = df0["Last Funding Amount"])
plot = pd.crosstab(index=df0['Headquarters Location'].head(20),columns = df0['Last Funding Amount'].head(20)).plot(kind='bar')


#%%

######### 3.	Distribución de ciudades por capital levantado por año 


df0['Last Funding Date'] = pd.to_datetime(df0['Last Funding Date'])
df0['año']= df0['Last Funding Date'].dt.year

#Me relaciona las variables
tabla = pd.pivot_table(df0, 'Last Funding Amount','año','Headquarters Location', aggfunc = np.sum)

tabla.fillna(0, inplace=True)

tabla.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('año')
plt.ylabel('Last Funding Amount USD')
plt.title('Last Funding Amount - año - Headquarters Location')
plt.show()

#%%

#########   4.	Distribución de ciudades por capital levantado por sector

T = df0.pivot_table(index= 'Industries',values = ("Last Funding Amount","Headquarters Location") , aggfunc = np.sum)
T = T.sort_values(['Last Funding Amount'],ascending=False)
T

df0["Industries"] = np.where(df0["Industries"] == 'Consumer Goods, E-Commerce, Food Delivery',
                             'Consumer Goods, E-Commerce, Pet, Retail', df0["Industries"])

df0["Industries"].unique()# UNIR las categorias en una sola#

df0["Industries"] = np.where(df0["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Financial Services, FinTech, Lending, Online Portals, Small and Medium Businesses', 'Credit, Finance, Financial Services, FinTech', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Food and Beverage, Food Delivery, Restaurants, Retail Technology', 'Food Delivery, Restaurants, Waste Management', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='E-Commerce, Logistics, Software', 'Computer, SaaS, Software', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Credit, Finance, Financial Services, FinTech', 'Financial Services', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Finance, Financial Services, FinTech', 'Financial Services', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Financial Services, FinTech, Personal Finance', 'Financial Services', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Financial Services, FinTech, Payments', 'Financial Services', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Automotive, Transportation, Travel', 'Automotive', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Industrial, Mining, Oil and Gas', 'Oil and Gas', df0["Industries"])
df0["Industries"] = np.where(df0["Industries"] =='Energy, Energy Efficiency, Oil and Gas, Renewable Energy', 'Oil and Gas', df0["Industries"])

T2 = pd.pivot_table(df0, 'Last Funding Amount','Headquarters Location','Industries', aggfunc=np.sum)
T2.fillna(0, inplace=True)
T2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Headquarters Location')
plt.ylabel('Last Funding Amount USD')
plt.title('Last Funding Amount - Industies- Headquarters Location')
plt.show()


#%%

#########   5.	Sectores con mayor inversión equity 

pd.crosstab(index=df0['Last Equity Funding Amount Currency (in USD)'],columns = df0['Industries'])
plot = pd.crosstab(index=df0['Last Equity Funding Amount Currency (in USD)'].head(20),columns = df0['Industries'].head(20)).plot(kind='bar')


#%%

#########   6.	15 rondas de financiación más grandes por año

#devuelve un objeto conteniendo, en orden descendiente por defecto
pd.value_counts(df0["año"])


df0['Last Funding Date'] = pd.to_datetime(df0['Last Funding Date'])#filtra variable de tiempo
df0['year']= df0['Last Funding Date'].dt.year# asigna nueva variable
pd.value_counts(df0["year"])
 
T = pd.pivot_table(df0,'Last Funding Amount','year','Organization Name', aggfunc=np.sum)
T.fillna(0, inplace=True)
T.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('Last Funding Amaunt')
plt.title('Last Funding Amount - year- Headquarters Location')
plt.show()
 
df0.sort_values(by=["year","Last Equity Funding Amount Currency (in USD)"], inplace=True, ascending=False)#organizarlos de manera ascendente
df0 = df0.copy()#guerda la lista en nuevo df
df0.drop(df0[df0["year"] == 0].index)
df0.drop(df0[df0["Last Equity Funding Amount Currency (in USD)"] == 0].index)
 
yearsSet = set(df0["year"].tolist())
yearOrderList = sorted(list(yearsSet))
 
for year in yearOrderList:
    results = df0[ df0["year"] == year ]
    results.sort_values(by=["Last Equity Funding Amount Currency (in USD)"], inplace=True, ascending=True)
    top15 = results.head(15).copy()
   
    plt.title("Año %s"%(year))
    plt.barh(top15["Organization Name"].tolist(),top15["Last Equity Funding Amount Currency (in USD)"].tolist())
    plt.show()



