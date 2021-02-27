# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 06:12:40 2021

@author: LENOVO
"""

import pandas as pd # Libreria para analisis de Datos#
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split# divide en DF en subconjuntos de entrenamientos aleatorio
from sklearn.utils import resample

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns # graficas y estadisticas , maquillaje#
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)


os.chdir("C:/Users/LENOVO/Documents/Clase1/")

cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
xls_file='Colombia-Feb21.xlsx' #asignacion dle nombre del archivo a una bandeja #

df=pd.read_excel(xls_file)
#df=pd.read_excel(xls_file, index_col=0)
#df=pd.read_excel(excel_file, header= 0)


print(df.head())#imprime las primeras 10#
print(df.tail()) #imprime las primeras 10#

print(df.shape)
df.head()# nombre de las columnas
plt.show()

df=df.drop(['Organization Name URL'], axis=1)


#Remplazo las palabras con caracteres especiales
df=df.replace({"BogotÃ¡, Distrito Especial, Colombia":"Bogotá",
                   "MedellÃ­n, Antioquia, Colombia":"Medellín", 
                   "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén", 
                   "BogotÃ¡, Distrito Especial, Colombia":"Bogotá",
                   "MedellÃ­n, Antioquia, Colombia":"Medellín", 
                   "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén"})


df.isnull().any()

df.describe() #estadistica descriptiva de la variables

from pandas_profiling import ProfileReport # Reportes de variables
ProfileReport(df)
profile = ProfileReport(df, title="Pandas Profiling Report",explorative=True)
profile = ProfileReport(df)
profile.to_file('profile_report.html')

df2 = df.dropna()
df2["Last Funding Amount"].plot(kind='bar')
plt.show()

sorted_by_gross = df2.sort_values(['Last Funding Amount'], ascending=False)
print(sorted_by_gross.head(10))

sorted_by_gross['Last Funding Amount'].head(10).plot(kind="barh")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
1. Top de las Ciudades por concentracion de empresas
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#1.	Top 20 de ciudades por concentración de empresas 
plot = df2['Headquarters Location'].value_counts().head(20).plot(kind='pie',
                                            title='Localización de Sedes')

plot = (100 * df2['Headquarters Location'].value_counts() / len(df2['Headquarters Location'])).plot(
kind='bar', title='Localización de Sedes %')

plot = df2['Headquarters Location'].value_counts().head(20).plot(kind='pie', autopct='%.2f', 
                                            figsize=(6, 6),
                                            title='Ciudades')
#calcula el numero de empresas por ciudad
City = pd.crosstab(index=df2["Headquarters Location"].head(20),columns=df2["Organization Name"].head(20), margins=True)
City.sort_values(["All"], ascending = False)
plt.show("City.sort_values")

df2.plot.scatter(x="Headquarters Location", y="Organization Name")
plt.title('ciudades por concentracion')
plt.xlabel("Headquarters Location")
plt.ylabel("Organization Name")
plt.savefig('')

#Otra forma de calcularlo
df2.groupby("Headquarters Location")["Headquarters Location"].count().reset_index(name="count") \
                             .sort_values(["count"], ascending=False) \
                             .head(20)
plt.barh(df2["Headquarters Location"].tolist(),df2["count"].tolist())

"""
2. Distribucion de Ciudades por capital levanado
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Ciudad por capital Levantado
pd.crosstab(index=df2["Headquarters Location"].value_counts(),columns=df2["Last Funding Amount"].value_counts())
plot = pd.crosstab(index=df2['Headquarters Location'],columns=df2['Last Funding Amount'].plot(kind='bar'))

df2.plot.scatter(x="Headquarters Location", y="Last Funding Amount")
plt.title('Distribucion de Ciudades por Levante')
plt.xlabel("Headquarters Location")
plt.ylabel("Last Funding Amount")
plt.savefig('Grafica Ejercicio 2')


#sns.catplot(x='Last Funding Amount', y='Headquarters Location', kind="bar", data=df1)

"""
3. Distribucion de Ciudades por capital levanado por año
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


df2['Last Funding Date'] = pd.to_datetime(df2['Last Funding Date'])#filtra variable de tiempo
df2['year']= df2['Last Funding Date'].dt.year# asigna nueva variable

tabla= pd.pivot_table(df2, 'Last Funding Amount','year','Headquarters Location', aggfunc=np.sum)
tabla.fillna(0, inplace=True)
tabla.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - año - Headquarters Location')
plt.show()
print(df2.shape)

"""
4. Distribucion de ciudades por capital Levantado por Sector
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# df2["Industries"].unique()# Permite Unir las categorias en una sola#

# df2["Industries"] = np.where(df2["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Lending, Online Portals, Small and Medium Businesses', 'Credit, Finance, Financial Services, FinTech', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Food and Beverage, Food Delivery, Restaurants, Retail Technology', 'Food Delivery, Restaurants, Waste Management', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='E-Commerce, Logistics, Software', 'Computer, SaaS, Software', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Credit, Finance, Financial Services, FinTech', 'Financial Services', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Finance, Financial Services, FinTech', 'Financial Services', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Personal Finance', 'Financial Services', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Payments', 'Financial Services', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Automotive, Transportation, Travel', 'Automotive', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Industrial, Mining, Oil and Gas', 'Oil and Gas', df2["Industries"])
# df2["Industries"] = np.where(df2["Industries"] =='Energy, Energy Efficiency, Oil and Gas, Renewable Energy', 'Oil and Gas', df2["Industries"])


tabla3 = pd.pivot_table(df2, 'Last Funding Amount','Headquarters Location','Industries', aggfunc=np.sum)
tabla3.fillna(0, inplace=True)
tabla3.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Headquarters Location')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - Industies- Headquarters Location')
plt.show()
print(df2.shape)
"""
5. Sectores con Mayor Inversion Equity
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pd.crosstab(index=df2['Industries'],columns = df2['Last Equity Funding Amount Currency (in USD)'])
plot = pd.crosstab(index=df2['Industries'].head(20),columns = df2['Last Equity Funding Amount Currency (in USD)'].head(20)).plot(kind='bar')


df2.plot.bar(x='Last Equity Funding Amount Currency (in USD)', y='Industries')
plt.title('Sectores con mayor inversion Equity')
plt.xlabel('Last Equity Funding Amount Currency (in USD)')
plt.ylabel('Industries')
plt.savefig('')

"""
6. 15 Rondas de financiacion mas grande por año
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOLUCION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

df2['Last Funding Date'] = pd.to_datetime(df2['Last Funding Date'])#filtra variable de tiempo
df2['year']= df2['Last Funding Date'].dt.year# asigna nueva variable
pd.value_counts(df2["year"])

tabla4 = pd.pivot_table(df2,'Last Funding Amount','year','Industries', aggfunc=np.sum)
tabla4.fillna(0, inplace=True)
tabla4.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('Last Funding Amaunt')
plt.title('Last Funding Amount - year- Industries')
plt.show()
print(df2.shape)

# ordenador de rondas por años
df2.sort_values(by=["year","Last Equity Funding Amount Currency (in USD)"], inplace=True, ascending=False)#organizarlos de manera ascendente
df3 = df2.copy()#guerda la lista en nuevo df
df3.drop(df3[df3["year"] == 0].index)
df3.drop(df3[df3["Last Equity Funding Amount Currency (in USD)"] == 0].index)

yearsSet = set(df3["year"].tolist())
yearOrderList = sorted(list(yearsSet))

for year in yearOrderList:
    results = df3[ df3["year"] == year ]
    results.sort_values(by=["Last Equity Funding Amount Currency (in USD)"], inplace=True, ascending=True)
    top15 = results.head(15).copy()
    plt.show()
    
    plt.title("Año %s"%(year))
    plt.barh(top15["Organization Name"].tolist(),top15["Last Equity Funding Amount Currency (in USD)"].tolist())
    plt.show()
    
    





