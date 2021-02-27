# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 25 10:30:00 2021

@author: larry Prentt
"""

#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

os.chdir("D:/4to_Semestre/Mineria de datos\python")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

xls_file = 'Colombia-Feb21.xlsx'

df=pd.read_excel(xls_file, header=0,sep=';', index_col=0)

df.head(20) # Muestra los primeros 20 datos

df = df.drop(['Organization Name URL'], axis=1) # elimina columna Organization Name URL 

# Limpieza de caracteres especiales y nombres largos
df = df.replace({"Ã¡": 'a', "Ã­­­": 'i', "Ãº":'u', "Ã³": 'o', "Ã©": 'e', "Ã¼Ã": 'ui'},regex=True)
df= df.replace({"MedellÃ­n, Antioquia, Colombia":'Medellin, Antioquia, Colombia'})
df= df.replace({"AndalucÃ­a, Valle del Cauca, Colombia":'Andalucia, Valle del Cauca, Colombia'})
df= df.replace({"ChÃ­a, Cundinamarca, Colombia":'Chia, Cundinamarca, Colombia'})
df= df.replace({", Distrito Especial, Colombia":""},regex=True)
df= df.replace({"Colombia":""},regex=True)

df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)
df.rename(columns={"Organization Name)": 'organization'}, inplace=True)

df2 = df[df['CBRank'].notna()]  # elimina solo las filas con datos nan en la columna CBRank

# elimina solo las filas con datos no numericos en la columna CBRank
df2 = df2[pd.to_numeric(df2['CBRank'], errors='coerce').notnull()]

# df2=df.dropna() # elimina filas con un o varios campo(s) con nan

df2["Last Funding Amount Currency (in USD)"].plot(kind="hist")
plt.xlabel("Last Funding Amount USD")
plt.show

sorted_by_gross = df2.sort_values(['Last Funding Amount Currency (in USD)'], ascending=False)
print(sorted_by_gross.head(10))

sorted_by_gross['Last Funding Amount Currency (in USD)'].head(10).plot(kind="barh")
plt.xlabel("Last Funding Amount USD")
plt.show()

df2.to_excel("Colombia-Feb21_Filter.xlsx")

######################################################## Punto 1

# Cuenta las empresas por cada ciudad
companies_by_city = pd.value_counts(df2['Headquarters Location'])
print(companies_by_city.head(20))

# Hace la grafica de las primeras 20 ciudades con mas empresas
companies_by_city.head(20).plot(kind="barh")
plt.xlabel("Numero de empresas por ciudad")
plt.title("Top de las primeras 20 ciudades con mas empresas")
plt.show()
############################################################


######################################################## Punto 2

# Distribucion de ciudades por capital levantado
a=df2.groupby(['Headquarters Location'])['Last Funding Amount Currency (in USD)'].agg('sum')/1000000

# Organiza distribucion de ciudades por capital levantado de mayor a menor
a=a.sort_values(ascending=False)

# grafica las primeras 20 ciudades por capital levantado
a.head(20).plot(kind="barh")
plt.xlabel("Last Funding Amount MMUSD")
plt.title("Top de las primeras 20 ciudades")

# No sirve porque cada dato de last funding lo toma categoria
#c=pd.crosstab(index=df2["Headquarters Location"],columns=df2["Last Funding Amount"],
#              values=df2["Last Funding Amount"], aggfunc='sum')

#plot = pd.crosstab(index=df2['Headquarters Location'],columns=df2['Last Funding Amount']).plot(kind='bar')
#c.plot(kind='bar')


###########################################################


#####################################################  Punto 3

df2['Last Funding Date'] = pd.to_datetime(df2['Last Funding Date'])
df2['year']= df2['Last Funding Date'].dt.year

b=df2.groupby(['Headquarters Location', 'year'])['Last Funding Amount Currency (in USD)'].agg('sum')/1000000
b.to_excel("punto3.xlsx") # Este excel reporta lo pedido

# b.to_frame("Headquarters Location","year")
# b = b.rename(columns = {"" : 'Amount'}, inplace = True)
# b.sort_values(['year'], ascending=True)
#b.sort_values(ascending=True)
#b.plot(kind="hist")

tabla = pd.pivot_table(df2, 'Last Funding Amount Currency (in USD)','year','Headquarters Location', aggfunc=np.sum )/1e6
tabla.fillna(0, inplace=True)
tabla.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('Last Funding Amount MMUSD')
plt.title('Last Funding Amount per year per city location')
plt.show()  # Este plot resuelve lo pedido


######################################################## 


#####################################################  Punto 4

df2['Industries2'] = pd.np.where(df2.Industries.str.contains("Fin"), "Finance",
                   pd.np.where(df2.Industries.str.contains("Logistics"), "Logistics",
                   pd.np.where(df2.Industries.str.contains("Transportation"), "Logistics",
                   pd.np.where(df2.Industries.str.contains("Automotive"), "Logistics",
                   pd.np.where(df2.Industries.str.contains("Food"), "Food",
                   pd.np.where(df2.Industries.str.contains("Fitness"), "Fitness",
                   pd.np.where(df2.Industries.str.contains("Fashion"), "Fashion",
                   pd.np.where(df2.Industries.str.contains("Artificial"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Apps"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Intelligence"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Database"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Software"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Animation"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Gaming"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Video"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Games"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Agriculture"), "Agriculture",
                   pd.np.where(df2.Industries.str.contains("Aerospace"), "Aerospace",
                   pd.np.where(df2.Industries.str.contains("Adventure"), "Tourism",
                   pd.np.where(df2.Industries.str.contains("Big Data"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Cloud Data"), "AI-Apps-Analytics",
                   pd.np.where(df2.Industries.str.contains("Energy"), "Energy",
                   pd.np.where(df2.Industries.str.contains("Oil and Gas"), "Energy",
                   pd.np.where(df2.Industries.str.contains("Advertising"), "Advertising",
                   pd.np.where(df2.Industries.str.contains("Auctions"), "Recycling",
                   pd.np.where(df2.Industries.str.contains("Recycling"), "Recycling",
                   pd.np.where(df2.Industries.str.contains("Waste"), "Recycling",
                   pd.np.where(df2.Industries.str.contains("Furniture"), "Furniture", "Others"))))))))))))))))))))))))))))

df2['Industries2'].unique()

tabla2 = pd.pivot_table(df2, 'Last Funding Amount Currency (in USD)','Headquarters Location','Industries2', aggfunc=np.sum )/1e6
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Industries2')
plt.ylabel('Last Fundind Amount MMUSD')
plt.title('Last Funding Amount - Industries2 - Headquarters Location')
plt.show()  # Este plot resuelve lo pedido

########################################################


#####################################################  Punto 5

tabla1 = pd.pivot_table(df2, values=["Last Equity Funding Amount Currency (in USD)"], index=["Industries2"],  aggfunc=np.sum)/1e6
tabla1.plot(kind="bar")
plt.xlabel('Industries2')
plt.ylabel('Last Equity Funding Amount Currency (in USD)')
plt.title('Equity Vs Industries')
plt.show()


########################################################


#####################################################  Punto 6

# tabla3 = pd.pivot_table(df2, values=["Last Funding Amount Currency (in USD)"], index=["year"],aggfunc=np.sum)

# tabla3.sort_values(['Last Funding Amount Currency (in USD)'], ascending=False)

# tabla3.plot(kind="bar")
# plt.xlabel('year')
# plt.ylabel('Last Funding Amount (in USD)')
# plt.title('Funding Vs Years')
# plt.show()


# tabla7 = pd.pivot_table(df2, 'Last Funding Amount Currency (in USD)','year','Industries2', aggfunc=np.sum )


# tabla7.fillna(0, inplace=True)
# tabla7.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
# plt.xlabel('year')
# plt.ylabel('Last Funding Amount MMUSD')
# plt.title('Last Funding Amount per year per Industries')
# plt.show()


c=df2.groupby(['year','Organization Name'])['Last Funding Amount Currency (in USD)'].agg('sum')/1000000
c.to_excel("punto6.xlsx") # Este excel reporta lo pedido

c2 = df2.groupby(['year','Organization Name']).agg({'Last Funding Amount Currency (in USD)':sum})
c2= c2['Last Funding Amount Currency (in USD)'].groupby('year', group_keys=False)
c2 = c2.apply(lambda x: x.sort_values(ascending=False).head(15))

c2.to_excel("punto6A.xlsx") # Este excel reporta lo pedido

c2=c2.to_frame()

tabla7 = pd.pivot_table(c2, 'Last Funding Amount Currency (in USD)','year','Organization Name', aggfunc="count" )/1000000
tabla7.fillna(0, inplace=True)
tabla7.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('Last Funding Amount MMUSD')
plt.title('Last Funding Amount per year per Organization Name')
plt.show()

# otra forma de obtener lo solicitado
c3=c2['Last Funding Amount Currency (in USD)'].groupby('year', group_keys=False).nlargest(15)

########################################################


#####################################################  Grafica Especial Pedida en Clase

tabla5 = pd.pivot_table(df2, values=["Last Funding Amount Currency (in USD)"], index=["year"],aggfunc=np.sum)/1e6
tabla6 = pd.pivot_table(df2, values=["Headquarters Location"], index=["year"],aggfunc="count")

tabla_merge= pd.concat([tabla5, tabla6], 1).dropna().mean(axis=1, level=0)
tabla_merge = tabla_merge.reset_index()

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(tabla_merge.year, tabla_merge["Last Funding Amount Currency (in USD)"], color=(190/255,190/255,190/255,0.7), label='Last Funding Amount Currency (in USD)')
ax2.plot(tabla_merge.year, tabla_merge["Headquarters Location"], color='green', label='Headquarters Location')
ax.set_xticklabels(tabla_merge.year)
ax.legend(loc='best')



########################################################