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

#df=pd.read_excel(xls_file, header=0,sep=';', index_col=0)
df=pd.read_excel(xls_file, header=0,sep=';')

df.head(20) # Muestra los primeros 20 datos

df = df.drop(['Organization Name URL'], axis=1) # elimina columna Organization Name URL 


############## Limpieza de compañias que no son de colombia
########################################################################

# df.drop(df[df['Age'] < 25].index, inplace = True) # Ejemplo encontrado

# Neoalgae compañia española, se elimina ciudad Asturias
# Suaval Group compañia española, se elimina ciudad Asturias

# como ambas compañias son de asturias, se hace filtro por asturias y se eliminan
# si una sola compañia fuera de asturia, se hace filtro por nombre de compañia

# Wisi Communications en Albania, La Guajira, Colombia
# Wisi Communications es de Alemania

# Junta de Andalucía y Courbox son de Andalucía Españna.  se eliminan

# Aspyre Solutions (hay una compañia en uSA y otra en UK), aparece con direccion en Barrios
# unidos. Barrios Unidos, Distrito Especial, Colombia Se elimina

# BD Sensors, Pinnery, Ratiokontakt y Atlantik Bruecke son compañia Alemana, 
# aparecen en Bavaria, Cundinamarca, Colombia. se eliminan

# LocalXXL empresa alemana, aparece en Bavaria, Magdalena, Colombia se elimina

df.drop(df[df['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='Albania, La Guajira, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='AndalucÃ­a, Valle del Cauca, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)

#df.drop(df[df['Headquarters Location']=='Bavaria, Cundinamarca, Colombia'].index, inplace =True)
#df.drop(df[df['Headquarters Location']=='Bavaria, Magdalena, Colombia'].index, inplace =True)

# df = df[~df['DB Serial'].str.contains('\*', na=False)] # ejemplo encontrado # drop with wildcard
# el ejemplo borra registros de una columna que contenga asterico

# df.drop(df[df['Headquarters Location']=='Bavaria(?!$)'].index, inplace =True) # No funciona
df = df[~df['Headquarters Location'].str.contains('Bavaria(?!$)')]


# M=df[df['Headquarters Location'] == 'Bavaria, Cundinamarca, Colombia']

# iFut, Inova House 3D, MÃ¡quina de Vendas y Acceleratus aparecen en
# Brasilia, Distrito Especial, Colombia.  son compañias de Brasil, se eliminan

df = df[~df['Headquarters Location'].str.contains('Brasilia')]

# Biometric Update, Boardwalk REIT, ChartBRAIN, LIFT Partners, canada Toronto, calgary
# Pulse Software, compañia australiana, pero aparece canada colombia

# se eliminan todas las Empresas de Canada, Cundinamarca, Colombia, excepto Qinaya y Partsium

df=df.drop(df[(df['Headquarters Location'] == 'CanadÃ¡, Cundinamarca, Colombia') 
           & ((df['Organization Name'] != 'Qinaya')  & (df['Organization Name'] != 'Partsium'))].index)

# data_new = data.drop(data[(data['col_1'] == 1.0) & (data['col_2'] == 0.0)].index) # Ej.
# Drop con 2 condiciones !!!
# data_new=df.drop(df[(df['Headquarters Location'] == 'Canada, Cundinamarca, Colombia') & (df['Organization Name'] != 'Qinaya')].index)

# M=data_new[data_new['Headquarters Location'] == 'Canada, Cundinamarca, Colombia']
# M=df[df['Headquarters Location'] == 'Canada, Cundinamarca, Colombia']

# mRisk, BattleBit, Cencosud Shopping Centers compañias de Chile
# Chile, Huila, Colombia  se Eliminan

df.drop(df[df['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)

# WindoTrader USA, aparece como Las Vegas, Sucre, Colombia
# se elimina

df.drop(df[df['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)

# Onyx, Elm, Photogramy, Ferrisland, BeyondROI sede en Los Angeles, Huila, Colombia
# se eliminan

df.drop(df[df['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)

# Peris Costumes, Cositas de EspaÃ±a, Esri, Pirsonal, Barrabes, Carousel Group, WORLD COMPLIANCE ASSOCIATION
# Clupik, Mobile Dreams ltd., Acqualia, LoAlkilo, Codekai, Vitriovr, El inmobiliario mes a mes
# Core Business Consulting, Renewable Energy Magazine, datosmacro, 1001talleres, GGBOX
# con sede en Madrid, Distrito Especial, Colombia

df.drop(df[df['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)

# Advanet (Japon) y Truland Service Corporation en USA. aparecen Maryland, Cundinamarca, Colombia

df.drop(df[df['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)

# POC Network Technologies (TransactRx), Alert Global Media.  sede Miami, Magdalena, Colombia

df.drop(df[df['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)

# Sicartsa sede en Mexico, Huila, Colombia

df.drop(df[df['Headquarters Location']=='MÃ©xico, Huila, Colombia'].index, inplace =True)

# 24marine, Merkadoo sede en Panama, Magdalena, Colombia

df.drop(df[df['Headquarters Location']=='PanamÃ¡, Magdalena, Colombia'].index, inplace =True)

# Agros, Downloadperu.com, Mesa 24/7, Dconfianza, Caja Los Andes, Snacks America Latina Peru S.R.L.
# Pandup, Apprende sede en Peru, Valle del Cauca, Colombia

df.drop(df[df['Headquarters Location']=='PerÃº, Valle del Cauca, Colombia'].index, inplace =True)

# Cocodrilo Dog es de Melbourne

df.drop(df[df['Organization Name']=='Cocodrilo Dog'].index, inplace =True)

# Homeland Security Careers es de USA

df.drop(df[df['Organization Name']=='Homeland Security Careers'].index, inplace =True)

# Big Picture Solutions

df.drop(df[df['Organization Name']=='Big Picture Solutions'].index, inplace =True)






################## Depuracion de nombres de ciudad incorrectos a correctos
##########################################################################

# La compañia Savy tiene sede usaquen, se cambia a Bogota

df= df.replace({"UsaquÃ©n, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})

# M=df[df['Organization Name'] == 'Savy']

# PagomÃ­o  es compania de medellin y aparece con sede
# Antioquia, Antioquia, Colombia

df= df.replace({"Antioquia, Antioquia, Colombia":'MedellÃ­n, Antioquia, Colombia'})

# El Herald  El Heraldo sede Atlantico, Magdalena, Colombia  Es Barranquilla
# Atlantico, Magdalena, Colombia
# 

df= df.replace({"El Herald":'El Heraldo'})
df= df.replace({"AtlÃ¡ntico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})

# compañia Monolegal es de tunja y aparece Boyaca, Boyaca, Colombia

df= df.replace({"BoyacÃ¡, Boyaca, Colombia":'Tunja, Boyaca, Colombia'})

# Celotor es de cali aparece como Colombiano, Magdalena, Colombia

df= df.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})

# Qinaya, compañia colombiana
# https://www.wradio.com.co/noticias/tecnologia/qinaya-el-emprendimiento-que-convierte-cualquier-televisor-en-un-computador/20210301/nota/4113498.aspx
# https://www.youtube.com/watch?v=XBgbwUxkatc
# Canada, Cundinamarca, Colombia vs Bogota, Distrito Especial, Colombia
# Replace with condition
df.loc[(df['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'

# Partsium
# Partsium, bogota.  El Sitio pone a disposición de los Usuarios un espacio virtual que les permite
# comunicarse mediante el uso de Internet para encontrar una forma de vender o comprar productos y
# servicios. PARTSIUM no es el propietario de los artículos ofrecidos, no tiene posesión de ellos ni
# los ofrece en venta. Los precios de los productos y servicios están sujetos a cambios sin previo
# aviso.
# website rental to do business

df.loc[(df['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
df.loc[(df['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'

##### asignar valores de una columna a otra para un mismo index o registro

################# BANCO DE OCCIDENTE
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Industries'] = df['Headquarters Location']
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Headquarters Location'] = 'Cali, Valle del Cauca, Colombia'
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Description'] = df['CB Rank (Company)']
df.loc[df['Organization Name'] == 'Banco de Occidente', 'CB Rank (Company)'] = df['Founded Date']
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Founded Date'] = df['Founded Date Precision']
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Founded Date Precision'] = df['Last Funding Date']
df.loc[df['Organization Name'] == 'Banco de Occidente', 'Last Funding Date'] = np.nan

################# ALEGRA
df.loc[df['Organization Name'] == 'Alegra', 'Headquarters Location'] = df['CB Rank (Company)']
df.loc[df['Organization Name'] == 'Alegra', 'CB Rank (Company)'] = df['Founded Date Precision']

df.loc[df['Organization Name'] == 'Alegra', 'Founded Date'] = df['Last Funding Date']
df.loc[df['Organization Name'] == 'Alegra', 'Founded Date Precision'] = df['Last Funding Amount']

df.loc[df['Organization Name'] == 'Alegra', 'Last Funding Date'] = df['Last Funding Amount Currency']
df.loc[df['Organization Name'] == 'Alegra', 'Last Funding Amount'] = df['Last Funding Amount Currency (in USD)']
df.loc[df['Organization Name'] == 'Alegra', 'Last Funding Amount Currency'] = df['Last Equity Funding Amount']
df.loc[df['Organization Name'] == 'Alegra', 'Last Funding Amount Currency (in USD)'] = df['Last Equity Funding Amount Currency']

df.loc[df['Organization Name'] == 'Alegra', 'Last Equity Funding Amount'] = np.nan
df.loc[df['Organization Name'] == 'Alegra', 'Last Equity Funding Amount Currency'] = np.nan

################# DHAMOVA

df.loc[df['Organization Name'] == 'Dhamova', 'Industries'] = df['Headquarters Location']
df.loc[df['Organization Name'] == 'Dhamova', 'Headquarters Location'] = df['Description']
df.loc[df['Organization Name'] == 'Dhamova', 'Description'] = df['CB Rank (Company)']
df.loc[df['Organization Name'] == 'Dhamova', 'CB Rank (Company)'] = df['Founded Date']
df.loc[df['Organization Name'] == 'Dhamova', 'Founded Date'] = df['Founded Date Precision']
df.loc[df['Organization Name'] == 'Dhamova', 'Founded Date Precision'] = df['Last Funding Date']
df.loc[df['Organization Name'] == 'Dhamova', 'Last Funding Date'] = np.nan

################# GEOMODELr

df.loc[df['Organization Name'] == 'Geomodelr', 'Industries'] = df['Headquarters Location']
df.loc[df['Organization Name'] == 'Geomodelr', 'Headquarters Location'] = df['Description']
df.loc[df['Organization Name'] == 'Geomodelr', 'Description'] = df['CB Rank (Company)']
df.loc[df['Organization Name'] == 'Geomodelr', 'CB Rank (Company)'] = df['Founded Date']
df.loc[df['Organization Name'] == 'Geomodelr', 'Founded Date'] = df['Founded Date Precision']
df.loc[df['Organization Name'] == 'Geomodelr', 'Founded Date Precision'] = df['Last Funding Date']
df.loc[df['Organization Name'] == 'Geomodelr', 'Last Funding Date'] = np.nan

################   Radikal Agency

df.loc[df['Organization Name'] == 'Radikal Agency', 'Industries'] = df['Headquarters Location']
df.loc[df['Organization Name'] == 'Radikal Agency', 'Headquarters Location'] = df['Description']
df.loc[df['Organization Name'] == 'Radikal Agency', 'Description'] = df['CB Rank (Company)']
df.loc[df['Organization Name'] == 'Radikal Agency', 'CB Rank (Company)'] = df['Founded Date']
df.loc[df['Organization Name'] == 'Radikal Agency', 'Founded Date'] = df['Founded Date Precision']
df.loc[df['Organization Name'] == 'Radikal Agency', 'Founded Date Precision'] = df['Last Funding Date']
df.loc[df['Organization Name'] == 'Radikal Agency', 'Last Funding Date'] = np.nan



########################################################################################################
########################################################################################################

# Si se corren las lineas de la 261 a la 303 No es necesario correr
# de la 310 a la 322 y Visceversa 
# Limpieza de caracteres especiales y nombres largos

df= df.replace({"Armenia, Quindio, Colombia":'Armenia'})
df= df.replace({"Barranquilla, Atlantico, Colombia":'Barranquilla'})
df= df.replace({"BogotÃ¡, Distrito Especial, Colombia":'Bogota'})
df= df.replace({"Bogota, Distrito Especial, Colombia":'Bogota'})
df= df.replace({"Tunja, Boyaca, Colombia":'Tunja'})
df= df.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga'})
df= df.replace({"Cali, Valle del Cauca, Colombia":'Cali'})
df= df.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta'})
df= df.replace({"Cartagena, Bolivar, Colombia":'Cartagena'})
df= df.replace({"Cartago, Valle del Cauca, Colombia":'Cartago'})
df= df.replace({"ChÃ­a, Cundinamarca, Colombia":'Chia'})
df= df.replace({"Copacabana, Antioquia, Colombia":'Copacabana'})
df= df.replace({"Cota, Cundinamarca, Colombia":'Cota'})
df= df.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota'})
df= df.replace({"Dosquebradas, Cundinamarca, Colombia":'Dosquebradas'})
df= df.replace({"Duitama, Boyaca, Colombia":'Duitama'})
df= df.replace({"Envigado, Antioquia, Colombia":'Envigado'})
df= df.replace({"Espinal, Tolima, Colombia":'Espinal'})
df= df.replace({"Floridablanca, Santander, Colombia":'Floridablanca'})
df= df.replace({"GarzÃ³n, Huila, Colombia":'Garzon'})
df= df.replace({"Girardot, Cundinamarca, Colombia":'Girardot'})
df= df.replace({"IbaguÃ©, Tolima, Colombia":'Ibague'})
df= df.replace({"ItagÃ¼Ã­, Antioquia, Colombia":'Itagui'})
df= df.replace({"Manizales, Caldas, Colombia":'Manizales'})
df= df.replace({"Mariquita, Tolima, Colombia":'Mariquita'})
df= df.replace({"MedellÃ­n, Antioquia, Colombia":'Medellin'})
df= df.replace({"Montenegro, Quindio, Colombia":'Montenegro'})
df= df.replace({"Mosquera, Cundinamarca, Colombia":'Mosquera'})
df= df.replace({"Neiva, Huila, Colombia, Colombia":'Neiva'})
df= df.replace({"Pasto, Narino, Colombia":'Pasto'})
df= df.replace({"Pereira, Risaralda, Colombia":'Pereira'})
df= df.replace({"PopayÃ¡n, Cordoba, Colombia":'Popayan'})
df= df.replace({"Rionegro, Antioquia, Colombia":'Rionegro'})
df= df.replace({"Sabaneta, Antioquia, Colombia":'Sabaneta'})
df= df.replace({"Santa Marta, Magdalena, Colombia":'Santa Marta'})
df= df.replace({"Santa Rosa De Cabal, Risaralda, Colombia":'Santa Rosa De Cabal'})
df= df.replace({"Santander, Bolivar, Colombia":'Bucaramanga'})
df= df.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali'})
df= df.replace({"Soacha, Cundinamarca, Colombia":'Soacha'})
df= df.replace({"SopÃ³, Cundinamarca, Colombia":'Sopo'})
df= df.replace({"TuluÃ¡, Valle del Cauca, Colombia":'Tulua'})
df= df.replace({"Valledupar, Cesar, Colombia":'Valledupar'})
df= df.replace({"Villavicencio, Meta, Colombia":'Villavicencio'})

#################################################################################################
#################################################################################################

################## METODO CORTO   ##################################################
### Separando Headquarters en 3 columnas se paradas por comas
df= df.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df= df.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df= df.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})


df3=df["Headquarters Location"].str.split(",", n = 2, expand = True) 

####  adicionando las nuevas columnas a df original
df["Ciudad"]= df3[0] 
df["Departamento"]= df3[1] 
df["Pais"]= df3[2]

df = df.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
df = df.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
df = df.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
df = df.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)

# df = df.replace({"CÃºcuta­": 'Cucuta'}) # no funciona

df.loc[df['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"
df= df.replace({"Cundinamarca":'Bogota'})
df= df.replace({"Santander":'Bucaramanga'})
df= df.replace({"Santiago De Cali":'Cali'})

df.to_excel("Colombia-Feb21_Filter.xlsx")

# df= df.replace({", Distrito Especial, Colombia":""},regex=True)
# df= df.replace({"Colombia":""},regex=True)

df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)
df.rename(columns={"Organization Name)": 'organization'}, inplace=True)

# df2 = df[df['CBRank'].notna()]  # elimina solo las filas con datos nan en la columna CBRank

# elimina solo las filas con datos no numericos en la columna CBRank
# df2 = df2[pd.to_numeric(df2['CBRank'], errors='coerce').notnull()]

df2=df
df2['Headquarters Location']=df2['Ciudad']  # ojo es case sensitive
ciudades = df2['Headquarters Location'].unique()

from pandas_profiling import ProfileReport
 
Out_LP = ProfileReport(df2)
Out_LP.to_file('output.html')

################ Graficas Previas en Clase #######################################

# df2=df.dropna() # elimina filas con un o varios campo(s) con nan
df2=df2.set_index('Organization Name')

df2["Last Funding Amount Currency (in USD)"].plot(kind="hist")
plt.xlabel("Histograma Last Funding Amount USD", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show

sorted_by_gross = df2.sort_values(['Last Funding Amount Currency (in USD)'], ascending=False)
print(sorted_by_gross.head(10))
sorted_by_gross['Last Funding Amount Currency (in USD)'].head(10).plot(kind="barh")
plt.xlabel("Last Funding Amount MM USD", fontsize=18)
plt.ylabel("Compañias con mas financiamiento", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

df2.to_excel("Colombia-Feb21_Filter.xlsx")


##################################################################
######################################################## Punto 1

# Cuenta las empresas por cada ciudad
companies_by_city = pd.value_counts(df2['Headquarters Location'])
print(companies_by_city.head(20))

# Hace la grafica de las primeras 20 ciudades con mas empresas
companies_by_city.head(20).plot(kind="barh")
plt.xlabel("Numero de empresas por ciudad", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Top de las primeras 20 ciudades con mas empresas", fontsize=22)
plt.show()

# Grafico por porcentajes
companies_by_perc=100 * df2['Headquarters Location'].value_counts() / len(df2['Headquarters Location'])
companies_by_perc.plot(kind="barh")
plt.xlabel("Numero de empresas por ciudad %", fontsize=18)
plt.xscale('log')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Top de  ciudades con mas empresas - Porcentajes", fontsize=22)
plt.show()
############################################################


##################################################################
######################################################## Punto 2

# Distribucion de ciudades por capital levantado
a=df2.groupby(['Headquarters Location'])['Last Funding Amount Currency (in USD)'].agg('sum')/1000000

# Organiza distribucion de ciudades por capital levantado de mayor a menor
a=a.sort_values(ascending=False)

# grafica las primeras 20 ciudades por capital levantado
a.head(20).plot(kind="barh")
plt.xlabel("Last Funding Amount MMUSD", fontsize=18)
plt.ylabel("Ciudades", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Top de las primeras 20 ciudades", fontsize=20)

# No sirve porque cada dato de last funding lo toma categoria
#c=pd.crosstab(index=df2["Headquarters Location"],columns=df2["Last Funding Amount"],
#              values=df2["Last Funding Amount"], aggfunc='sum')

#plot = pd.crosstab(index=df2['Headquarters Location'],columns=df2['Last Funding Amount']).plot(kind='bar')
#c.plot(kind='bar')
###########################################################


##################################################################
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
plt.xlabel('year', fontsize=20)
plt.ylabel('Last Funding Amount MMUSD', fontsize=20)
plt.title('Last Funding Amount per year per city location', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=14)
plt.show()
######################################################## 


##################################################################
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
plt.xlabel('Industries2 o Sectores', fontsize = 20)
plt.ylabel('Last Fundind Amount MMUSD', fontsize = 20)
plt.title('Last Funding Amount - Industries2 - Headquarters Location', fontsize = 22)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.show()

########################################################


##################################################################
#####################################################  Punto 5

tabla1 = pd.pivot_table(df2, values=["Last Equity Funding Amount Currency (in USD)"], index=["Industries2"],  aggfunc=np.sum)/1e6
tabla1.plot(kind="bar")
plt.xlabel('Industries2 o sectores', fontsize=18)
plt.ylabel('Last Equity Funding Amount Currency (in USD)', fontsize=18)
plt.title('Equity Vs Industries', fontsize=22)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.legend(fontsize=14)
plt.show()
########################################################


##################################################################
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
plt.xlabel('year', fontsize=18)
plt.ylabel('Last Funding Amount MMUSD', fontsize=18)
plt.title('Last Funding Amount per year per Organization Name', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=2)
plt.show()

# otra forma de obtener lo solicitado
c3=c2['Last Funding Amount Currency (in USD)'].groupby('year', group_keys=False).nlargest(15)
########################################################


##################################################################
#####################################################  Grafica Especial Pedida en Clase

tabla5 = pd.pivot_table(df2, values=["Last Funding Amount Currency (in USD)"], index=["year"],aggfunc=np.sum)/1e6
tabla6 = pd.pivot_table(df2, values=["Headquarters Location"], index=["year"],aggfunc="count")

tabla_merge= pd.concat([tabla5, tabla6], 1).dropna().mean(axis=1, level=0)
tabla_merge = tabla_merge.reset_index()

tabla_merge.rename(columns={"Last Funding Amount Currency (in USD)": 'LFA'}, inplace=True)

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(tabla_merge.year, tabla_merge.LFA, color=(190/255,190/255,190/255,0.7), label='Last Funding Amount Currency (in USD)')
ax2.plot(tabla_merge.year, tabla_merge["Headquarters Location"], color='green', label='Headquarters Location')
ax.set_xticklabels(tabla_merge.year, fontsize=18)

ax.legend(loc='best', fontsize=16)
# ax2.legend(loc='best', fontsize=16)
ax.set_xlabel('Años', fontsize=22)
ax.set_ylabel('Total Invertido', fontsize=22)
ax2.set_ylabel('# Negocios', fontsize=22)

ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)

ax2.tick_params(axis = 'both', which = 'major', labelsize = 24)
ax2.tick_params(axis = 'both', which = 'minor', labelsize = 16)

# Annotate Text
#for i, LFA in enumerate(tabla_merge.LFA):
#     ax.text(i, LFA+0.5, round(LFA, 1), horizontalalignment='center')

#for i, val in enumerate(tabla_merge['LFA'].values):
#     ax.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

########################################################

# https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
# https://matplotlib.org/2.0.2/users/pyplot_tutorial.html
# https://relopezbriega.github.io/blog/2016/02/29/analisis-de-datos-categoricos-con-python/
# https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
# https://www.southampton.ac.uk/~feeg1001/notebooks/Matplotlib.html
# https://www.machinelearningplus.com/plots/matplotlib-tutorial-complete-guide-python-plot-examples/
# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html