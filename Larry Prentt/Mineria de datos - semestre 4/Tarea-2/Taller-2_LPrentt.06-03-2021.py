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
import seaborn as sns

os.chdir("D:/4to_Semestre/Mineria de datos/python/Segundo archivo/Data/Data")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

# C_csv_file = 'ColombiaCB-5March21.csv'
# Ch_csv_file = 'ChileCB-5March21.csv'
# M_csv_file = 'MexicoCB-5March21.csv'
# B_csv_file = 'BrazilCB-5March21.csv'
# A_csv_file = 'ArgentinaCB-5March21.csv'
# U_csv_file = 'UruguayCB-5March21.csv'
# df=pd.read_excel(xls_file, header=0,sep=';', index_col=0)

# 1 = Colombia, 2 = Chile, 3 = Brazil, 4 = Mexico, 5 = Argentina
# 6 = Uruguay, 7 = España, 8 = Alemania, 9 = Suiza, 10 = Israel
# 11 = USA

df1=pd.read_csv('ColombiaCB-5March21.csv')
df2=pd.read_csv('ChileCB-5March21.csv')
df3=pd.read_csv('BrazilCB-5March21.csv')
df4=pd.read_csv('MexicoCB-5March21.csv')
df5=pd.read_csv('ArgentinaCB-5March21.csv')
df6=pd.read_csv('UruguayCB-5March21.csv')
df7=pd.read_csv('SpainCB-5March21.csv')
df8=pd.read_csv('GermanyCB-5March21.csv')
df9=pd.read_csv('SwitzerlandCB-5March21.csv')
df10=pd.read_csv('IsraelCB-5March21.csv')
df11=pd.read_csv('USACB-5March21.csv')

Lista = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]

for i in Lista:
    print(i, i.isnull().sum(),)

##  Envia el resultado de nulos a un archivo de texto
with open("file.txt", "w") as output:
    output.write(str(Lista))

### Como pasar una lista a un Excel ?????


# compara los nombres de los 4 primeros dataframes y los nombres comunes van al conjunto a
a=set(df1.columns).intersection(set(df2.columns)).intersection(set(df3.columns)).intersection(set(df4.columns))
# compara los nombres de los 4 siguientes dataframes y los nombres comunes van al conjunto b
b=set(df5.columns).intersection(set(df6.columns)).intersection(set(df7.columns)).intersection(set(df8.columns))
# compara los nombres de los 3 siguientes dataframes y los nombres comunes van al conjunto c
c=set(df9.columns).intersection(set(df10.columns)).intersection(set(df11.columns))
# intersecta los conjuntos a, b y c.  Si la longitud de d es 103
# significa que todos los 11 dataframes tienen los mismos nombres en todas las columnas
d=set(a).intersection(set(b)).intersection(set(c))

if len(d) == 103:
  print("Todos los dataframes tienen los mismos nombres en columnas, se pueden unir")


### Concat, append

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11 ])
df = df.drop(['Organization Name URL'], axis=1) # elimina columna Organization Name URL 

# se renombra nombre columna CB Rank (Company) a CBRank
df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)


df.to_excel("All.xlsx")


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

########################################################

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

################## METODO CORTO   ##################################################
### Separando Headquarters en 3 columnas se paradas por comas
df= df.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df= df.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df= df.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})


########################### Tarea 1 ########################################
############################################################################
######  Separando columna Headquarters Location en 3 columnas
######  Ciudad, Departamento y Pais

df3=df["Headquarters Location"].str.split(",", n = 2, expand = True) 

####  adicionando las nuevas columnas a df original
df["Ciudad"]= df3[0] 
df["Departamento"]= df3[1] 
df["Pais"]= df3[2]
############################################################################


df = df.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
df = df.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
df = df.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
df = df.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)

# df = df.replace({"CÃºcuta­": 'Cucuta'}) # no funciona

df.loc[df['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"
df= df.replace({"Cundinamarca":'Bogota'})
df= df.replace({"Santander":'Bucaramanga'})
df= df.replace({"Santiago De Cali":'Cali'})


# se renombra nombre columna Organization Name a Organization
df.rename(columns={"Organization Name": 'Organization'}, inplace=True)
dfx = df

del dfx["Website"]
del dfx["Twitter"]
del dfx["Facebook"]
del dfx["LinkedIn"]

# CBrank tiene datos no numericos, se convierten a numeros
dfx["CBRank"] = dfx["CBRank"].str.replace(r'\D', '').astype(int)
dfx["CBRank"] = pd.to_numeric(dfx["CBRank"])

dfx[["Number of Articles"]] = dfx[["Number of Articles"]].fillna('') # Specific columns
dfx["Number of Articles"] = dfx["Number of Articles"].str.replace(r'\D', '')
dfx["Number of Articles"] = pd.to_numeric(dfx["Number of Articles"])


dfx[["CB Rank (Organization)"]] = dfx[["CB Rank (Organization)"]].fillna('') # Specific columns
dfx["CB Rank (Organization)"] = dfx["CB Rank (Organization)"].str.replace(r'\D', '')
dfx["CB Rank (Organization)"] = pd.to_numeric(dfx["CB Rank (Organization)"])

dfx[["Apptopia - Downloads Last 30 Days"]] = dfx[["Apptopia - Downloads Last 30 Days"]].fillna('') # Specific columns
dfx["Apptopia - Downloads Last 30 Days"] = dfx["Apptopia - Downloads Last 30 Days"].str.replace(r'\D', '')
dfx["Apptopia - Downloads Last 30 Days"] = pd.to_numeric(dfx["Apptopia - Downloads Last 30 Days"])


dfx=dfx.set_index('Organization')

dfx = dfx.drop(['Contact Email'], axis=1) # elimina columna Contact Email 
dfx = dfx.drop(['Phone Number'], axis=1)  # elimina columna Phone Number
dfx = dfx.drop(['Full Description'], axis=1) # elimina columna Full Description
dfx = dfx.drop(['Transaction Name URL'], axis=1) # elimina columna Transaction Name URL
dfx = dfx.drop(['Acquired by URL'], axis=1) # elimina columna Acquired by URL

dfx.to_excel("All2.xls")

from pandas_profiling import ProfileReport
 
Out_LP = ProfileReport(dfx)
Out_LP.to_file('Profile.html')


#####################################################################
#####################################################################
sorted_by_CBRank = dfx.sort_values(['CBRank'], ascending=False)
sorted_by_CBRank['CBRank'] = sorted_by_CBRank['CBRank']/1000000

print(sorted_by_CBRank.head(10))

sorted_by_CBRank['CBRank'].head(10).plot(kind="barh")
plt.xlabel("MM-CBRank", fontsize=18)
plt.ylabel("Compañias con Mejor Ranking", fontsize=18)
plt.title("Top 10 de Compañias con mejor CBRank", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#################################################################
#### Grafica de Rangos de ingreso por Pais

pd.crosstab(dfx.Pais,dfx['Estimated Revenue Range']).plot(kind='bar')
plt.title('Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Revenue Range', fontsize=20)
#################################################################

#################################################################
#### Grafica de Rangos de ingreso por Pais Normalizados

pd.crosstab(dfx.Pais,dfx['Estimated Revenue Range'],normalize = "index").plot(kind='bar')
plt.title('Normalized Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Percentage of Revenue Range', fontsize=20)
#################################################################


#################################################################
#### Grafica de Rangos de ingreso por Pais & LFA 
pd.crosstab(dfx.Pais,dfx['Estimated Revenue Range'],aggfunc="sum",values=dfx['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Last Funding Amount - Total  per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Last Funding Amount in USD', fontsize=20)
#################################################################

#################################################################
#Top 10 de compañias con mas Financiamiento
sorted_by_gross = dfx.sort_values(['Last Funding Amount Currency (in USD)'], ascending=False)
print(sorted_by_gross.head(10))
sorted_by_gross['Last Funding Amount Currency (in USD)'].head(10).plot(kind="barh")
plt.xlabel("Last Funding Amount MM USD", fontsize=18)
plt.title("Top 10 de Compañias con mas financiamiento", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#######################################################################
companies_by_city = pd.value_counts(dfx['Ciudad'])
print(companies_by_city.head(20))

# Hace la grafica de las primeras 20 ciudades con mas empresas
companies_by_city.head(20).plot(kind="barh")
plt.xlabel("Numero de empresas por ciudad", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Top de las primeras 20 ciudades con mas empresas", fontsize=22)
plt.show()

######################################################################
# Distribucion de ciudades por capital levantado
a=dfx.groupby(['Pais', 'Ciudad'])['Last Funding Amount Currency (in USD)'].agg('sum')/1000000

# Organiza distribucion de ciudades por capital levantado de mayor a menor
a=a.sort_values(ascending=False)

# grafica las primeras 20 ciudades por capital levantado
a.head(20).plot(kind="barh")
plt.xlabel("Last Funding Amount MMUSD", fontsize=18)
plt.ylabel("Ciudades", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title("Top de las primeras 20 ciudades", fontsize=20)


e=dfx.Pais.unique()
for i in range (len(e)):
    print(e[i])

for i in Lista:
    i["Last Funding Amount Currency (in USD)"].plot(kind='hist')
    plt.show()

# Mapa de Calor
dfxcor= dfx[["Number of Lead Investments","Number of Founders","Number of Funding Rounds","Last Funding Amount","Total Funding Amount",'Number of Investors','Number of Acquisitions','Price','Valuation at IPO','Number of Events','Apptopia - Number of Apps',]]
sns.heatmap(dfxcor.corr(),annot=True,cmap="RdYlGn") ######## CLAVE
