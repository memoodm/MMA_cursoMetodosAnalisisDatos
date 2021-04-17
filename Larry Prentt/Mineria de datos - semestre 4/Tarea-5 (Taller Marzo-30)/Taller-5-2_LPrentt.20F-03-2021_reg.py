# -*- coding: utf-8 -*-
"""
Created on Thursday Mar 31 10:30:00 2021

@author: larry Prentt
"""

#import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.chdir("D:/4to_Semestre/Mineria de datos/python/Tercer Archivo")
cwd=os.getcwd()   # asigna a cwd el directorio de trabajo

# C_csv_file = 'ColombiaCB-5March21.csv'
# 1 = Colombia

df=pd.read_csv('ColombiaCB-5March21.csv')
df["Organization Name"]=df["Organization Name"].str.lower()

#df=pd.read_excel(xls_file, header=0,sep=';', index_col=0)
xls_file = 'Top100Startups- Colombia.xlsx'
dftop=pd.read_excel(xls_file)
dftop["Organization"]=dftop["Organization"].str.lower()

xls_file2 = 'Empresas Unicorn - Contactos.xlsx'
dfunicorn = pd.read_excel(xls_file2)
dfunicorn["Name"]=dfunicorn["Name"].str.lower()

# Ojo No usar df1 = df


Lista = [df]

for i in Lista:
    print(i, i.isnull().sum(),)

##  Envia el resultado de nulos a un archivo de texto
with open("fileC.txt", "w") as output:
    output.write(str(Lista))

### Como pasar una lista a un Excel ?????

# compara los nombres de los 4 primeros dataframes y los nombres comunes van al conjunto a
#a=set(df1.columns).intersection(set(df2.columns)).intersection(set(df3.columns)).intersection(set(df4.columns))
# compara los nombres de los 4 siguientes dataframes y los nombres comunes van al conjunto b
#b=set(df5.columns).intersection(set(df6.columns)).intersection(set(df7.columns)).intersection(set(df8.columns))
# compara los nombres de los 3 siguientes dataframes y los nombres comunes van al conjunto c
#c=set(df9.columns).intersection(set(df10.columns)).intersection(set(df11.columns))
# intersecta los conjuntos a, b y c.  Si la longitud de d es 103
# significa que todos los 11 dataframes tienen los mismos nombres en todas las columnas
#d=set(a).intersection(set(b)).intersection(set(c))

#if len(d) == 103:
#  print("Todos los dataframes tienen los mismos nombres en columnas, se pueden unir")
#else:
#    print('hay columnas con diferente nombre en los dataframe, no se pueden unir')

### Se elimina columna Organization Name URL
df = df.drop(['Organization Name URL'], axis=1)

### Se renombra nombre columna CB Rank (Company) a CBRank
df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)

#%%
######## Funcion para determinar el porcentaje de datos nulos

def cols_90per_nulls(data, perc):
    count = 0
    cols_to_drop = {}
    for col in data.columns:
        per_nulls = data[col].isna().sum()/len(data[col])
        if per_nulls >= perc:
            cols_to_drop[col] = per_nulls 
            # print(col, per_nulls)
            count+=1
        else:
            None
    
    print('Number of cols with > ', perc*100, '% nulls:', count)
    return cols_to_drop

# cols_to_drop es un diccionario que tiene los nombres de 
# las Columnas del dataframe con mas del 80% de datos nulos

dict_col_nul=cols_90per_nulls(df, 0.80)
  
# Dataframe data mantiene el dataframe original
data = df

# Dataframe con las columnas eliminadas, en este caso 56 columnas
# con mas del 80% de datos nulos.  Dataframe df pasó de 102 a 46 Columnas
df = df.drop(columns=dict_col_nul)
df.shape

#%%

# df.to_excel("All3.xlsx")

#%%

############## Limpieza de compañias que no son de colombia
########################################################################

# df.drop(df[df['Age'] < 25].index, inplace = True) # Ejemplo encontrado
# df = df[~df['DB Serial'].str.contains('\*', na=False)] # ejemplo encontrado # drop with wildcard
# el ejemplo borra registros de una columna que contenga asterico
# df.drop(df[df['Headquarters Location']=='Bavaria(?!$)'].index, inplace =True) # No funciona

# Junta de Andalucía y Courbox son de Andalucía España.  Se eliminan
df.drop(df[df['Headquarters Location']=='Andalucía, Valle del Cauca, Colombia'].index, inplace =True)

# Neoalgae compañia española, se elimina ciudad Asturias
# Suaval Group compañia española, se elimina ciudad Asturias
df.drop(df[df['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)

# Aspyre Solutions (hay una compañia en USA y otra en UK), aparece con direccion en Barrios
# unidos. Barrios Unidos, Distrito Especial, Colombia Se elimina
df.drop(df[df['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)

# BD Sensors, Pinnery, Ratiokontakt y Atlantik Bruecke son compañias Alemana, 
# aparecen en Bavaria, Cundinamarca, Colombia. se eliminan
# LocalXXL empresa alemana, aparece en Bavaria, Magdalena, Colombia se elimina
df = df[~df['Headquarters Location'].str.contains('Bavaria(?!$)')]
# df.drop(df[df['Headquarters Location']=='Bavaria, Cundinamarca, Colombia'].index, inplace =True)
# df.drop(df[df['Headquarters Location']=='Bavaria, Magdalena, Colombia'].index, inplace =True)

# iFut, Inova House 3D, MÃ¡quina de Vendas, Capital Empreendedora y Acceleratus aparecen
# en Brasilia, Distrito Especial, Colombia.  son compañias de Brasil, se eliminan
df = df[~df['Headquarters Location'].str.contains('Brasilia')]

##########################################
# instruccion para encontrar registros especificos
# M=df[df['Headquarters Location'] == 'Bavaria, Cundinamarca, Colombia']
# M=data_new[data_new['Headquarters Location'] == 'Canada, Cundinamarca, Colombia']
# M=df[df['Headquarters Location'] == 'Canada, Cundinamarca, Colombia']

##########################################

###################################################
# Biometric Update, Boardwalk REIT, ChartBRAIN, LIFT Partners, canada Toronto, calgary
# Pulse Software, compañia australiana, pero aparece canada colombia

# se eliminan todas las Empresas de Canada, Cundinamarca, Colombia, 
# excepto Qinaya y Partsium
# Ejemplo de Drop con 3 condiciones !!!
df=df.drop(df[(df['Headquarters Location'] == 'Canadá, Cundinamarca, Colombia') 
           & ((df['Organization Name'] != 'Qinaya')  & (df['Organization Name'] != 'Partsium'))].index)
# df.drop(df[df['Headquarters Location']=='Canada, Cundinamarca, Colombia'].index, inplace =True)
##########################################################

########################################################
# Ejemplo de Drop con 2 condiciones !!!
# data_new = data.drop(data[(data['col_1'] == 1.0) & (data['col_2'] == 0.0)].index) # Ej.
# data_new=df.drop(df[(df['Headquarters Location'] == 'Canada, Cundinamarca, Colombia') & (df['Organization Name'] != 'Qinaya')].index)
###############################################################


# mRisk, BattleBit, Cencosud Shopping Centers compañias de Chile
# Chile, Huila, Colombia  se Eliminan
df.drop(df[df['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)

# Cocodrilo Dog es de Melbourne y aparece en Cundinamarca, Distrito Especial, Colombia
# se elimina
df.drop(df[df['Organization Name']=='Cocodrilo Dog'].index, inplace =True)

# Homeland Security Careers es de USA y esta en El Paso, Cesar, Colombia
df.drop(df[df['Headquarters Location']=='El Paso, Cesar, Colombia'].index, inplace =True)
# df.drop(df[df['Organization Name']=='Homeland Security Careers'].index, inplace =True)

# WindoTrader USA, aparece como Las Vegas, Sucre, Colombia
# se elimina
df.drop(df[df['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)

# Onyx, Elm, Photogramy, Ferrisland, BeyondROI sede en Los Angeles, Huila, Colombia
# se eliminan
df.drop(df[df['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)

# Peris Costumes, Cositas de España, Esri, Pirsonal, Barrabes, Carousel Group, WORLD COMPLIANCE ASSOCIATION
# Clupik, Mobile Dreams ltd., Acqualia, LoAlkilo, Codekai, Vitriovr, El inmobiliario mes a mes
# Core Business Consulting, Renewable Energy Magazine, datosmacro, 1001talleres, GGBOX
# Puravida Software y Consultia IT con sede en Madrid, Distrito Especial, Colombia
df.drop(df[df['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)

# Advanet (Japon) y Truland Service Corporation en USA. aparecen Maryland, Cundinamarca, Colombia
df.drop(df[df['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)

# POC Network Technologies (TransactRx), Alert Global Media.  sede Miami, Magdalena, Colombia
df.drop(df[df['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)

# Sicartsa sede en Mexico, Huila, Colombia
df.drop(df[df['Headquarters Location']=='México, Huila, Colombia'].index, inplace =True)

# 24marine, Merkadoo sede en Panama, Magdalena, Colombia
df.drop(df[df['Headquarters Location']=='Panamá, Magdalena, Colombia'].index, inplace =True)

# Agros, Downloadperu.com, Mesa 24/7, Dconfianza, Caja Los Andes, Snacks America Latina Peru S.R.L.
# Pandup, Apprende sede en Peru, Valle del Cauca, Colombia
df.drop(df[df['Headquarters Location']=='Perú, Valle del Cauca, Colombia'].index, inplace =True)

# Big Picture Solutions en Florida, Santander, Colombia
df.drop(df[df['Headquarters Location']=='Florida, Santander, Colombia'].index, inplace =True)
# df.drop(df[df['Organization Name']=='Big Picture Solutions'].index, inplace =True)

#%%

##########################################################################
################## Depuracion de nombres de ciudad incorrectos a correctos
##########################################################################

# La compañia Savy tiene sede Usaquen, se cambia a Bogota
df= df.replace({"Usaquén, Distrito Especial, Colombia":'Bogotá, Distrito Especial, Colombia'})
# M=df[df['Organization Name'] == 'Savy']

# Se cambia El Herald por El Heraldo
# Tiene Sede Atlantico, Magdalena, Colombia se cambia a Barranquilla, Atlantico, Colombia
df= df.replace({"El Herald":'El Heraldo'})
df= df.replace({"Atlántico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})

# compañia Monolegal es de tunja y aparece Boyaca, Boyaca, Colombia
df= df.replace({"Boyacá, Boyaca, Colombia":'Tunja, Boyacá, Colombia'})

# Celotor es de cali aparece como Colombiano, Magdalena, Colombia
df= df.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})

# Santiago De Cali, Valle del Cauca, Colombia por Cali
df= df.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})


# Qinaya, compañia colombiana
# https://www.wradio.com.co/noticias/tecnologia/qinaya-el-emprendimiento-que-convierte-cualquier-televisor-en-un-computador/20210301/nota/4113498.aspx
# https://www.youtube.com/watch?v=XBgbwUxkatc
# Canada, Cundinamarca, Colombia vs Bogota, Distrito Especial, Colombia
# Replace with condition
df.loc[(df['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogotá, Distrito Especial, Colombia'

# Partsium
# Partsium, bogota.  El Sitio pone a disposición de los Usuarios un espacio virtual que les permite
# comunicarse mediante el uso de Internet para encontrar una forma de vender o comprar productos y
# servicios. PARTSIUM no es el propietario de los artículos ofrecidos, no tiene posesión de ellos ni
# los ofrece en venta. Los precios de los productos y servicios están sujetos a cambios sin previo aviso.
# website rental to do business

df.loc[(df['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogotá, Distrito Especial, Colombia'
df.loc[(df['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'

# Chiper, SkyFunders, Plastic Surgery Colombia Cias de Bogota y aparecen
# en Cundinamarca, Distrito Especial, Colombia
df= df.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogotá, Distrito Especial, Colombia'})

df= df.replace({"Antioquia, Antioquia, Colombia":'Medellín, Antioquia, Colombia'})
df= df.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df= df.replace({"Cúcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df= df.replace({"Popayán, Cordoba, Colombia":'Popayán, Cauca, Colombia'})


#%%

# df.to_excel("All3.xlsx")

#%%

############################################################################
######  Separando columna Headquarters Location en 3 columnas
######  Ciudad, Departamento y Pais

df3=df["Headquarters Location"].str.split(",", n = 2, expand = True) 

####  adicionando las nuevas columnas a df original
df["Headquarters Location"]= df3[0] 
df["Departamento"]= df3[1] 
df["Pais"]= df3[2]
############################################################################

# Se elimina espacio al inicio y al final de cada string
df.columns = df.columns.str.strip()

# To remove white space at the beginning of string:
# df.columns = df.columns.str.lstrip()
# To remove white space at the end of string:
# df.columns = df.columns.str.rstrip()

# remove special character 
# df.columns = df.columns.str.replace('[#,@,&]', '') 

# se renombra nombre columna Organization Name a Organization
df.rename(columns={"Organization Name": 'Organization'}, inplace=True)

# con estos cambios hasta aquí, se genera dataframe dfx = df
dfx = df.copy()

# Se eliminan las columnas Website, Twitter, Facebook y Linkedin
del dfx["Website"]
del dfx["Twitter"]
del dfx["Facebook"]
del dfx["LinkedIn"]

# CBrank tiene datos no numericos, se convierten a numeros
# dfx["CBRank"] = dfx["CBRank"].str.replace(r'\D', '').astype(int)
dfx["CBRank"] = dfx["CBRank"].str.replace(r'\D', '')
dfx["CBRank"] = pd.to_numeric(dfx["CBRank"])

# Number of Articles tiene datos no numericos, se convierten a numeros
dfx[["Number of Articles"]] = dfx[["Number of Articles"]].fillna('')
dfx["Number of Articles"] = dfx["Number of Articles"].str.replace(r'\D', '')
dfx["Number of Articles"] = pd.to_numeric(dfx["Number of Articles"])

# CBrank Organization tiene datos no numericos, se convierten a numeros
dfx[["CB Rank (Organization)"]] = dfx[["CB Rank (Organization)"]].fillna('')
dfx["CB Rank (Organization)"] = dfx["CB Rank (Organization)"].str.replace(r'\D', '')
dfx["CB Rank (Organization)"] = pd.to_numeric(dfx["CB Rank (Organization)"])

# dfx["BuiltWith - Active Tech Country"] = pd.to_numeric(dfx["BuiltWith - Active Tech Country"])

# set new Index
# dfx=dfx.set_index('Organization')

dfx = dfx.drop(['Contact Email'], axis=1) # elimina columna Contact Email 
dfx = dfx.drop(['Phone Number'], axis=1)  # elimina columna Phone Number
dfx = dfx.drop(['Full Description'], axis=1) # elimina columna Full Description
#dfx = dfx.drop(['Transaction Name URL'], axis=1) # elimina columna Transaction Name URL

#%%
#Funcion que corrige espacios 
def correct_word(word):
    
    new_word = word.split()[0]
    return new_word

#Aplicandon la funcion para la columna Departamento
dfx['Departamento'] = dfx['Departamento'].apply(correct_word)

#Aplicandon la funcion para la columna Pais
dfx['Pais'] = dfx['Pais'].apply(correct_word)

#Cambiar las siguientes 2 variable a formato fecha
dfx['Last Funding Date'] = pd.to_datetime(dfx['Last Funding Date'])
dfx['Founded Date'] = pd.to_datetime(dfx['Founded Date'])

################################################################################
################## CrunchBase vs Top 100

df1 = dfx.copy()
df2 = dfx.copy()


################## CrunchBase vs Top 100
dfx["y"]=0

intersect=set(dfx['Organization']).intersection(set(dftop['Organization']))
len(intersect)

        
for j in intersect:
    dfx.loc[dfx['Organization'] == j, ['y']] = 1


##############################################################################
################## CrunchBase vs Unicorn

df1["y"]=0

intersect2=set(df1['Organization']).intersection(set(dfunicorn['Name']))
len(intersect2)

for t in intersect2:
    df1.loc[df1['Organization'] == t, ['y']] = 1

#############################################################################
################################################################################
################## CrunchBase vs Unicorn vs Top 100

# df2 = df2.drop(['y'], axis=1) # elimina columna Organization Name URL 
df2["y"]=0

intersect3=set(df['Organization']).intersection(set(dftop['Organization'])).intersection(set(dfunicorn['Name']))
len(intersect3)

for l in intersect3:
    df2.loc[df2['Organization'] == l, ['y']] = 1


# df["Organization Name"]=df["Organization Name"].str.upper()

##############################################################################
##### Base de datos Final para Regresion que contiene 3 intersecciones
df2.to_excel("BD_Final.xlsx")


#%%

from pandas_profiling import ProfileReport
 
Out_LP = ProfileReport(df2)
Out_LP.to_file('Profile_Colombia_Sin_Filtro.html')

#%%
# correlaciones de variables numericas
df2.corr(method='pearson')

# Mapa de Calor
sns.heatmap(df2.corr(),annot=True,cmap="RdYlGn") ######## CLAVE

#sns.pairplot(dfx)

#%%

# Headquarters Location                         *
# CBRank                                        
# Headquarters Regions
# Estimated Revenue Range                       *
# Operating Status
# Founded Date
# Founded Date Precision
# Company Type
# Number of Articles                            *
# Industry Groups
# Number of Founders                            *
# Founders                                      No Aplica
# Number of Employees                           *
# Number of Funding Rounds                      *
# Funding Status
# Last Funding Date                             No Aplica
# Last Funding Amount                           No Aplica por tener cantidades en 
#                                               diferentes monedas 
# Last Funding Amount Currency (in USD)         * (si aplica)
# Last Funding Type                             Redundante con LFET se deja LFET
# Last Equity Funding Amount Currency (in USD)  *
# Last Equity Funding Type                      *
# Total Equity Funding Amount Currency (in USD) *
# Total Funding Amount Currency (in USD)        *
# Number of Investors                           *
# IPO Status
# CB Rank (Organization)  igual a CBRank (No es necesario hacerla)
# BuiltWith - Active Tech Count                 *
# G2 Stack - Total Products Active              *
# Departamento                                  No Aplica
# Pais                                          No Aplica

# Grafica 1 Y vs Headquarters Location
pd.crosstab(df2['Headquarters Location'],df2.y).plot(kind='bar')
plt.title('Headquarters Location Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Headquarters Location', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 2 Y vs Headquarters Location
df2.plot.scatter(x='CBRank', y='y')
plt.title('CBRank Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('CBRank', fontsize = 20)
plt.ylabel('Y', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 3 Y vs Headquarters Regions
pd.crosstab(df2['Headquarters Regions'],df2.y).plot(kind='bar')
plt.title('Headquarters Regions Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Headquarters Regions', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 4 Y vs Estimated Revenue Range
pd.crosstab(df2['Estimated Revenue Range'],df2.y).plot(kind='bar')
plt.title('Estimated Revenue Range Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Estimated Revenue Range', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 5 Y vs Operating Status
pd.crosstab(df2['Operating Status'],df2.y).plot(kind='bar')
plt.title('Operating Status Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Operating Status', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 6 Y vs Founded Date
df2.plot.scatter(x='Founded Date', y='y')
# pd.crosstab(df2['Founded Date'],df2.y).plot(kind='bar')
plt.title('Founded Date Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Founded Date', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 7 Y vs Founded Date Precision
pd.crosstab(df2['Founded Date Precision'],df2.y).plot(kind='bar')
plt.title('Founded Date Precision Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Founded Date Precision', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 8 Y vs Company Type
pd.crosstab(df2['Company Type'],df2.y).plot(kind='bar')
plt.title('Company Type Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Company Type', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 9 Y vs Number of Articles
df2.plot.scatter(x='Number of Articles', y='y')
# pd.crosstab(df2['Number of Articles'],df2.y).plot(kind='bar')
plt.title('Number of Articles Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Articles', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.xlim(0, 150)

# Variable Industry Groups necesita tratamiento especial. Posiblemente dividirse en 11
# subcategorias

# Grafica 10 Y vs Number of Founders
df2.plot.scatter(x='Number of Founders', y='y')
# pd.crosstab(df2['Number of Articles'],df2.y).plot(kind='bar')
plt.title('Number of Founders Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Founders', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 10A Y vs Number of Founders
# df2.plot.scatter(x='Number of Founders', y='y')
pd.crosstab(df2['Number of Founders'],df2.y).plot(kind='bar')
plt.title('Number of Founders Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Founders', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 11 Y vs Number of Employees
pd.crosstab(df2['Number of Employees'],df2.y).plot(kind='bar')
plt.title('Number of Employees Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Employees', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 12 Y vs Number of Funding Rounds
# df2.plot.scatter(x='Number of Funding Rounds', y='y')
pd.crosstab(df2['Number of Funding Rounds'],df2.y).plot(kind='bar')
plt.title('Number of Funding Rounds Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Funding Rounds', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 13 Y vs Funding Status
pd.crosstab(df2['Funding Status'],df2.y).plot(kind='bar')
plt.title('Funding Status Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Funding Status', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 14 Y vs Last Equity Funding Amount Currency (in USD)
df2.plot.scatter(x='Last Equity Funding Amount Currency (in USD)', y='y')
plt.title('Last Equity Funding Amount Currency (in USD) Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Last Equity Funding Amount Currency (in USD)', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 15 Y vs Funding Status
pd.crosstab(df2['Last Equity Funding Type'],df2.y).plot(kind='bar')
plt.title('Last Equity Funding Type Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Last Equity Funding Type', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 16 Y vs Total Equity Funding Amount Currency (in USD)
df2.plot.scatter(x='Total Equity Funding Amount Currency (in USD)', y='y')
plt.title('Total Equity Funding Amount Currency (in USD) Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Total Equity Funding Amount Currency (in USD)', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 17 Y vs Total Funding Amount Currency (in USD)
df2.plot.scatter(x='Total Funding Amount Currency (in USD)', y='y')
plt.title('Total Funding Amount Currency (in USD) Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Total Funding Amount Currency (in USD)', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 18 Y vs Number of Investors
df2.plot.scatter(x='Number of Investors', y='y')
#pd.crosstab(df2['Number of Investors'],df2.y).plot(kind='bar')
plt.title('Number of Investors Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Investors', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


# Grafica 19 Y vs Number of Investors
pd.crosstab(df2['IPO Status'],df2.y).plot(kind='bar')
plt.title('IPO Status Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('IPO Status', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)

# Grafica 20 Y vs BuiltWith - Active Tech Count
df2.plot.scatter(x='BuiltWith - Active Tech Count', y='y')
plt.title('BuiltWith - Active Tech Count Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('BuiltWith - Active Tech Count', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


# Grafica 21 Y vs G2 Stack - Total Products Active
df2.plot.scatter(x='G2 Stack - Total Products Active', y='y')
plt.title('G2 Stack - Total Products Active Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('G2 Stack - Total Products Active', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)


#%%

dfy = df2.copy()

dfy = dfy.drop(['Organization'], axis=1)
dfy = dfy.drop(['Industries'], axis=1)
dfy = dfy.drop(['Description'], axis=1)
dfy = dfy.drop(['Headquarters Regions'], axis=1)
dfy = dfy.drop(['Operating Status'], axis=1)
dfy = dfy.drop(['Founded Date'], axis=1)
dfy = dfy.drop(['Founded Date Precision'], axis=1)
dfy = dfy.drop(['Company Type'], axis=1)

dfy = dfy.drop(['Industry Groups'], axis=1)

dfy = dfy.drop(['Founders'], axis=1)
dfy = dfy.drop(['Last Funding Date'], axis=1)
dfy = dfy.drop(['Last Funding Amount'], axis=1)
dfy = dfy.drop(['Last Funding Amount Currency'], axis=1)
dfy = dfy.drop(['Last Funding Type'], axis=1)
dfy = dfy.drop(['Last Equity Funding Amount'], axis=1)
dfy = dfy.drop(['Last Equity Funding Amount Currency'], axis=1)
dfy = dfy.drop(['Total Equity Funding Amount'], axis=1)
dfy = dfy.drop(['Total Equity Funding Amount Currency'], axis=1)
dfy = dfy.drop(['Total Funding Amount'], axis=1)
dfy = dfy.drop(['Total Funding Amount Currency'], axis=1)
dfy = dfy.drop(['Top 5 Investors'], axis=1)
dfy = dfy.drop(['CB Rank (Organization)'], axis=1)
dfy = dfy.drop(['Departamento'], axis=1)
dfy = dfy.drop(['Pais'], axis=1)
dfy = dfy.drop(['IPO Status'], axis=1)
dfy = dfy.drop(['Headquarters Location'], axis=1)


dfz=pd.get_dummies(dfy, columns =['Estimated Revenue Range', 'Number of Employees', 'Funding Status', 'Last Equity Funding Type'])

dfz.to_excel("All4.xlsx")

dfz.corr(method='pearson')

# Mapa de Calor
sns.heatmap(dfz.corr(),annot=True,cmap="RdYlGn") ######## CLAVE

#%%
from pandas_profiling import ProfileReport
 
Out_LP = ProfileReport(dfz)
Out_LP.to_file('Profile_Colombia_Con_Filtro.html')

#%%

# import statsmodels.api as sm

Xtrain = dfz[[dfz.columns[0]]]

for i in range(1,(dfz.shape[1])):
    if dfz.columns[i] != 'y':
        Xtrain = Xtrain.join(dfz[[dfz.columns[i]]])

# Xtrain=Xtrain.dropna()
Xtrain = Xtrain.fillna(0)

#%%

def cols_90per_zeros(data, perc):
    count = 0
    cols_to_drop = {}
    for col in data.columns: 
        per_zeros = data[col][data[col]==0].count()/len(data[col])
        if per_zeros >= perc:
            cols_to_drop[col] = per_zeros 
            # print(col, per_nulls)
            count+=1
        else:
            None
    
    print('Number of cols with > ', perc*100, '% Zeros:', count)
    return cols_to_drop

# este diccionario tiene los nombres de 
# Columnas con mas del 80% de datos con ceros

dict_col_zeros=cols_90per_zeros(Xtrain, 0.95)

#num = 0
#for i in df.columns:
#   num += df[i][df[i]==0].count()
#num


#%%


# Xtrain = Xtrain.drop(['Estimated Revenue Range_$10B+'], axis=1)
# Xtrain = Xtrain.drop(['Last Equity Funding Type_Equity Crowdfunding'], axis=1)
# Xtrain = Xtrain.drop(['Last Equity Funding Type_Post-IPO Equity'], axis=1)
# Xtrain = Xtrain.drop(['Estimated Revenue Range_$100M to $500M'], axis=1)
# Xtrain.columns[12][Xtrain.columns[12]==1].count() ### No funciona
# df.Open[df.Open==0].count()
# Xtrain.CBRank[Xtrain.CBRank==0].count()

X = Xtrain.drop(columns=dict_col_zeros)

X = X.drop(['Number of Employees_1-10'], axis=1)
X = X.drop(['Funding Status_Seed'], axis=1)
X = X.drop(['Last Equity Funding Type_Pre-Seed'], axis=1)
X = X.drop(['Last Equity Funding Type_Seed'], axis=1)

# X = X.drop(['Last Funding Amount Currency (in USD)'], axis=1)
# X = X.drop(['Last Equity Funding Amount Currency (in USD)'], axis=1)
# X = X.drop(['Total Equity Funding Amount Currency (in USD)'], axis=1)
# X = X.drop(['Total Funding Amount Currency (in USD)'], axis=1)

y = dfz[['y']]

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

smt = SMOTE(random_state=0)  # Muestra de entrenamiento balanceado con variable binaria
data_X,data_y=smt.fit_sample(X_train, y_train) # en misma proporcion

sns.countplot(x=data_X['Estimated Revenue Range_$1M to $10M'],data=data_y)
plt.title('No Exitosas vs Exitosas')
plt.show()

import statsmodels.api as sm

log_reg = sm.Logit(data_y, data_X).fit(method='newton', maxiter=200)

print(log_reg.summary())

summary = log_reg.summary()

file = open("summaryF-4-4-21.txt", "w")
str_summary = repr(summary)
file.write("summary = " + str_summary + "\n")
# "\n" creates newline for next write to file

file.close()

#%%

################ Prediction values
yhat = log_reg.predict(X_test)
prediction = list(map(round, yhat))
print('Actual values', list(y_test.values)) 
print('Predictions :', prediction)

from sklearn.metrics import (confusion_matrix,  
                           accuracy_score)
cm = confusion_matrix(y_test, prediction)

print ("Confusion Matrix : \n", cm)
print('Test accuracy = ', accuracy_score(y_test, prediction))
##############################################################

######################## Tested Values
yhatt = log_reg.predict(data_X)
predictiont = list(map(round, yhatt))
cm = confusion_matrix(data_y, predictiont)

print ("Confusion Matrix : \n", cm)
print('Test accuracy = ', accuracy_score(data_y, predictiont))
###############################################################

log_reg.pred_table()  # Igual a procedimiento de Tested Values

#%%%