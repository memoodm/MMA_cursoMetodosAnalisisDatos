# -*- coding: utf-8 -*-
"""
Created on Thursday Marzo 13 10:30:00 2021

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

#%%

############## Limpieza de compañias que no son de colombia
########################################################################

# df.drop(df[df['Age'] < 25].index, inplace = True) # Ejemplo encontrado
# df = df[~df['DB Serial'].str.contains('\*', na=False)] # ejemplo encontrado # drop with wildcard
# el ejemplo borra registros de una columna que contenga asterico
# df.drop(df[df['Headquarters Location']=='Bavaria(?!$)'].index, inplace =True) # No funciona

# Junta de Andalucía y Courbox son de Andalucía España.  Se eliminan
df1.drop(df1[df1['Headquarters Location']=='Andalucía, Valle del Cauca, Colombia'].index, inplace =True)

# Neoalgae compañia española, se elimina ciudad Asturias
# Suaval Group compañia española, se elimina ciudad Asturias
df1.drop(df1[df1['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)

# Aspyre Solutions (hay una compañia en USA y otra en UK), aparece con direccion en Barrios
# unidos. Barrios Unidos, Distrito Especial, Colombia Se elimina
df1.drop(df1[df1['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)

# BD Sensors, Pinnery, Ratiokontakt y Atlantik Bruecke son compañias Alemana, 
# aparecen en Bavaria, Cundinamarca, Colombia. se eliminan
# LocalXXL empresa alemana, aparece en Bavaria, Magdalena, Colombia se elimina
df1 = df1[~df1['Headquarters Location'].str.contains('Bavaria(?!$)')]
# df.drop(df[df['Headquarters Location']=='Bavaria, Cundinamarca, Colombia'].index, inplace =True)
# df.drop(df[df['Headquarters Location']=='Bavaria, Magdalena, Colombia'].index, inplace =True)

# iFut, Inova House 3D, MÃ¡quina de Vendas, Capital Empreendedora y Acceleratus aparecen
# en Brasilia, Distrito Especial, Colombia.  son compañias de Brasil, se eliminan
df1 = df1[~df1['Headquarters Location'].str.contains('Brasilia')]

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
df1=df1.drop(df1[(df1['Headquarters Location'] == 'Canadá, Cundinamarca, Colombia') 
           & ((df1['Organization Name'] != 'Qinaya')  & (df1['Organization Name'] != 'Partsium'))].index)
# df.drop(df[df['Headquarters Location']=='Canada, Cundinamarca, Colombia'].index, inplace =True)
##########################################################

########################################################
# Ejemplo de Drop con 2 condiciones !!!
# data_new = data.drop(data[(data['col_1'] == 1.0) & (data['col_2'] == 0.0)].index) # Ej.
# data_new=df.drop(df[(df['Headquarters Location'] == 'Canada, Cundinamarca, Colombia') & (df['Organization Name'] != 'Qinaya')].index)
###############################################################

# mRisk, BattleBit, Cencosud Shopping Centers compañias de Chile
# Chile, Huila, Colombia  se Eliminan
df1.drop(df1[df1['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)

# Cocodrilo Dog es de Melbourne y aparece en Cundinamarca, Distrito Especial, Colombia
# se elimina
df1.drop(df1[df1['Organization Name']=='Cocodrilo Dog'].index, inplace =True)

# Homeland Security Careers es de USA y esta en El Paso, Cesar, Colombia
df1.drop(df1[df1['Headquarters Location']=='El Paso, Cesar, Colombia'].index, inplace =True)
# df.drop(df[df['Organization Name']=='Homeland Security Careers'].index, inplace =True)

# WindoTrader USA, aparece como Las Vegas, Sucre, Colombia
# se elimina
df1.drop(df1[df1['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)

# Onyx, Elm, Photogramy, Ferrisland, BeyondROI sede en Los Angeles, Huila, Colombia
# se eliminan
df1.drop(df1[df1['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)

# Peris Costumes, Cositas de España, Esri, Pirsonal, Barrabes, Carousel Group, WORLD COMPLIANCE ASSOCIATION
# Clupik, Mobile Dreams ltd., Acqualia, LoAlkilo, Codekai, Vitriovr, El inmobiliario mes a mes
# Core Business Consulting, Renewable Energy Magazine, datosmacro, 1001talleres, GGBOX
# Puravida Software y Consultia IT con sede en Madrid, Distrito Especial, Colombia
df1.drop(df1[df1['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)

# Advanet (Japon) y Truland Service Corporation en USA. aparecen Maryland, Cundinamarca, Colombia
df1.drop(df1[df1['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)

# POC Network Technologies (TransactRx), Alert Global Media.  sede Miami, Magdalena, Colombia
df1.drop(df1[df1['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)

# Sicartsa sede en Mexico, Huila, Colombia
df1.drop(df1[df1['Headquarters Location']=='México, Huila, Colombia'].index, inplace =True)

# 24marine, Merkadoo sede en Panama, Magdalena, Colombia
df1.drop(df1[df1['Headquarters Location']=='Panamá, Magdalena, Colombia'].index, inplace =True)

# Agros, Downloadperu.com, Mesa 24/7, Dconfianza, Caja Los Andes, Snacks America Latina Peru S.R.L.
# Pandup, Apprende sede en Peru, Valle del Cauca, Colombia
df1.drop(df1[df1['Headquarters Location']=='Perú, Valle del Cauca, Colombia'].index, inplace =True)

# Big Picture Solutions en Florida, Santander, Colombia
df1.drop(df1[df1['Headquarters Location']=='Florida, Santander, Colombia'].index, inplace =True)
# df.drop(df[df['Organization Name']=='Big Picture Solutions'].index, inplace =True)

#%%

##########################################################################
################## Depuracion de nombres de ciudad incorrectos a correctos
##########################################################################

# La compañia Savy tiene sede Usaquen, se cambia a Bogota
df1= df1.replace({"Usaquén, Distrito Especial, Colombia":'Bogotá, Distrito Especial, Colombia'})
# M=df[df['Organization Name'] == 'Savy']

# Se cambia El Herald por El Heraldo
# Tiene Sede Atlantico, Magdalena, Colombia se cambia a Barranquilla, Atlantico, Colombia
df1= df1.replace({"El Herald":'El Heraldo'})
df1= df1.replace({"Atlántico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})

# compañia Monolegal es de tunja y aparece Boyaca, Boyaca, Colombia
df1= df1.replace({"Boyacá, Boyaca, Colombia":'Tunja, Boyacá, Colombia'})

# Celotor es de cali aparece como Colombiano, Magdalena, Colombia
df1= df1.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})

# Santiago De Cali, Valle del Cauca, Colombia por Cali
df1= df1.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})

# Qinaya, compañia colombiana
# https://www.wradio.com.co/noticias/tecnologia/qinaya-el-emprendimiento-que-convierte-cualquier-televisor-en-un-computador/20210301/nota/4113498.aspx
# https://www.youtube.com/watch?v=XBgbwUxkatc
# Canada, Cundinamarca, Colombia vs Bogota, Distrito Especial, Colombia
# Replace with condition
df1.loc[(df1['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogotá, Distrito Especial, Colombia'

# Partsium
# Partsium, bogota.  El Sitio pone a disposición de los Usuarios un espacio virtual que les permite
# comunicarse mediante el uso de Internet para encontrar una forma de vender o comprar productos y
# servicios. PARTSIUM no es el propietario de los artículos ofrecidos, no tiene posesión de ellos ni
# los ofrece en venta. Los precios de los productos y servicios están sujetos a cambios sin previo aviso.
# website rental to do business

df1.loc[(df1['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogotá, Distrito Especial, Colombia'
df1.loc[(df1['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'

# Chiper, SkyFunders, Plastic Surgery Colombia Cias de Bogota y aparecen
# en Cundinamarca, Distrito Especial, Colombia
df1= df1.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogotá, Distrito Especial, Colombia'})

df1= df1.replace({"Antioquia, Antioquia, Colombia":'Medellín, Antioquia, Colombia'})
df1= df1.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df1= df1.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df1= df1.replace({"Cúcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df1= df1.replace({"Popayán, Cordoba, Colombia":'Popayán, Cauca, Colombia'})

#%%

Lista = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]

for i in Lista:
    print(i, i.isnull().sum(),)

##  Envia el resultado de nulos a un archivo de texto
with open("file.txt", "w") as output:
    output.write(str(Lista))

### Como pasar una lista a un Excel ?

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
else:
    print('hay columnas con diferente nombre en los dataframe, no se pueden unir')

#%%

### Concat, append

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11 ])
df = df.drop(['Organization Name URL'], axis=1) # elimina columna Organization Name URL 

# se renombra nombre columna CB Rank (Company) a CBRank
df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)

#%%


# Dataframe data mantiene el dataframe original
data = df.copy()

data = data.drop(['Website'], axis=1)
data = data.drop(['Twitter'], axis=1)
data = data.drop(['Facebook'], axis=1)
data = data.drop(['LinkedIn'], axis=1)
data = data.drop(['Contact Email'], axis=1)
data = data.drop(['Phone Number'], axis=1)
data = data.drop(['Full Description'], axis=1)
data = data.drop(['Transaction Name URL'], axis=1)
data = data.drop(['Acquired by URL'], axis=1) 
data = data.drop(['Stock Symbol URL'], axis=1)

data["CBRank"] = data["CBRank"].str.replace(r'\D', '')
data["CBRank"] = pd.to_numeric(data["CBRank"])

# Number of Articles tiene datos no numericos, se convierten a numeros
data[["Number of Articles"]] = data[["Number of Articles"]].fillna('')
data["Number of Articles"] = data["Number of Articles"].str.replace(r'\D', '')
data["Number of Articles"] = pd.to_numeric(data["Number of Articles"])

# CBrank Organization tiene datos no numericos, se convierten a numeros
data[["CB Rank (Organization)"]] = data[["CB Rank (Organization)"]].fillna('')
data["CB Rank (Organization)"] = data["CB Rank (Organization)"].str.replace(r'\D', '')
data["CB Rank (Organization)"] = pd.to_numeric(data["CB Rank (Organization)"])

df3A=data["Headquarters Location"].str.split(",", n = 2, expand = True) 

####  adicionando las nuevas columnas a df original
data["Headquarters Location"]= df3A[0] 
data["Departamento"]= df3A[1] 
data["Pais"]= df3A[2]

data.columns = data.columns.str.strip()

data.rename(columns={"Organization Name": 'Organization'}, inplace=True)

#Cambiar la variable a formato fecha
data['Founded Date'] = pd.to_datetime(data['Founded Date'])
data['Exit Date'] = pd.to_datetime(data['Exit Date'])
data['Closed Date'] = pd.to_datetime(data['Closed Date'])
data['Last Funding Date'] = pd.to_datetime(data['Last Funding Date'])

data['Announced Date'] = pd.to_datetime(data['Announced Date'])
data['IPO Date'] = pd.to_datetime(data['IPO Date'])
data['Delisted Date'] = pd.to_datetime(data['Delisted Date'])
data['Last Leadership Hiring Date'] = pd.to_datetime(data['Last Leadership Hiring Date'])

data['year']= data['Last Funding Date'].dt.year

def correct_word(word):
    
    new_word = word.split()[0]
    return new_word

#Aplicandon la funcion para la columna Departamento
data['Departamento'] = data['Departamento'].apply(correct_word)

#Aplicandon la funcion para la columna Pais
data['Pais'] = data['Pais'].apply(correct_word)

data["Pais"] = data["Pais"].str.lower()
data.loc[(data.Pais == 'united'),'Pais']='usa'

data.to_excel("All50.xlsx")


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

# Dataframe con las columnas eliminadas, en este caso 56 columnas
# con mas del 80% de datos nulos.  Dataframe df pasó de 102 a 46 Columnas
df = df.drop(columns=dict_col_nul)
df.shape


#%%

############################################################################
######  Separando columna Headquarters Location en 3 columnas
######  Ciudad, Departamento y Pais

df3A=df["Headquarters Location"].str.split(",", n = 2, expand = True) 

####  adicionando las nuevas columnas a df original
df["Headquarters Location"]= df3A[0] 
df["Departamento"]= df3A[1] 
df["Pais"]= df3A[2]
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

#Aplicandon la funcion para la columna Departamento
dfx['Departamento'] = dfx['Departamento'].apply(correct_word)

#Aplicandon la funcion para la columna Pais
dfx['Pais'] = dfx['Pais'].apply(correct_word)

#Cambiar la variable a formato fecha
dfx['Last Funding Date'] = pd.to_datetime(dfx['Last Funding Date'])
dfx['Founded Date'] = pd.to_datetime(dfx['Founded Date'])

dfx["Pais"] = dfx["Pais"].str.lower()
dfx.loc[(dfx.Pais == 'united'),'Pais']='usa'
#%%

dfx['Last Funding Date'] = pd.to_datetime(dfx['Last Funding Date'])
dfx['year']= dfx['Last Funding Date'].dt.year

dfx.to_excel("All51.xlsx")

#%%

from pandas_profiling import ProfileReport
 
Out_LP = ProfileReport(dfx)
Out_LP.to_file('World_Profile_dfx.html')

Out_LP = ProfileReport(data)
Out_LP.to_file('World_Profile_data.html')

#%%
# correlaciones de variables numericas
dfx.corr(method='pearson')

# Mapa de Calor
sns.heatmap(dfx.corr(),annot=True,cmap="RdYlGn") ######## CLAVE

#sns.pairplot(dfx)
#%%%%
#########################################
############### Punto-1
### Cuánto capital se ha invertido en LaTAM durante el último año.
### Desagregue gráficamente por país.

#dfx1=data[data.columns[:]][(data["Headquarters Regions"] == 'Latin America') & (data["year"] == 2021)]
dfx1=data[(data["Headquarters Regions"] == 'Latin America') & (data["year"] == 2021)]
dfx1['Last Funding Amount Currency (in USD)']=dfx1['Last Funding Amount Currency (in USD)']/1e6

pd.crosstab(dfx1.Pais,dfx1['year'],aggfunc="sum",values=dfx1['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Last Funding Amount - Total  per Country in 2021', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Last Funding Amount in MM - USD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html

############### Punto-2
dfx2=data[data.columns[:]][(data["Headquarters Regions"] == 'Latin America') & (data["year"] >= 2016)]
dfx2['Last Funding Amount Currency (in USD)']=dfx2['Last Funding Amount Currency (in USD)']/1e6

# Last Funding Amount Currency (in USD)
pd.crosstab(dfx2.Pais,dfx2['year'],aggfunc="sum",values=dfx2['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Last Funding Amount - Total  per Country in the last 6 years', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Last Funding Amount in MM - USD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Number of Articles
pd.crosstab(dfx2.Pais,dfx2['year'],aggfunc="sum",values=dfx2['Number of Articles']).plot(kind='bar')
plt.title('Number of Articles - Total  per Country in the last 6 years', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Number of Articles - Sum', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#################################################################
#### Grafica de Rangos de ingreso por Pais

pd.crosstab(data.Pais,data['Estimated Revenue Range']).plot(kind='bar')
plt.title('Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Revenue Range', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#################################################################

#################################################################
#### Grafica de Rangos de ingreso por Pais Normalizados

pd.crosstab(data.Pais,data['Estimated Revenue Range'],normalize = "index").plot(kind='bar')
plt.title('Normalized Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Percentage of Revenue Range', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#################################################################


#################################################################
#### Grafica de Rangos de ingreso por Pais & LFA 
pd.crosstab(data.Pais,data['Estimated Revenue Range'],aggfunc="sum",values=data['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Last Funding Amount - Total  per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Last Funding Amount in USD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#################################################################

#%%
################################################################
# LFA Normalizado per capita por pais y rango de ingreso de empresas

w=pd.crosstab(data.Pais,data['Estimated Revenue Range'],aggfunc="sum",values=data['Last Funding Amount Currency (in USD)'])

e=w.index.unique()


from countryinfo import CountryInfo
population = np.zeros(len(e))

for i in range (len(e)):
    nombrepais=str(e[i])
    country=CountryInfo(nombrepais)
    population[i] = country.population()

# country2 = CountryInfo('brazil')
# country2.population()

for j in range (len(e)):
    w.iloc[[j]]=w.iloc[[j]]/population[j]

w.plot(kind='bar')
plt.title('Total Last Funding Amount Normalized per capita per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Normalized Last Funding Amount in USD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
################################################################
#%%        

#################################################################
#### Punto - 3
dfx3=data[data["Pais"] == 'colombia' ]
pd.crosstab(dfx3["Pais"],dfx3['Investor Type']).plot(kind='bar')
plt.title('Investor Type per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

dfx3=data[data["Pais"] == 'colombia' ]
pd.crosstab(dfx3['Investor Type'],dfx3['Last Funding Amount Currency (in USD)'] ).plot(kind='bar')
plt.title('Investor Type per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

pd.crosstab(dfx3.Pais,dfx3['Investor Type'],aggfunc="sum",values=dfx3['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Investor Type per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Funding per Investor Type -  MM-USD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

print('los inversionistas prefieren los siguientes grupos de industrias')

lista_inv = dfx3[dfx3.columns[30:31]][(dfx3["Investor Type"] != np.nan)]
lista_inv = dfx3[dfx3.columns[30:31]][(dfx3.index != np.nan)]
lista_inv = lista_inv.dropna()

################################################################
#### Punto-4
pd.crosstab(dfx3['Organization'],dfx3['Number of Exits']).plot(kind='bar')
plt.title('Numero de Exit de Capital privado en Colombia', fontsize=22)
plt.xlabel('Empresa', fontsize=20)
plt.ylabel('# de Exit de Capital', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


splot =sns.scatterplot(data=data, x='Total Equity Funding Amount Currency (in USD)', y='Number of Exits', hue='Pais')
splot.set(xscale="log")
splot.set(yscale="log")

data.rename(columns={"Total Equity Funding Amount Currency (in USD)": 'TEFA (USD)'}, inplace=True)
sns.relplot(
    data=data, x='TEFA (USD)', y='Number of Exits',
    col="Pais", hue="year", style="year",
    kind="scatter"
)


# https://seaborn.pydata.org/examples/scatter_bubbles.html
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
# https://stackoverflow.com/questions/43061768/plotting-multiple-scatter-plots-pandas
# https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot

# sns.scatterplot(data=data, x='Total Equity Funding Amount Currency (in USD)', y='Number of Exits', hue='Pais', style="Pais")

################################################################
#### Punto-5

dfz=pd.get_dummies(data, columns =['Estimated Revenue Range'])

dfz.to_excel("AllD.xlsx")

# tabla1=pd.crosstab(dfz.Pais,dfz.Pais,aggfunc="sum",values=dfz['Estimated Revenue Range_$100M to $500M'])
# tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']], columns=dfz['Estimated Revenue Range_$100M to $500M'], margins=True)

# 1
#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_Less than $1M'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_Less than $1M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_Less than $1M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_Less than $1M'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge1= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge1=tabla_merge1.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge1=tabla_merge1[tabla_merge1.columns[:]][(tabla_merge1["year"] >= 2016) & (tabla_merge1["year"] <= 2020)]
tabla_merge1['Last Funding Amount Currency (in USD) < 1M']=tabla_merge1['Last Funding Amount Currency (in USD) < 1M']/1e6
tabla_merge1['Ratio']=tabla_merge1['Estimated Revenue Range_Less than $1M']*1/tabla_merge1['Last Funding Amount Currency (in USD) < 1M']

tabla_merge1['Growth < 1M']=0.0
tabla_merge1 = tabla_merge1.fillna(0)
k = int(tabla_merge1.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge1.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge1.Ratio.values[i+k*j]
            b1=tabla_merge1.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge1.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge1.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge1.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge1['Growth < 1M']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso < 1M / LFA < 1M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

# 2
#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$1M to $10M'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$1M to $10M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$1M to $10M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$1M to $10M'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) 1M-10M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge2= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge2=tabla_merge2.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge2=tabla_merge2[tabla_merge2.columns[:]][(tabla_merge2["year"] >= 2016) & (tabla_merge2["year"] <= 2020)]
tabla_merge2['Last Funding Amount Currency (in USD) 1M-10M']=tabla_merge2['Last Funding Amount Currency (in USD) 1M-10M']/1e6
tabla_merge2['Ratio']=tabla_merge2['Estimated Revenue Range_$1M to $10M']*5.5/tabla_merge2['Last Funding Amount Currency (in USD) 1M-10M']

tabla_merge2['Growth 1M-10M']=0.0
tabla_merge2 = tabla_merge2.fillna(0)
k = int(tabla_merge2.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge2.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge2.Ratio.values[i+k*j]
            b1=tabla_merge2.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge2.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge2.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge2.Pais,tabla_merge2.year
            ,aggfunc="mean",values=tabla_merge2['Growth 1M-10M']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 1M-10M / LFA 1M-10M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#################################################################################################

# 3
#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$10M to $50M'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$10M to $50M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$10M to $50M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$10M to $50M'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) 10M-50M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge3= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge3=tabla_merge3.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge3=tabla_merge3[tabla_merge3.columns[:]][(tabla_merge3["year"] >= 2016) & (tabla_merge3["year"] <= 2020)]
tabla_merge3['Last Funding Amount Currency (in USD) 10M-50M']=tabla_merge3['Last Funding Amount Currency (in USD) 10M-50M']/1e6
tabla_merge3['Ratio']=tabla_merge3['Estimated Revenue Range_$10M to $50M']*30/tabla_merge3['Last Funding Amount Currency (in USD) 10M-50M']

tabla_merge3['Growth 10M-50M']=0.0
tabla_merge3 = tabla_merge3.fillna(0)
k = int(tabla_merge3.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge3.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge3.Ratio.values[i+k*j]
            b1=tabla_merge3.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge3.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge3.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge3.Pais,tabla_merge3.year
            ,aggfunc="mean",values=tabla_merge3['Growth 10M-50M']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 10M-50M / LFA 10M-50M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

# 4
#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$50M to $100M'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$50M to $100M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$50M to $100M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$50M to $100M'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) 50M-100M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge4= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge4=tabla_merge4.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge4=tabla_merge4[tabla_merge4.columns[:]][(tabla_merge4["year"] >= 2016) & (tabla_merge4["year"] <= 2020)]
tabla_merge4['Last Funding Amount Currency (in USD) 50M-100M']=tabla_merge4['Last Funding Amount Currency (in USD) 50M-100M']/1e6
tabla_merge4['Ratio']=tabla_merge4['Estimated Revenue Range_$50M to $100M']*75/tabla_merge4['Last Funding Amount Currency (in USD) 50M-100M']

tabla_merge4['Growth 50M-100M']=0.0
tabla_merge4 = tabla_merge4.fillna(0)
k = int(tabla_merge4.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge4.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge4.Ratio.values[i+k*j]
            b1=tabla_merge4.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge4.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge4.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge4.Pais,tabla_merge4.year
            ,aggfunc="mean",values=tabla_merge4['Growth 50M-100M']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 50M-100M / LFA 50M-100M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

# 5

#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$100M to $500M'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$100M to $500M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$100M to $500M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$100M to $500M'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) 100M-500M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge5= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge5=tabla_merge5.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge5=tabla_merge5[tabla_merge5.columns[:]][(tabla_merge5["year"] >= 2016) & (tabla_merge5["year"] <= 2020)]
tabla_merge5['Last Funding Amount Currency (in USD) 100M-500M']=tabla_merge5['Last Funding Amount Currency (in USD) 100M-500M']/1e6
tabla_merge5['Ratio']=tabla_merge5['Estimated Revenue Range_$100M to $500M']*300/tabla_merge5['Last Funding Amount Currency (in USD) 100M-500M']

tabla_merge5['Growth 100M-500M']=0.0
tabla_merge5 = tabla_merge5.fillna(0)
k = int(tabla_merge5.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge5.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge5.Ratio.values[i+k*j]
            b1=tabla_merge5.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge5.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge5.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge5.Pais,tabla_merge5.year
            ,aggfunc="mean",values=tabla_merge5['Growth 100M-500M']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 100M-500M / LFA 100M-500M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

## 6

#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$500M to $1B'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$500M to $1B'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$500M to $1B'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$500M to $1B'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) 500M-1B'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge6= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge6=tabla_merge6.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge6=tabla_merge6[tabla_merge6.columns[:]][(tabla_merge6["year"] >= 2016) & (tabla_merge6["year"] <= 2020)]
tabla_merge6['Last Funding Amount Currency (in USD) 500M-1B']=tabla_merge6['Last Funding Amount Currency (in USD) 500M-1B']/1e6
tabla_merge6['Ratio']=tabla_merge6['Estimated Revenue Range_$500M to $1B']*750/tabla_merge6['Last Funding Amount Currency (in USD) 500M-1B']

tabla_merge6['Growth 500M-1B']=0.0
tabla_merge6 = tabla_merge6.fillna(0)
k = int(tabla_merge6.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge6.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge6.Ratio.values[i+k*j]
            b1=tabla_merge6.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge6.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge6.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge6.Pais,tabla_merge6.year
            ,aggfunc="mean",values=tabla_merge6['Growth 500M-1B']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 500M-1B / LFA 500M-1B', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

## 7

#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$1B to $10B'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$1B to $10B'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$1B to $10B'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$1B to $10B'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD)1B to $10B'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge7= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge7=tabla_merge7.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge7=tabla_merge7[tabla_merge7.columns[:]][(tabla_merge7["year"] >= 2016) & (tabla_merge7["year"] <= 2020)]
tabla_merge7['Last Funding Amount Currency (in USD)1B to $10B']=tabla_merge7['Last Funding Amount Currency (in USD)1B to $10B']/1e6
tabla_merge7['Ratio']=tabla_merge7['Estimated Revenue Range_$1B to $10B']*5500/tabla_merge7['Last Funding Amount Currency (in USD)1B to $10B']

tabla_merge7['Growth 1B to $10B']=0.0
tabla_merge7 = tabla_merge7.fillna(0)
k = int(tabla_merge7.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge7.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge7.Ratio.values[i+k*j]
            b1=tabla_merge7.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge7.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge7.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge7.Pais,tabla_merge7.year
            ,aggfunc="mean",values=tabla_merge7['Growth 1B to $10B']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso 1B to $10B / LFA 1B to $10B', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

## 8

#################################################################################################
tabla1 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$10B+'],
                     aggfunc="sum",values=dfz['Estimated Revenue Range_$10B+'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$10B+'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dfz['Pais'],dfz['year']],
                     columns=dfz['Estimated Revenue Range_$10B+'],
                     aggfunc="sum",values=dfz['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD)1B > $10B'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge8= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge8=tabla_merge8.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge8=tabla_merge8[tabla_merge8.columns[:]][(tabla_merge8["year"] >= 2016) & (tabla_merge8["year"] <= 2020)]
tabla_merge8['Last Funding Amount Currency (in USD)1B > $10B']=tabla_merge8['Last Funding Amount Currency (in USD)1B > $10B']/1e6
tabla_merge8['Ratio']=tabla_merge8['Estimated Revenue Range_$10B+']*10000/tabla_merge8['Last Funding Amount Currency (in USD)1B > $10B']

tabla_merge8['Growth > $10B']=0.0
tabla_merge8 = tabla_merge8.fillna(0)
k = int(tabla_merge8.shape[0]/11)
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_merge8.iloc[i+k*j:i+k*j+1, k:k+1] = 0
        else:
            a1=tabla_merge8.Ratio.values[i+k*j]
            b1=tabla_merge8.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_merge8.iloc[i+k*j:i+k*j+1, k:k+1]=0
            else:
                tabla_merge8.iloc[i+k*j:i+k*j+1, k:k+1] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_merge8.Pais,tabla_merge8.year
            ,aggfunc="mean",values=tabla_merge8['Growth > $10B']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso > $10B / LFA > $10B', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################


tabla_merge1.set_index(['Pais', 'year'], inplace=True)
tabla_merge2.set_index(['Pais', 'year'], inplace=True)
tabla_merge3.set_index(['Pais', 'year'], inplace=True)
tabla_merge4.set_index(['Pais', 'year'], inplace=True)
tabla_merge5.set_index(['Pais', 'year'], inplace=True)
tabla_merge6.set_index(['Pais', 'year'], inplace=True)
tabla_merge7.set_index(['Pais', 'year'], inplace=True)
tabla_merge8.set_index(['Pais', 'year'], inplace=True)

tabla_mergef= pd.concat([tabla_merge1, tabla_merge2, tabla_merge3, tabla_merge4,
                         tabla_merge5, tabla_merge6, tabla_merge7, tabla_merge8], axis=1).reindex(tabla_merge1.index)


tabla_mergef= tabla_mergef.drop(['Ratio'], axis=1) # elimina columna Organization Name URL 
tabla_mergef.to_excel("Merge.xlsx")

tabla_mergef=tabla_mergef.reset_index()

tabla_mergef['ERR']=tabla_mergef['Estimated Revenue Range_Less than $1M']*1+ tabla_mergef['Estimated Revenue Range_$1M to $10M']*5.5 + tabla_mergef['Estimated Revenue Range_$10M to $50M']*30 +tabla_mergef['Estimated Revenue Range_$50M to $100M']*75 + tabla_mergef['Estimated Revenue Range_$100M to $500M']*300 + tabla_mergef['Estimated Revenue Range_$500M to $1B']*750 +  tabla_mergef['Estimated Revenue Range_$1B to $10B']*5500 + tabla_mergef['Estimated Revenue Range_$10B+']*10000 

tabla_mergef['LFA']=tabla_mergef['Last Funding Amount Currency (in USD) < 1M']+ tabla_mergef['Last Funding Amount Currency (in USD) 1M-10M'] + tabla_mergef['Last Funding Amount Currency (in USD) 10M-50M'] +tabla_mergef['Last Funding Amount Currency (in USD) 50M-100M'] + tabla_mergef['Last Funding Amount Currency (in USD) 100M-500M'] + tabla_mergef['Last Funding Amount Currency (in USD) 500M-1B'] +  tabla_mergef['Last Funding Amount Currency (in USD)1B to $10B'] + tabla_mergef['Last Funding Amount Currency (in USD)1B > $10B'] 

tabla_mergef['Ratio']=tabla_mergef['ERR']/tabla_mergef['LFA']
tabla_mergef['Growth']=0.0
for j in range(11):
    for i in range(k):
        if i == 0:
            tabla_mergef.iloc[i+k*j:i+k*j+1, 29:30] = 0
        else:
            a1=tabla_mergef.Ratio.values[i+k*j]
            b1=tabla_mergef.Ratio.values[i+k*j-1]
            if b1==0:
                tabla_mergef.iloc[i+k*j:i+k*j+1, 29:30]=0
            else:
                tabla_mergef.iloc[i+k*j:i+k*j+1, 29:30] = ((a1-b1)/b1+1)**(1/12)-1
            
pd.crosstab(tabla_mergef.Pais,tabla_mergef.year
            ,aggfunc="mean",values=tabla_mergef['Growth']).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso Total / LFA Total', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


















###############################################
# https://support.crunchbase.com/hc/en-us/articles/115010624128-Glossary-of-Investor-Types ###
###############################################

# https://support.crunchbase.com/hc/en-us/articles/115010477187-Crunchbase-Rank-CB-Rank-

# https://www.javaer101.com/es/article/54436158.html

# conditional crosstab pandas

###################################
# https://towardsdatascience.com/meet-the-hardest-functions-of-pandas-part-ii-f8029a2b0c9b
###################################

# https://stackoverflow.com/questions/41730267/cross-tab-using-conditional-sub-populations


#%%
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

##### codigo 3 variables
#################################################################
#### Grafica de Rangos de ingreso por Pais & LFA 
w=pd.crosstab(dfx.Pais,dfx['Estimated Revenue Range'],aggfunc="sum",values=dfx['Last Funding Amount Currency (in USD)'])

w2=dfx.groupby(['Pais', 'Estimated Revenue Range'])['Last Funding Amount Currency (in USD)'].agg('sum')
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

#### Bacana
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

#%%        
#########################
##############


# dfy=dfx.loc[(dfx["Pais"] == 'colombia') & (dfx["Pais"] == 'mexico'), "A"]

dfy = dfx[(dfx["Pais"] == 'colombia') | (dfx["Pais"] == 'mexico')]

sns.displot(
    dfy, x="Last Funding Amount Currency (in USD)", col="Pais", row="Number of Employees",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
)

#######################################
from random import randint
dft = pd.DataFrame({'A': [randint(1, 9) for x in range(10)],
                   'B': [randint(1, 9)*10 for x in range(10)],
                   'C': [randint(1, 9)*100 for x in range(10)]})

ss=dft[dft.columns[:]][(dft["B"] > 30) & (dft["C"] == 900)]

##############################################################

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