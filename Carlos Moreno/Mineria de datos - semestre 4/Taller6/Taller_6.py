#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 22:26:34 2021
Técnicas Avanzadas de Minería de 
Datos y Machine Learning

            ACTIVIDAD CLASE 6

Miller Alexander Quiroga Campos
@author: milleralexanderquirogacampos
"""

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import os

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn import preprocessing # procesamiento de datos
from pandas_profiling import ProfileReport
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%% importamos archivos de excel y csv y los convertimos en df1

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Taller6")
cwd = os.getcwd()

#df1 ColombiaCB
a = pd.read_csv('ColombiaCB-5March21.csv')#, index_col = 0)
#df1 Top100Startups
b = pd.read_excel('Top100Startups- Colombia.xlsx')#, index_col = 0)
#df1 Empresas Unicorn
c = pd.read_excel('Empresas Unicorn - Contactos.xlsx')#, index_col = 0)

print(a.head(5))
print(a.tail(5))

print(b.head(5))
print(b.tail(5))

print(c.head(5))
print(c.tail(5))

#Estadística de las variables cuantitativas
a.describe()

#%%

a1 = a.join(b) #, rsuffix='_right')
df = a1.join(c, rsuffix='_right')

df['Intersección'] = 0
df.shape

df["Organization Name"] = df["Organization Name"].str.upper()
df["Co"] = df["Co"].str.upper()
df["Name"] = df["Name"].str.upper()

Int = set(df['Organization Name']).intersection(set(df['Co']))
Int2 = set(df['Organization Name']).intersection(set(df['Name']))
Int3 = set(df['Co']).intersection(set(df['Name']))
Int4 = set(df['Co']).intersection(set(df['Name'])).intersection(set(df['Organization Name']))

for i in Int4:
    df.loc[df['Organization Name'] == i, ['Intersección']] = 1

#%%     LIMPIEZA DE LA DATA

df.drop(df[df['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='Albania, La Guajira, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='AndalucÃ­a, Valle del Cauca, Colombia'].index, inplace =True)
df.drop(df[df['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)
df = df[~df['Headquarters Location'].str.contains('Bavaria(?!$)')]
df = df[~df['Headquarters Location'].str.contains('Brasilia')]

# Se eliminan todas las Empresas de Canada, Cundinamarca, Colombia, excepto Qinaya y Partsium
df=df.drop(df[(df['Headquarters Location'] == 'CanadÃ¡, Cundinamarca, Colombia') 
           & ((df['Organization Name'] != 'Qinaya')  & (df['Organization Name'] != 'Partsium'))].index)

# mRisk, BattleBit, Cencosud Shopping Centers compañias de Chile
# Chile, Huila, Colombia  se Eliminan
df.drop(df[df['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)

# WindoTrader USA, aparece como Las Vegas, Sucre, Colombia
# se elimina

df.drop(df[df['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)

# Onyx, Elm, Photogramy, Ferrisland, BeyondROI sede en Los Angeles, Huila, Colombia
# se eliminan
df.drop(df[df['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)

#Peris Costumes, Cositas de EspaÃ±a, Esri, Pirsonal, Barrabes, Carousel Group, WORLD COMPLIANCE ASSOCIATION
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

#%%
##Depuracion de nombres de ciudad incorrectos a correctos

# La compañia Savy tiene sede usaquen, se cambia a Bogota

df = df.replace({"UsaquÃ©n, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})

# M=df[df['Organization Name'] == 'Savy']

# PagomÃ­o  es compania de medellin y aparece con sede
# Antioquia, Antioquia, Colombia

df = df.replace({"Antioquia, Antioquia, Colombia":'MedellÃ­n, Antioquia, Colombia'})

# El Herald  El Heraldo sede Atlantico, Magdalena, Colombia  Es Barranquilla
# Atlantico, Magdalena, Colombia
# 

df = df.replace({"El Herald":'El Heraldo'})
df = df.replace({"AtlÃ¡ntico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})


# compañia Monolegal es de tunja y aparece Boyaca, Boyaca, Colombia

df = df.replace({"BoyacÃ¡, Boyaca, Colombia":'Tunja, Boyaca, Colombia'})

# Celotor es de cali aparece como Colombiano, Magdalena, Colombia

df = df.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})

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

df = df.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df = df.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df = df.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df = df.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df = df.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})

df = df.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
df = df.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
df = df.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
df = df.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)

# df = df.replace({"CÃºcuta­": 'Cucuta'}) # no funciona

df.loc[df['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"
df = df.replace({"Cundinamarca":'Bogota'})
df = df.replace({"Santander":'Bucaramanga'})
df = df.replace({"Santiago De Cali":'Cali'})


#F["CB Rank (Organization)"] = F["CB Rank (Organization)"].str.replace(r'\D', '').astype(int)
#F["CB Rank (Organization)"] = pd.to_numeric(F["CB Rank (Organization)"])

df[["Number of Articles"]] = df[["Number of Articles"]].fillna('') # Specific columns
df["Number of Articles"] = df["Number of Articles"].str.replace(r'\D', '')
df["Number of Articles"] = pd.to_numeric(df["Number of Articles"])

df[["CB Rank (Organization)"]] = df[["CB Rank (Organization)"]].fillna('') # Specific columns
df["CB Rank (Organization)"] = df["CB Rank (Organization)"].str.replace(r'\D', '')
df["CB Rank (Organization)"] = pd.to_numeric(df["CB Rank (Organization)"])

df[["Apptopia - Downloads Last 30 Days"]] = df[["Apptopia - Downloads Last 30 Days"]].fillna('') # Specific columns
df["Apptopia - Downloads Last 30 Days"] = df["Apptopia - Downloads Last 30 Days"].str.replace(r'\D', '')
df["Apptopia - Downloads Last 30 Days"] = pd.to_numeric(df["Apptopia - Downloads Last 30 Days"])

df = df.drop(['Contact Email'], axis=1) # elimina columna Contact Email 
df = df.drop(['Phone Number'], axis=1)  # elimina columna Phone Number
df = df.drop(['Full Description'], axis=1) # elimina columna Full Description
df = df.drop(['Transaction Name URL'], axis=1) # elimina columna Transaction Name URL
df = df.drop(['Acquired by URL'], axis=1) # elimina columna Acquired by URL
df = df.drop(['Exit Date'], axis = 1)
df = df.drop(['Exit Date Precision'], axis = 1)
df = df.drop(['Closed Date'], axis = 1)
df = df.drop(['Website'], axis = 1)
df = df.drop(['Twitter'], axis = 1)
df = df.drop(['Facebook'], axis = 1)
df = df.drop(['LinkedIn'], axis = 1)
df = df.drop(['Hub Tags'], axis = 1)
df = df.drop(['Investor Type'], axis = 1)
df = df.drop(['Investment Stage'], axis = 1)
df = df.drop(['Number of Portfolio Organizations'], axis = 1)
df = df.drop(['Number of Investments'], axis = 1)
df = df.drop(['Number of Lead Investments'], axis = 1)
df = df.drop(['Number of Exits'], axis = 1)
df = df.drop(['Number of Exits (IPO)'], axis = 1)
df = df.drop(['Accelerator Program Type'], axis = 1)
df = df.drop(['Accelerator Duration (in weeks)'], axis = 1)
df = df.drop(['School Type'], axis = 1)
df = df.drop(['School Program'], axis = 1)
df = df.drop(['Number of Enrollments'], axis = 1)
df = df.drop(['Number of Founders (Alumni)'], axis = 1)
df = df.drop(['Number of Acquisitions'], axis = 1)
df = df.drop(['Acquisition Status'], axis = 1)
df = df.drop(['Transaction Name'], axis = 1)
df = df.drop(['Acquired by'], axis = 1)
df = df.drop(['Announced Date'], axis = 1)
df = df.drop(['Announced Date Precision'], axis = 1)
df = df.drop(['Price'], axis = 1)
df = df.drop(['Price Currency'], axis = 1)
df = df.drop(['Price Currency (in USD)'], axis = 1)
df = df.drop(['Acquisition Type'], axis = 1)
df = df.drop(['Acquisition Terms'], axis = 1)
df = df.drop(['IPO Date'], axis = 1)
df = df.drop(['Delisted Date'], axis = 1)
df = df.drop(['Delisted Date Precision'], axis = 1)
df = df.drop(['Money Raised at IPO'], axis = 1)
df = df.drop(['Money Raised at IPO Currency'], axis = 1)
df = df.drop(['Money Raised at IPO Currency (in USD)'], axis = 1)
df = df.drop(['Valuation at IPO'], axis = 1)
df = df.drop(['Valuation at IPO Currency'], axis = 1)
df = df.drop(['Valuation at IPO Currency (in USD)'], axis = 1)
df = df.drop(['Stock Symbol'], axis = 1)
df = df.drop(['Stock Symbol URL'], axis = 1)
df = df.drop(['Stock Exchange'], axis = 1)
df = df.drop(['Last Leadership Hiring Date'], axis = 1)
df = df.drop(['Number of Events'], axis = 1)
df = df.drop(['Apptopia - Number of Apps'], axis = 1)
df = df.drop(['Apptopia - Downloads Last 30 Days'], axis = 1)
df = df.drop(['IPqwery - Patents Granted'], axis = 1)
df = df.drop(['IPqwery - Trademarks Registered'], axis = 1)
df = df.drop(['IPqwery - Most Popular Patent Class'], axis = 1)
df = df.drop(['IPqwery - Most Popular Trademark Class'], axis = 1)
df = df.drop(['Aberdeen - IT Spend'], axis = 1)
df = df.drop(['Aberdeen - IT Spend Currency'], axis = 1)
df = df.drop(['Aberdeen - IT Spend Currency (in USD)'], axis = 1)
df = df.drop(['School Method'], axis = 1)
df = df.drop(['No'], axis = 1)
df = df.drop(['Em'], axis = 1)
df = df.drop(['NIT'], axis = 1)
df = df.drop(['CORREO ELECTRONICO'], axis = 1)
df = df.drop(['TELÉFONO '], axis = 1)
df = df.drop(['Unnamed: 16'], axis = 1)
df = df.drop(['Ciudad'], axis = 1)
df = df.drop(['Closed Date Precision'], axis = 1)
df = df.drop(['Organization Name URL'], axis = 1)
df = df.drop(['Headquarters Regions'], axis = 1)
df = df.drop(['Founded Date_right'], axis = 1)
df = df.drop(['Last Funding Date_right'], axis = 1)
# df = df.drop([''], axis = 1)
# df = df.drop([''], axis = 1)
# df = df.drop([''], axis = 1)
# df = df.drop([''], axis = 1)
# df = df.drop([''], axis = 1)


#df.to_excel("All.xls")

#935x42

#ProfileReport(df)
#report = ProfileReport(df)
#report.to_file('profile_report.html')

#%% PLOT ONE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(df['Organization Name'],df.Intersección).plot(kind='barh')
plt.title('Organization Name for 12 Company', fontsize = 22)
plt.xlabel('Organization Name', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('1_Organization Name')

#%% PLOT THREE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(df['Industries'],df.Intersección).plot(kind='barh')
plt.title('Industries for 12 Company', fontsize = 22)
plt.xlabel('Industries', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('2_Industries')

#%%
pd.crosstab(df['Headquarters Location'],df.Intersección).plot(kind='barh')
plt.title('Headquarters Location for 12 Company', fontsize = 22)
plt.xlabel('Headquarters Location', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('3_Headquarters Location')

#%%
pd.crosstab(df['Description'],df.Intersección).plot(kind='barh')
plt.title('Description for 12 Company', fontsize = 22)
plt.xlabel('Description', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('4_Description')

#%%
df.plot.scatter(x='CB Rank (Company)', y='Intersección')
plt.title('CB Rank (Company) for 12 Company', fontsize = 22)
plt.xlabel("CB Rank (Company)", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('5_CB Rank (Company)')

#%%
pd.crosstab(df['Estimated Revenue Range'],df.Intersección).plot(kind='bar')
plt.title('Estimated Revenue Range for 12 Company', fontsize = 22)
plt.xlabel('Estimated Revenue Range)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('6_CB Rank (Company)')

#%%
pd.crosstab(df['Operating Status'],df.Intersección).plot(kind='bar')
plt.title('Operating Status for 12 Company', fontsize = 22)
plt.xlabel('Operating Status)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('7_Operating Status')

#%%
pd.crosstab(df['Founded Date Precision'],df.Intersección).plot(kind='bar')
plt.title('Founded Date Precision for 12 Company', fontsize = 22)
plt.xlabel('Founded Date Precision)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('8_Founded Date Precision')

#%%
pd.crosstab(df['Company Type'],df.Intersección).plot(kind='bar')
plt.title('Company Type for 12 Company', fontsize = 22)
plt.xlabel('Company Type)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('9_Company Type')

#%%
df.plot.scatter(x='Number of Articles', y='Intersección')
plt.title('Number of Articles for 12 Company', fontsize = 22)
plt.xlabel("Number of Articles", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('10_Number of Articles')

#%%
pd.crosstab(df['Industry Groups'],df.Intersección).plot(kind='bar')
plt.title('Industry Groups for 12 Company', fontsize = 22)
plt.xlabel('Industry Groups', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('11_Industry Groups')

#%%
df.plot.scatter(x='Number of Founders', y='Intersección')
plt.title('Number of Founders for 12 Company', fontsize = 22)
plt.xlabel("Number of Founders", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('12_Number of Founders')

#%%
pd.crosstab(df['Number of Employees'],df.Intersección).plot(kind='barh')
plt.title('Number of Employees for 12 Company', fontsize = 22)
plt.xlabel('Number of Employees', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('13_Industry Groups')

#%%
df.plot.scatter(x='Number of Funding Rounds', y='Intersección')
plt.title('Number of Funding Rounds for 12 Company', fontsize = 22)
plt.xlabel("Number of Funding Rounds", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('14_Number of Funding Rounds')

#%%
pd.crosstab(df['Funding Status'],df.Intersección).plot(kind='barh')
plt.title('Funding Status for 12 Company', fontsize = 22)
plt.xlabel('Funding Status', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('15_Funding Status')

#%%
pd.crosstab(df['Last Funding Amount'],df.Intersección).plot(kind='barh')
plt.title('Last Funding Amount for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Amount', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('16_Last Funding Amount')

#%%
df.plot.scatter(x='Last Funding Amount', y='Intersección')
plt.title('Last Funding Amount for 12 Company', fontsize = 22)
plt.xlabel("Last Funding Amount", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('17_Last Funding Amount')

#%%
pd.crosstab(df['Last Funding Amount Currency'],df.Intersección).plot(kind='barh')
plt.title('Last Funding Amount Currency for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Amount Currency', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('18_Last Funding Amount Currency')

#%%
df.plot.scatter(x='Last Funding Amount Currency (in USD)', y='Intersección')
plt.title('Last Funding Amount Currency (in USD) for 12 Company', fontsize = 22)
plt.xlabel("Last Funding Amount Currency (in USD)", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('19_Last Funding Amount Currency (in USD)')

#%%
pd.crosstab(df['Last Funding Type'],df.Intersección).plot(kind='barh')
plt.title('Last Funding Type for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Type', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('20_Last Funding Type')

#%%
df.plot.scatter(x='Last Equity Funding Amount', y='Intersección')
plt.title('Last Equity Funding Amount for 12 Company', fontsize = 22)
plt.xlabel("Last Equity Funding Amount", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('21_Last Equity Funding Amount')

#%%
pd.crosstab(df['Last Equity Funding Amount Currency'],df.Intersección).plot(kind='barh')
plt.title('Last Equity Funding Amount Currency for 12 Company', fontsize = 22)
plt.xlabel('Last Equity Funding Amount Currency', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('22_Last Equity Funding Amount Currency')

#%% ####    
#           SE HARA EL DATA WAREHOUSE

# from sklearn import preprocessing # procesamiento de datos
# from sklearn.linear_model import LinearRegression #Regresión Lineal
# from sklearn.linear_model import LogisticRegression #Regresión Logistica
# from pandas_profiling import ProfileReport

sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)


# ProfileReport(df)
# report = ProfileReport(df)
# report.to_file('profile_report.html')


#Cantidad de empresas que no están en las 3 datas
count_int = len(df[df['Intersección']==0])
#Cantidad de empresas que están en las 3 datas
count_int2 = len(df[df['Intersección']==1])

pct_int = count_int / (count_int + count_int2)
pct_int2 = count_int2 / (count_int + count_int2)

print('Porcentaje de Empresas, No estan en las 3 data', pct_int*100)
print('Porcentaje de Empresas en las 3 data', pct_int2*100)

# sns.countplot(x='Intersección', data = df, palette = 'Blues')
# sns.countplot(x='Headquarters Location', data = df, palette = 'Blues')
# sns.countplot(x='Operating Status', data = df, palette = 'Blues')
# #sns.countplot(x='Industry Groups', data = df, palette = 'Blues')
# sns.countplot(x = 'Intersección', data = df, palette = 'GnBu_d')
# sns.countplot(x = 'Intersección', data = df, palette = 'cubehelix')

# sns.pairplot(df, hue = 'Intersección', diag_kind = 'hist')


df1 = df.copy()

df1 = df1.drop(['Organization Name','Industries','Headquarters Location','Description' ], axis = 1)
df1 = df1.drop(['Operating Status','Founded Date Precision','Founded Date','Founded Date Precision'], axis = 1)
df1 = df1.drop(['Company Type','Industry Groups','Founders','Funding Status', 'Last Funding Date','Last Funding Amount Currency','Last Funding Type'], axis = 1)
df1 = df1.drop(['Last Equity Funding Amount Currency','Last Equity Funding Type','Total Equity Funding Amount Currency'], axis = 1)
df1 = df1.drop(['Total Funding Amount Currency','Top 5 Investors','IPO Status','Co','Name','Industries_right','Description_right','Last Funding Amount Currency_right'], axis = 1)
df1 = df1.drop(['NOMBRE EMPRESA','CIIU','EMPRENDEDOR','DEPARTAMENTO','CIUDAD'], axis = 1)

df1 = df1.drop(['Estimated Revenue Range','Number of Employees'], axis = 1)

# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)
# df1 = df1.drop([''], axis = 1)


df1.to_excel("All2.xls")

df1.fillna(0)

df1.info()

df1["CB Rank (Company)"] = df1["CB Rank (Company)"].str.replace(r'\D', '')
df1["CB Rank (Company)"] = pd.to_numeric(df1["CB Rank (Company)"])

# Number of Articles tiene datos no numericos, se convierten a numeros
# df1[["Number of Articles"]] = df1[["Number of Articles"]].fillna('')
# df1["Number of Articles"] = df1["Number of Articles"].str.replace(r'\D', '')
# df1["Number of Articles"] = pd.to_numeric(df1["Number of Articles"])

# CBrank Organization tiene datos no numericos, se convierten a numeros
# df1[["CB Rank (Organization)"]] = df1[["CB Rank (Organization)"]].fillna('')
# df1["CB Rank (Organization)"] = df1["CB Rank (Organization)"].str.replace(r'\D', '')
# df1["CB Rank (Organization)"] = pd.to_numeric(df1["CB Rank (Organization)"])

x = df1.copy()
x = x.drop(['Intersección'], axis = 1)
#x = x.fillna(0)

numericColumnsNames = [
    "CB Rank (Company)",
    "Number of Articles",
    "Number of Founders",
    "Number of Funding Rounds",
    "Last Funding Amount",
    "Last Funding Amount Currency (in USD)",
    "Last Equity Funding Amount",
    "Last Equity Funding Amount Currency (in USD)",
    "Total Equity Funding Amount",
    "Total Equity Funding Amount Currency (in USD)",
    "Total Funding Amount",
    "Total Funding Amount Currency (in USD)",
    "Number of Lead Investors",
    "Number of Investors",
    #"Number of Events",
    "CB Rank (Organization)",
    "BuiltWith - Active Tech Count",
    "G2 Stack - Total Products Active"
]

for columnName in numericColumnsNames:
    df1[columnName] = df1[columnName].fillna(0)

df1 = df1.drop(['Total Funding Amount_right','Last Funding Amount_right'], axis = 1)

df1.to_excel("ETL_out.xls")


#%%
            ###Y POR ÚLTIMO SE REALIZA LA REGRESIÓN

os.chdir("/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Taller6")
cwd = os.getcwd()
df = pd.read_excel('ETL_out.xls')#, index_col = 0)
del df["Unnamed: 0"]
df["CB Rank (Company)"] = df["CB Rank (Company)"].str.replace(r'\D','')
df["CB Rank (Company)"] = pd.to_numeric(df["CB Rank (Company)"])

df1.drop(['CB Rank (Organization)'],1).hist()
plt.show()

sns.pairplot(df1.dropna(), hue='Intersección',size=4,vars=["Number of Articles", "CB Rank (Organization)","CB Rank (Organization)"],kind='reg')


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

Y = df.iloc[:, 17].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

# X = []
# for index in range(len(df)):
#     temp = []
#     for column in df.columns:
#         if column == "Intersección":
#             pass
#         content = df.iloc[index][columnName]
#         temp.append(np.array(content))
#     X.append(temp)
# X = np.array(X)

# linear_regressor = LinearRegression()  # Se crea el objeto para la clase
# linear_regressor.fit(X, Y)  # Se realiza la regresión log
# Y_pred = linear_regressor.predict(X)  # realiza la predicción

#linear_regressor.coef_

#%%

from sklearn.metrics import confusion_matrix

matriz = confusion_matrix

import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix,accuracy_score)

#log_reg = sm.Logit(Y, X).fit()#ajuste


def correct_word(word):
    new_word = word.split()[0]
    return new_word
#x = dt['Pais'].apply(correct_word)


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
# Columnas con mas del 80% de datos 


df0 = df1.drop(['Intersección' ], axis = 1)
z = cols_90per_zeros(df0, 0.75)

z

df0["CB Rank (Company)"] = df0["CB Rank (Company)"].str.replace(r'\D', '')
df0["CB Rank (Company)"] = pd.to_numeric(df0["CB Rank (Company)"])


log_reg = sm.Logit(Y, df0).fit()#ajuste

df0 = df0.drop(['Number of Lead Investors'], axis = 1)
df0 = df0.drop(['Last Funding Amount'], axis = 1)
df0 = df0.drop(['Last Equity Funding Amount'], axis = 1)
df0 = df0.drop(['Total Equity Funding Amount'], axis = 1)
#df0 = df0.drop(['Last Equity Funding Amount'], axis = 1)

#df0 = df0.drop(['Last Equity Funding Amount'], axis = 1)

log_reg = sm.Logit(Y, df0).fit()#ajuste


print(log_reg.summary())


#prueba

yhat = log_reg.predict(df0)#entrenamiento

prediction = list(map(round, yhat))




#print('Actal values', list(Y.values))
#print('Predictions :', prediction)


#matriz de precisión
cm = confusion_matrix(Y, prediction)
print ("Confusion Matrix : \n", cm)

#print('Test accuracy = ', accuracy_score(ytest, prediction))


            #%%     Análisis de componentes principales

#df0 es mi Xtrain

print('----------------------')
print('Media de cada variable')
print('----------------------')
df0.mean(axis=0)

print('----------------------')
print('Media de cada variable')
print('----------------------')
df0.mean(axis=0)


# Entrenamiento modelo PCA con escalado de los datos
# ==============================================================================
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(df0)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']
        
    
# Entrenamiento modelo PCA con escalado de los datos
# ==============================================================================
pca_pipe = make_pipeline(StandardScaler(), PCA())
pca_pipe.fit(df0)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

# Se convierte el array a dataframe para añadir nombres a los ejes.
pd.DataFrame(
    data    = modelo_pca.components_,
    columns = df0.columns,
    index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6','PC7',
               'PC8', 'PC9', 'PC10', 'PC11', 'PC12','PC13']
)

#%matplotlib qt

# Heatmap componentes
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
componentes = modelo_pca.components_
plt.imshow(componentes.T, cmap='viridis', aspect='auto')
plt.yticks(range(len(df0.columns)), df0.columns)
plt.xticks(range(len(df0.columns)), np.arange(modelo_pca.n_components_) + 1)
plt.grid(False)
plt.colorbar();



# Porcentaje de varianza explicada por cada componente
# ==============================================================================
print('----------------------------------------------------')
print('Porcentaje de varianza explicada por cada componente')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(df0.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada');


# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(df0.columns)) + 1,
    prop_varianza_acum,
    marker = 'o'
)

for x, y in zip(np.arange(len(df0.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada');





# Proyección de las observaciones de entrenamiento
# ==============================================================================
proyecciones = pca_pipe.transform(X = df0)
proyecciones = pd.DataFrame(
    proyecciones,
    columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7',
               'PC8', 'PC9', 'PC10', 'PC11', 'PC12','PC13'],
    index   = df0.index
)
proyecciones.head()


# dejando conjunto de training con 5 Variables
for i in range(12,4,-1):
    proyecciones = proyecciones.drop([proyecciones.columns[i]], axis=1)


#%% y nuevamente se realizaliza La Regresión Logística

log_reg = sm.Logit(Y, proyecciones).fit()#ajuste


print(log_reg.summary())


#prueba

yhat = log_reg.predict(proyecciones)#entrenamiento

prediction = list(map(round, yhat))




#print('Actal values', list(Y.values))
#print('Predictions :', prediction)


#matriz de precisión
cm = confusion_matrix(Y, prediction)
print ("Confusion Matrix : \n", cm)





