#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:53:59 2021

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

#%% importamos archivos de excel y csv y los convertimos en df


os.chdir('/Users/milleralexanderquirogacampos/OneDrive - Universidad Sergio Arboleda/4_TAMDML/Taller6/2/')
cwd=os.getcwd() #Asigna la variable cwd el directorio de trabajo

cb1 = pd.read_excel('CB_Insights.xlsx', sheet_name=0)#, index_col = 0)
cb2 = pd.read_excel('CB_Insights.xlsx', sheet_name=1)#, index_col = 0)
cb3 = pd.read_excel('CB_Insights.xlsx', sheet_name=2)#, index_col = 0)

cbs = [cb1, cb2, cb3]

print(cb1.columns)
print(cb2.columns)
print(cb3.columns)

for i in cbs:
    print(i.info())

for i in cbs:
    print(i.columns)
    print(i.shape)
    

for i in cbs:
    i["Country"].value_counts().plot(kind='bar',  title='Paises')
   
cb3["Country"].head(50).value_counts().plot(kind="bar")
cb2["Country"].head(50).value_counts().plot(kind="bar")
cb1["Country"].head(50).value_counts().plot(kind="bar")


#sns.pairplot(df0)

#           PAISES LATINOAMERICANOS
co = pd.read_csv('ColombiaCB-5March21.csv')#, index_col= 0)
cl = pd.read_csv('ChileCB-5March21.csv')#, index_col= 0)
br = pd.read_csv('BrazilCB-5March21.csv')#, index_col= 0)
ar = pd.read_csv('ArgentinaCB-5March21.csv')#, index_col= 0)
mx = pd.read_csv('MexicoCB-5March21.csv')#, index_col= 0)
ur = pd.read_csv('UruguayCB-5March21.csv')#, index_col= 0)

#           Paises Europa, Asia y NortAmerica
sp = pd.read_csv('SpainCB-5March21.csv')#, index_col= 0)
ge = pd.read_csv('GermanyCB-5March21.csv')#, index_col= 0)
sw = pd.read_csv('SwitzerlandCB-5March21.csv')#, index_col= 0)
Is = pd.read_csv('IsraelCB-5March21.csv')#, index_col= 0)
us = pd.read_csv('USACB-5March21.csv')#, index_col= 0)


#%%     LIMPIEZA DE LA DATA

co.drop(co[co['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)
co.drop(co[co['Headquarters Location']=='Albania, La Guajira, Colombia'].index, inplace =True)
co.drop(co[co['Headquarters Location']=='AndalucÃ­a, Valle del Cauca, Colombia'].index, inplace =True)
co.drop(co[co['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)
co = co[~co['Headquarters Location'].str.contains('Bavaria(?!$)')]
co = co[~co['Headquarters Location'].str.contains('Brasilia')]

# Se eliminan todas las Empresas de Canada, Cundinamarca, Colombia, excepto Qinaya y Partsium
co=co.drop(co[(co['Headquarters Location'] == 'CanadÃ¡, Cundinamarca, Colombia') 
            & ((co['Organization Name'] != 'Qinaya')  & (co['Organization Name'] != 'Partsium'))].index)

# mRisk, BattleBit, Cencosud Shopping Centers compañias de Chile
# Chile, Huila, Colombia  se Eliminan
co.drop(co[co['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)

# WindoTrader USA, aparece como Las Vegas, Sucre, Colombia
# se elimina

co.drop(co[co['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)

# Onyx, Elm, Photogramy, Ferrisland, BeyondROI sede en Los Angeles, Huila, Colombia
# se eliminan
co.drop(co[co['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)

#Peris Costumes, Cositas de EspaÃ±a, Esri, Pirsonal, Barrabes, Carousel Group, WORLD COMPLIANCE ASSOCIATION
# Clupik, Mobile Dreams ltd., Acqualia, LoAlkilo, Codekai, Vitriovr, El inmobiliario mes a mes
# Core Business Consulting, Renewable Energy Magazine, datosmacro, 1001talleres, GGBOX
# con sede en Madrid, Distrito Especial, Colombia

co.drop(co[co['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)

# Advanet (Japon) y Truland Service Corporation en USA. aparecen Maryland, Cundinamarca, Colombia

co.drop(co[co['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)

# POC Network Technologies (TransactRx), Alert Global Media.  sede Miami, Magdalena, Colombia

co.drop(co[co['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)

# Sicartsa sede en Mexico, Huila, Colombia

co.drop(co[co['Headquarters Location']=='MÃ©xico, Huila, Colombia'].index, inplace =True)

# 24marine, Merkadoo sede en Panama, Magdalena, Colombia

co.drop(co[co['Headquarters Location']=='PanamÃ¡, Magdalena, Colombia'].index, inplace =True)

# Agros, Downloadperu.com, Mesa 24/7, Dconfianza, Caja Los Andes, Snacks America Latina Peru S.R.L.
# Pandup, Apprende sede en Peru, Valle del Cauca, Colombia

co.drop(co[co['Headquarters Location']=='PerÃº, Valle del Cauca, Colombia'].index, inplace =True)

#%%
##Depuracion de nombres de ciudad incorrectos a correctos

# La compañia Savy tiene sede usaquen, se cambia a Bogota

co = co.replace({"UsaquÃ©n, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})

# M=co[co['Organization Name'] == 'Savy']

# PagomÃ­o  es compania de medellin y aparece con sede
# Antioquia, Antioquia, Colombia

co = co.replace({"Antioquia, Antioquia, Colombia":'MedellÃ­n, Antioquia, Colombia'})

# El Herald  El Heraldo sede Atlantico, Magdalena, Colombia  Es Barranquilla
# Atlantico, Magdalena, Colombia
# 

co = co.replace({"El Herald":'El Heraldo'})
co = co.replace({"AtlÃ¡ntico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})


# compañia Monolegal es de tunja y aparece Boyaca, Boyaca, Colombia

co = co.replace({"BoyacÃ¡, Boyaca, Colombia":'Tunja, Boyaca, Colombia'})

# Celotor es de cali aparece como Colombiano, Magdalena, Colombia

co = co.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})

# Qinaya, compañia colombiana
# https://www.wradio.com.co/noticias/tecnologia/qinaya-el-emprendimiento-que-convierte-cualquier-televisor-en-un-computador/20210301/nota/4113498.aspx
# https://www.youtube.com/watch?v=XBgbwUxkatc
# Canada, Cundinamarca, Colombia vs Bogota, Distrito Especial, Colombia
# Replace with condition
co.loc[(co['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'

# Partsium
# Partsium, bogota.  El Sitio pone a disposición de los Usuarios un espacio virtual que les permite
# comunicarse mediante el uso de Internet para encontrar una forma de vender o comprar productos y
# servicios. PARTSIUM no es el propietario de los artículos ofrecidos, no tiene posesión de ellos ni
# los ofrece en venta. Los precios de los productos y servicios están sujetos a cambios sin previo
# aviso.
# website rental to do business

co.loc[(co['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
co.loc[(co['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'

co = co.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
co = co.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
co = co.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
co = co.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
co = co.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})

co = co.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
co = co.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
co = co.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
co = co.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)

# co = co.replace({"CÃºcuta­": 'Cucuta'}) # no funciona

# co.loc[co['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"
co = co.replace({"Cundinamarca":'Bogota'})
co = co.replace({"Santander":'Bucaramanga'})
co = co.replace({"Santiago De Cali":'Cali'})


#F["CB Rank (Organization)"] = F["CB Rank (Organization)"].str.replace(r'\D', '').astype(int)
#F["CB Rank (Organization)"] = pd.to_numeric(F["CB Rank (Organization)"])

co[["Number of Articles"]] = co[["Number of Articles"]].fillna('') # Specific columns
co["Number of Articles"] = co["Number of Articles"].str.replace(r'\D', '')
co["Number of Articles"] = pd.to_numeric(co["Number of Articles"])

co[["CB Rank (Organization)"]] = co[["CB Rank (Organization)"]].fillna('') # Specific columns
co["CB Rank (Organization)"] = co["CB Rank (Organization)"].str.replace(r'\D', '')
co["CB Rank (Organization)"] = pd.to_numeric(co["CB Rank (Organization)"])

co[["Apptopia - Downloads Last 30 Days"]] = co[["Apptopia - Downloads Last 30 Days"]].fillna('') # Specific columns
co["Apptopia - Downloads Last 30 Days"] = co["Apptopia - Downloads Last 30 Days"].str.replace(r'\D', '')
co["Apptopia - Downloads Last 30 Days"] = pd.to_numeric(co["Apptopia - Downloads Last 30 Days"])

co.to_excel("ArchivoCol.xls")

#935x42

#ProfileReport(co)
#report = ProfileReport(co)
#report.to_file('profile_report.html')

# se renombra nombre columna CB Rank (Company) a CBRank
#df.rename(columns={"CB Rank (Company)": 'CBRank'}, inplace=True)


#%%

#ARMA UNA LISTA DE LOS 11 ARCHIVOS
datos = [co, cl, br, ar, mx, ur, sp, ge, sw, Is, us]

#VERIFICA SI HAY DATOS NULOS EN CADA VARIABLE DE CADA ARCHIVO
for i in datos:
    print(i.isnull().sum())

#DIMENSIÓN DE CADA VARIABLE DE LAS DATA
for i in datos:
   print(i.shape)
   
#IMPRIME LOS 11 PRIMEROS INVERSIONES DE CADA DATA FRAME
for i in datos:
    i["Last Funding Amount"].head(5).plot(kind="barh")

for k in datos:
    k["Last Funding Amount"].plot(kind="hist")
    plt.grid(True)
    plt.show




for j in datos:
    pd.crosstab(j["Organization Name"].head(10),j["Last Funding Amount"].head(10)).plot(kind='bar')

#%%         UNIMOS TODAS LAS BASES DE DATOS A UNA SOLA

df = pd.concat([co, cl, br, ar, mx, ur, sp, ge, sw, Is, us])
print(df.shape)

#%%         Realizamos intersecciones entre los dataframe de df y 3archivos

cbs

print(cb1.columns)
print(cb2.columns)
print(cb3.columns)
print(df.columns)

a = pd(cb1["Company"])
b = pd.cb2["Company"]
c = pd.cb3["Company"]

Int = set(df['Organization Name']).intersection(set(cb1['Company'])).intersection(set(cb2['Company'])).intersection(set(cb3['Company']))

Int = set(df['Organization Name']).intersection(set(a)).intersection(set(b)).intersection(set(c))

for i in Int:
    df.loc[df['Organization Name'] == i, ['Intersección']] = 1



