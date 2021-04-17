# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 11:32:29 2021

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


import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)
from countryinfo import CountryInfo

os.chdir("C:/Users/LENOVO/Documents/Clase6/")

#lectura de datos
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv_file='ColombiaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
excel_file1='Top100Startups- Colombia.xlsx'#, index_col = 0)
excel_file2='Empresas Unicorn - Contactos.xlsx' #asignacion dle nombre del archivo a una bandeja #

#lectura de datos
df_1 =pd.read_csv(csv_file)
df_2= pd.read_excel('Top100Startups- Colombia.xlsx')#, index_col = 0)
df_3= pd.read_excel('Empresas Unicorn - Contactos.xlsx')#, index_col = 0)



#realizo el marge de las dos datas ne un DF
df_marge2 = df_1.join(df_2, rsuffix='_right')
print(df_marge2)
df_marge2.shape

df_marge3 = df_marge2.join(df_3, rsuffix='_right')
print(df_marge3)
df_marge3.shape

#asignacion de Mayusaculas a la lista de las variables
df_marge3["Organization Name"]=df_marge3["Organization Name"].str.upper()
df_marge3["Name"]=df_marge3["Name"].str.upper()
df_marge3["Organization"]=df_marge3["Organization"].str.upper()
                            
#Realizo la inteccion en las Columnas para saber cual estan entan en la misma data 

Int = set(df_marge3['Organization Name']).intersection(set(df_marge3['Organization']))
len(Int)

Int2 = set(df_marge3['Organization Name']).intersection(set(df_marge3['Name']))
len(Int2)


intersec=set(df_marge3["Organization Name"]).intersection(set(df_marge3['Organization'])).intersection(set(df_marge3["Name"]))

len(intersec)
for elemento in intersec:
    print(elemento)
    

df['Intersección'] = 0
df.shape

#asigana 1 en las intesecciones y ceros a los no intersecciones.

for i in intersec:
   df_marge3.loc[df_marge2['Organization Name'] == i, ['Intersección']] = 1

#Guarda el documeto
df_marge3.to_csv('./dataall.csv',index=False)

#datos nulos
Lista = [df_marge3]

for i in Lista:
    print(i, i.isnull().sum(),)

##  Envia el resultado de nulos a un archivo de texto
with open("fileC.txt", "w") as output:
    output.write(str(Lista))


# #Perfilaje de Variables
# from pandas_profiling import ProfileReport
# ProfileReport(df_marge3)
# profile = ProfileReport(df_marge3, title="Profiling Report",explorative=True)

# profile = ProfileReport(df_marge3)
# profile.to_file('profile_report.html')



#%%      limpieza de variables


#Cracterizacion  variables 
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Albania, La Guajira, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='AndalucÃ­a, Valle del Cauca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)
df_marge3 = df_marge3[~df_marge3['Headquarters Location'].str.contains('Bavaria(?!$)')]
df_marge3= df_marge3[~df_marge3['Headquarters Location'].str.contains('Brasilia')]



# Se eliminan las siguintes Filas
df_marge3=df_marge3.drop(df_marge3[(df_marge3['Headquarters Location'] == 'CanadÃ¡, Cundinamarca, Colombia') 
           & ((df_marge3['Organization Name'] != 'Qinaya')  & (df['Organization Name'] != 'Partsium'))].index)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='MÃ©xico, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='PanamÃ¡, Magdalena, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='PerÃº, Valle del Cauca, Colombia'].index, inplace =True)

#%%Correcion de Nombres


# La compañia Savy tiene sede usaquen, se cambia a Bogota

df_marge3 = df_marge3.replace({"UsaquÃ©n, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df_marge3 = df_marge3.replace({"Antioquia, Antioquia, Colombia":'MedellÃ­n, Antioquia, Colombia'})
df_marge3 = df_marge3.replace({"El Herald":'El Heraldo'})
df_marge3= df_marge3.replace({"AtlÃ¡ntico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})
df_marge3 = df_marge3.replace({"BoyacÃ¡, Boyaca, Colombia":'Tunja, Boyaca, Colombia'})
df_marge3 = df_marge3.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})
df_marge3 = df_marge3.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df_marge3 = df_marge3.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df_marge3 = df_marge3.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df_marge3 = df_marge3.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df_marge3 = df_marge3.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})
df_marge3 = df_marge3.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
df_marge3 = df_marge3.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
df_marge3 = df_marge3.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
df_marge3 = df_marge3.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)
df_marge3 = df_marge3.replace({"Cundinamarca":'Bogota'})
df_marge3 = df_marge3.replace({"Santander":'Bucaramanga'})
df_marge3 = df_marge3.replace({"Santiago De Cali":'Cali'})

# Replace with condition
df_marge3.loc[(df_marge3['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
df_marge3.loc[(df_marge3['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
df_marge3.loc[(df_marge3['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'
df_marge3.loc[df_marge3['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"

#Reemplazo columnas especificas variable Numerica

df_marge3[["Number of Articles"]] = df_marge3[["Number of Articles"]].fillna('') # Specific columns
df_marge3["Number of Articles"] = df_marge3["Number of Articles"].str.replace(r'\D', '')
df_marge3["Number of Articles"] = pd.to_numeric(df_marge3["Number of Articles"])

df_marge3[["CB Rank (Organization)"]] = df_marge3[["CB Rank (Organization)"]].fillna('') # Specific columns
df_marge3["CB Rank (Organization)"] = df_marge3["CB Rank (Organization)"].str.replace(r'\D', '')
df_marge3["CB Rank (Organization)"] = pd.to_numeric(df_marge3["CB Rank (Organization)"])

df_marge3[["Apptopia - Downloads Last 30 Days"]] = df_marge3[["Apptopia - Downloads Last 30 Days"]].fillna('') # Specific columns
df_marge3["Apptopia - Downloads Last 30 Days"] = df_marge3["Apptopia - Downloads Last 30 Days"].str.replace(r'\D', '')
df_marge3["Apptopia - Downloads Last 30 Days"] = pd.to_numeric(df_marge3["Apptopia - Downloads Last 30 Days"])



# #Es una funcion para eliminar entradas nulas de acuerdo al porcentaje    
# def cols_90per_nulls(df_marge3):
#     count = 0
#     cols_to_drop = {}
#     for col in df_marge3.columns: 
#         per_nulls = df_marge3[col].isna().sum()/len(df_marge3[col])
#         if per_nulls >= 0.80:
#             cols_to_drop[col] = per_nulls 
#             # print(col, per_nulls)
#             count+=1
#         else:
#             None
    
#     print('Number of cols with >90% nulls:', count)
#     return cols_to_drop



# nulls_Colums = cols_90per_nulls(df_marge3)


# dataFrame = df_marge3.drop(columns = nulls_Colums)#Elimina las columnas con un 75% nan
# print(dataFrame.shape)



#Eliminacion de Columnasde variables que no aportan con base al perfilamiento

df_marge3 = df_marge3.drop(['Organization Name URL'], axis=1)
df_marge3 = df_marge3.drop(['Contact Email'], axis=1) 
df_marge3 = df_marge3.drop(['Phone Number'], axis=1)  
df_marge3 = df_marge3.drop(['Full Description'], axis=1) 
df_marge3 = df_marge3.drop(['Transaction Name URL'], axis=1) 
df_marge3 = df_marge3.drop(['Acquired by URL'], axis=1) 
df_marge3 = df_marge3.drop(['Exit Date'], axis = 1)
df_marge3 = df_marge3.drop(['Exit Date Precision'], axis = 1)
df_marge3 = df_marge3.drop(['Closed Date'], axis = 1)
df_marge3 = df_marge3.drop(['Website'], axis = 1)
df_marge3 = df_marge3.drop(['Twitter'], axis = 1)
df_marge3 = df_marge3.drop(['Facebook'], axis = 1)
df_marge3 = df_marge3.drop(['LinkedIn'], axis = 1)
df_marge3 = df_marge3.drop(['Hub Tags'], axis = 1)
df_marge3 = df_marge3.drop(['Investor Type'], axis = 1)
df_marge3 = df_marge3.drop(['Investment Stage'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Portfolio Organizations'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Investments'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Lead Investments'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Exits'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Exits (IPO)'], axis = 1)
df_marge3 = df_marge3.drop(['Accelerator Program Type'], axis = 1)
df_marge3 = df_marge3.drop(['Accelerator Duration (in weeks)'], axis = 1)
df_marge3 = df_marge3.drop(['School Type'], axis = 1)
df_marge3 = df_marge3.drop(['School Program'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Enrollments'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Founders (Alumni)'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Acquisitions'], axis = 1)
df_marge3 = df_marge3.drop(['Acquisition Status'], axis = 1)
df_marge3 = df_marge3.drop(['Transaction Name'], axis = 1)
df_marge3 = df_marge3.drop(['Acquired by'], axis = 1)
df_marge3 = df_marge3.drop(['Announced Date'], axis = 1)
df_marge3 = df_marge3.drop(['Announced Date Precision'], axis = 1)
df_marge3 = df_marge3.drop(['Price'], axis = 1)
df_marge3 = df_marge3.drop(['Price Currency'], axis = 1)
df_marge3 = df_marge3.drop(['Price Currency (in USD)'], axis = 1)
df_marge3 = df_marge3.drop(['Acquisition Type'], axis = 1)
df_marge3 = df_marge3.drop(['Acquisition Terms'], axis = 1)
df_marge3 = df_marge3.drop(['IPO Date'], axis = 1)
df_marge3 = df_marge3.drop(['Delisted Date'], axis = 1)
df_marge3 = df_marge3.drop(['Delisted Date Precision'], axis = 1)
df_marge3 = df_marge3.drop(['Money Raised at IPO'], axis = 1)
df_marge3 = df_marge3.drop(['Money Raised at IPO Currency'], axis = 1)
df_marge3 = df_marge3.drop(['Money Raised at IPO Currency (in USD)'], axis = 1)
df_marge3 = df_marge3.drop(['Valuation at IPO'], axis = 1)
df_marge3 = df_marge3.drop(['Valuation at IPO Currency'], axis = 1)
df_marge3 = df_marge3.drop(['Valuation at IPO Currency (in USD)'], axis = 1)
df_marge3 = df_marge3.drop(['Stock Symbol'], axis = 1)
df_marge3 = df_marge3.drop(['Stock Symbol URL'], axis = 1)
df_marge3 = df_marge3.drop(['Stock Exchange'], axis = 1)
df_marge3 = df_marge3.drop(['Last Leadership Hiring Date'], axis = 1)
df_marge3 = df_marge3.drop(['Number of Events'], axis = 1)
df_marge3 = df_marge3.drop(['Apptopia - Number of Apps'], axis = 1)
df_marge3 = df_marge3.drop(['Apptopia - Downloads Last 30 Days'], axis = 1)
df_marge3 = df_marge3.drop(['IPqwery - Patents Granted'], axis = 1)
df_marge3 = df_marge3.drop(['IPqwery - Trademarks Registered'], axis = 1)
df_marge3 = df_marge3.drop(['IPqwery - Most Popular Patent Class'], axis = 1)
df_marge3 = df_marge3.drop(['IPqwery - Most Popular Trademark Class'], axis = 1)
df_marge3 = df_marge3.drop(['Aberdeen - IT Spend'], axis = 1)
df_marge3 = df_marge3.drop(['Aberdeen - IT Spend Currency'], axis = 1)
df_marge3 = df_marge3.drop(['Aberdeen - IT Spend Currency (in USD)'], axis = 1)
df_marge3 = df_marge3.drop(['School Method'], axis = 1)
df_marge3 = df_marge3.drop(['NIT'], axis = 1)
df_marge3 = df_marge3.drop(['CORREO ELECTRONICO'], axis = 1)
df_marge3 = df_marge3.drop(['TELÉFONO '], axis = 1)
df_marge3 = df_marge3.drop(['Unnamed: 16'], axis = 1)
df_marge3 = df_marge3.drop(['Closed Date Precision'], axis = 1)
df_marge3 = df_marge3.drop(['Headquarters Regions'], axis = 1)
df_marge3 = df_marge3.drop(['Founded Date_right'], axis = 1)
df_marge3 = df_marge3.drop(['Last Funding Date_right'], axis = 1)
df_marge3 = df_marge3.drop(['Last Funding Date_right'], axis = 1)
df_marge3 = df_marge3.drop(['Ciudad'], axis = 1)

dataFrame=df_marge3

  
dfx=dataFrame["Headquarters Location"].str.split(",", n = 2, expand = True) 
####  adicionando las nuevas columnas a df original
dataFrame["Ciudad"]= dfx[0] 
dataFrame["Departamento"]= dfx[1] 
dataFrame["Pais"]= dfx[2]


#Funcion que corrige espacios 
def correct_word(word):
    
    new_word = word.split()[0]
    return new_word

#Aplicandon la funcion para la columna Departamento
dataFrame['Departamento'] = dataFrame['Departamento'].apply(correct_word)

#Aplicandon la funcion para la columna Pais
dataFrame['Pais'] = dataFrame['Pais'].apply(correct_word)

print(dataFrame.shape)

#Guarda el documeto
dataFrame.to_excel('./Finaldataframe.xls',index=False)


#Perfilaje de Variables finales
from pandas_profiling import ProfileReport
ProfileReport(dataFrame)
profile = ProfileReport(dataFrame, title="Profiling Report",explorative=True)

profile = ProfileReport(dataFrame)
profile.to_file('profile_report.html')



#Correlacion de variables numericas
dataFrame.corr(method='pearson')
dataFrame.corr()
#Mapa de calor para las variables numericas y encontrar el grado de importancia de cada variable
sns.heatmap(dataFrame.corr(method='spearman'));
#Veremos mejor las distribuciones de cada variable apartir del comportamiento de cada pais
sns.pairplot(dataFrame)

#Regresion logistica

Xtrain = dataFrame[['Last Funding Amount Currency(in USD)', 'Last Equity Funding Amount Currency(In USD)']]
ytrain = dataFrame[['Intersección']]

log_reg = sm.Logit(ytrain, Xtrain).fit()#ajuste

print(log_reg.summary())

#prueba
Xtest = data[['Last Funding Amount Currency(in USD)', 'Last Equity Funding Amount Currency(In USD)']] 
ytest = data['Intersección']

yhat = log_reg.predict(Xtest)#entrenamiento

prediction = list(map(round, yhat))

print('Acutal values', list(ytest.values))
print('Predictions :', prediction)

#matriz de precisión
cm = confusion_matrix(ytest, prediction)
print ("Confusion Matrix : \n", cm)

print('Test accuracy = ', accuracy_score(ytest, prediction))


#%% PLOT ONE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(df['Organization Name'],dataFrame.Intersección).plot(kind='barh')
plt.title('Organization Name for 12 Company', fontsize = 22)
plt.xlabel('Organization Name', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Organization Name')

#%% PLOT THREE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(df['Industries'],dataFrame.Intersección).plot(kind='barh')
plt.title('Industries for 12 Company', fontsize = 22)
plt.xlabel('Industries', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Industries')

#%%
pd.crosstab(df['Headquarters Location'],dataFrame.Intersección).plot(kind='barh')
plt.title('Headquarters Location for 12 Company', fontsize = 22)
plt.xlabel('Headquarters Location', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Headquarters Location')

#%%
pd.crosstab(df['Description'],dataFrame.Intersección).plot(kind='barh')
plt.title('Description for 12 Company', fontsize = 22)
plt.xlabel('Description', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Description')

#%%
dataFrame.plot.scatter(x='CB Rank (Company)', y='Intersección')
plt.title('CB Rank (Company) for 12 Company', fontsize = 22)
plt.xlabel("CB Rank (Company)", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('CB Rank (Company)')

#%%
pd.crosstab(dataFrame['Estimated Revenue Range'],dataFrame.Intersección).plot(kind='bar')
plt.title('Estimated Revenue Range for 12 Company', fontsize = 22)
plt.xlabel('Estimated Revenue Range)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('CB Rank (Company)')

#%%
pd.crosstab(dataFrame['Operating Status'],dataFrame.Intersección).plot(kind='bar')
plt.title('Operating Status for 12 Company', fontsize = 22)
plt.xlabel('Operating Status)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Operating Status')

#%%
pd.crosstab(dataFrame['Founded Date Precision'],dataFrame.Intersección).plot(kind='bar')
plt.title('Founded Date Precision for 12 Company', fontsize = 22)
plt.xlabel('Founded Date Precision)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Founded Date Precision')

#%%
pd.crosstab(dataFrame['Company Type'],dataFrame.Intersección).plot(kind='bar')
plt.title('Company Type for 12 Company', fontsize = 22)
plt.xlabel('Company Type)', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Company Type')

#%%
dataFrame.plot.scatter(x='Number of Articles', y='Intersección')
plt.title('Number of Articles for 12 Company', fontsize = 22)
plt.xlabel("Number of Articles", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Number of Articles')

#%%
pd.crosstab(dataFrame['Industry Groups'],dataFrame.Intersección).plot(kind='bar')
plt.title('Industry Groups for 12 Company', fontsize = 22)
plt.xlabel('Industry Groups', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Industry Groups')

#%%
pd.crosstab(dataFrame['Number of Founders'],dataFrame.Intersección).plot(kind='bar')
plt.title('Number of Founders Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Founders', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Number of Founders')

#%%
pd.crosstab(dataFrame['Number of Employees'],dataFrame.Intersección).plot(kind='barh')
plt.title('Number of Employees for 12 Company', fontsize = 22)
plt.xlabel('Number of Employees', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Industry Groups')

#%%
pd.crosstab(dataFrame['Number of Funding Rounds'],dataFrame.Intersección).plot(kind='bar')
plt.title('Number of Funding Rounds Frequency for Successful Start-Ups', fontsize = 22)
plt.xlabel('Number of Funding Rounds', fontsize = 20)
plt.ylabel('Frequency of Successful Start-Ups', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Number of Funding Rounds')

#%%
pd.crosstab(dataFrame['Funding Status'],dataFrame.Intersección).plot(kind='barh')
plt.title('Funding Status for 12 Company', fontsize = 22)
plt.xlabel('Funding Status', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Funding Status')

#%%
pd.crosstab(dataFrame['Last Funding Amount'],dataFrame.Intersección).plot(kind='barh')
plt.title('Last Funding Amount for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Amount', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Funding Amount')

#%%
dataFrame.plot.scatter(x='Last Funding Amount', y='Intersección')
plt.title('Last Funding Amount for 12 Company', fontsize = 22)
plt.xlabel("Last Funding Amount", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Funding Amount')

#%%
pd.crosstab(dataFrame['Last Funding Amount Currency'],dataFrame.Intersección).plot(kind='barh')
plt.title('Last Funding Amount Currency for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Amount Currency', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Funding Amount Currency')

#%%
dataFrame.plot.scatter(x='Last Funding Amount Currency (in USD)', y='Intersección')
plt.title('Last Funding Amount Currency (in USD) for 12 Company', fontsize = 22)
plt.xlabel("Last Funding Amount Currency (in USD)", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Funding Amount Currency (in USD)')

#%%
pd.crosstab(dataFrame['Last Funding Type'],df.Intersección).plot(kind='barh')
plt.title('Last Funding Type for 12 Company', fontsize = 22)
plt.xlabel('Last Funding Type', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Funding Type')

#%%
dataFrame.plot.scatter(x='Last Equity Funding Amount', y='Intersección')
plt.title('Last Equity Funding Amount for 12 Company', fontsize = 22)
plt.xlabel("Last Equity Funding Amount", fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Equity Funding Amount')

#%%
pd.crosstab(dataFrame['Last Equity Funding Amount Currency'],dataFrame.Intersección).plot(kind='barh')
plt.title('Last Equity Funding Amount Currency for 12 Company', fontsize = 22)
plt.xlabel('Last Equity Funding Amount Currency', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Last Equity Funding Amount Currency')

#%%

