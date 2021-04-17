# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:33:00 2021

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
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

import os # Operative System #
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#
from sklearn.metrics import (confusion_matrix, accuracy_score)
from countryinfo import CountryInfo

os.chdir("C:/Users/LENOVO/Documents/Clase6/")

#Region
cwd = os.getcwd() #asigna la variable cwd a la directorio de trabajo#
csv1_file='ColombiaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv2_file='ChileCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv3_file='BrazilCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv4_file='ArgentinaCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv5_file='MexicoCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv6_file='UruguayCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #

df1=pd.read_csv(csv1_file)
df2=pd.read_csv(csv2_file)
df3=pd.read_csv(csv3_file)
df4=pd.read_csv(csv4_file)
df5=pd.read_csv(csv5_file)
df6=pd.read_csv(csv6_file)

#Pais no Region

csv7_file='SpainCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv8_file='GermanyCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv9_file='SwitzerlandCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv10_file='IsraelCB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #
csv11_file='USACB-5March21.csv' #asignacion dle nombre del archivo a una bandeja #

df7=pd.read_csv(csv7_file)
df8=pd.read_csv(csv8_file)
df9=pd.read_csv(csv9_file)
df10=pd.read_csv(csv10_file)
df11=pd.read_csv(csv11_file)


#%%      limpieza de variables


#Cracterizacion  variables 
df1.drop(df1[df1['Headquarters Location']=='Asturias, Cundinamarca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Albania, La Guajira, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='AndalucÃ­a, Valle del Cauca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Barrios Unidos, Distrito Especial, Colombia'].index, inplace =True)
df1 = df1[~df1['Headquarters Location'].str.contains('Bavaria(?!$)')]
df1= df1[~df1['Headquarters Location'].str.contains('Brasilia')]
df1.shape
# Se eliminan las siguintes Filas
df1=df1.drop(df1[(df1['Headquarters Location'] == 'CanadÃ¡, Cundinamarca, Colombia') 
           & ((df1['Organization Name'] != 'Qinaya')  & (df1['Organization Name'] != 'Partsium'))].index)
df1.drop(df1[df1['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='MÃ©xico, Huila, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='PanamÃ¡, Magdalena, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='PerÃº, Valle del Cauca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Perú, Valle del Cauca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Canadá, Cundinamarca, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Panamá, Magdalena, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='México, Huila, Colombia'].index, inplace =True)
df1.drop(df1[df1['Headquarters Location']=='Andalucía, Valle del Cauca, Colombia'].index, inplace =True)
df1.shape



#%%Correcion de Nombres

# La compañia Savy tiene sede usaquen, se cambia a Bogota

df1 = df1.replace({"UsaquÃ©n, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df1 = df1.replace({"Antioquia, Antioquia, Colombia":'MedellÃ­n, Antioquia, Colombia'})
df1 = df1.replace({"El Herald":'El Heraldo'})
df1= df1.replace({"AtlÃ¡ntico, Magdalena, Colombia":'Barranquilla, Atlantico, Colombia'})
df1 = df1.replace({"BoyacÃ¡, Boyaca, Colombia":'Tunja, Boyaca, Colombia'})
df1 = df1.replace({"Colombiano, Magdalena, Colombia":'Cali, Valle del Cauca, Colombia'})
df1 = df1.replace({"Bucaramanga, Cundinamarca, Colombia":'Bucaramanga, Santander, Colombia'})
df1 = df1.replace({"CÃºcuta, Antioquia, Colombia":'Cucuta, Norte de Santander, Colombia'})
df1 = df1.replace({"Cundinamarca, Distrito Especial, Colombia":'Bogota, Distrito Especial, Colombia'})
df1 = df1.replace({"Santander, Bolivar, Colombia":'Bucaramanga, Santander, Colombia'})
df1 = df1.replace({"Santiago De Cali, Valle del Cauca, Colombia":'Cali, Valle del Cauca, Colombia'})
df1 = df1.replace({"Ã¡": 'a', "Ã­": 'i', "Ã³":'o', "Ã©": 'e', "Ã¼Ã­": 'ui', "Ãº­": 'u'},regex=True)
df1 = df1.replace({"Ã‰": 'E', "BambbÃº": 'Bambbu', "Ã±": 'ñ', "PÃºblicas": 'Publicas'},regex=True)
df1 = df1.replace({"FÃºtbol": 'Futbol',"Ãºltimo":'Ultimo',"PÃºbliKo":'Publiko'},regex=True)
df1 = df1.replace({"TakÃºm": 'Takum',"ItagÃ¼i":'Itagui'},regex=True)
df1 = df1.replace({"Cundinamarca":'Bogota'})
df1 = df1.replace({"Santander":'Bucaramanga'})
df1 = df1.replace({"Santiago De Cali":'Cali'})

df1.shape

# Replace with condition
# df1.loc[(df1['Organization Name'] == 'Qinaya'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
# df1.loc[(df1['Organization Name'] == 'Partsium'),'Headquarters Location']='Bogota, Distrito Especial, Colombia'
# df1.loc[(df1['Organization Name'] == 'Partsium'),'Industries']='Website rental, Doing business'
# df1.loc[df1['Organization Name'] == 'Sanaty IPS', 'Ciudad'] = "Cucuta"
# df1.shape

df1['Number of Events'].isna().sum()/len(df1['Number of Events']) # porcentaje de datos nulos en la columna

data=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]
for i in data:
    print(i.isnull().sum())


data=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]
for i in data:
   print(i.shape)

data=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]
for i in data:
    i["Last Funding Amount"].head(10).plot(kind="barh")
    

data=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]
for i in data:
  print(i.columns)
  
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



df_all=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
print(df_all.shape)



#%%      limpieza de variables

#Reemplazo columnas especificas variable Numerica

df_all[["Number of Articles"]] = df_all[["Number of Articles"]].fillna('') # Specific columns
df_all["Number of Articles"] = df_all["Number of Articles"].str.replace(r'\D', '')
df_all["Number of Articles"] = pd.to_numeric(df_all["Number of Articles"])

df_all[["CB Rank (Organization)"]] = df_all[["CB Rank (Organization)"]].fillna('') # Specific columns
df_all["CB Rank (Organization)"] = df_all["CB Rank (Organization)"].str.replace(r'\D', '')
df_all["CB Rank (Organization)"] = pd.to_numeric(df_all["CB Rank (Organization)"])

df_all[["Apptopia - Downloads Last 30 Days"]] = df_all[["Apptopia - Downloads Last 30 Days"]].fillna('') # Specific columns
df_all["Apptopia - Downloads Last 30 Days"] = df_all["Apptopia - Downloads Last 30 Days"].str.replace(r'\D', '')
df_all["Apptopia - Downloads Last 30 Days"] = pd.to_numeric(df_all["Apptopia - Downloads Last 30 Days"])

df_all["CB Rank (Company)"] = df_all["CB Rank (Company)"].str.replace(r'\D', '')
df_all["CB Rank (Company)"] = pd.to_numeric(df_all["CB Rank (Company)"])

# #Es una funcion para eliminar entradas nulas de acuerdo al porcentaje    
# def cols_90per_nulls(df_all):
#     count = 0
#     cols_to_drop = {}
#     for col in df_all.columns: 
#         per_nulls = df_all[col].isna().sum()/len(df_all[col])
#         if per_nulls >= 0.8:
#             cols_to_drop[col] = per_nulls 
#             # print(col, per_nulls)
#             count+=1
#         else:
#             None
    
#     print('Number of cols with >80% nulls:', count)
#     return cols_to_drop


# nulls_Colums = cols_90per_nulls(df_all)

# dataFrame=df_all.drop(columns = nulls_Colums)#Elimina las columnas con un 75% nan

# print(dataFrame.shape)


#Eliminacion de Columnasde variables que no aportan con base al perfilamiento

df_all = df_all.drop(['Organization Name URL'], axis=1)
df_all = df_all.drop(['Website'], axis=1)
df_all = df_all.drop(['Twitter'], axis=1)
df_all = df_all.drop(['Facebook'], axis=1)
df_all = df_all.drop(['LinkedIn'], axis=1)
df_all = df_all.drop(['Contact Email'], axis=1)
df_all = df_all.drop(['Phone Number'], axis=1)
df_all = df_all.drop(['Full Description'], axis=1)
df_all = df_all.drop(['Transaction Name URL'], axis=1)
df_all = df_all.drop(['Acquired by URL'], axis=1) 
df_all = df_all.drop(['Stock Symbol URL'], axis=1)

df_all.shape

dfx=df_all["Headquarters Location"].str.split(",", n = 2, expand = True) 
####  adicionando las nuevas columnas a df original
df_all["Ciudad"]= dfx[0] 
df_all["Departamento"]= dfx[1] 
df_all["Pais"]= dfx[2]

#Funcion que corrige espacios 
def correct_word(word):
    
    new_word = word.split()[0]
    return new_word

#Aplicandon la funcion para la columna Departamento
df_all['Departamento'] = df_all['Departamento'].apply(correct_word)

#Aplicandon la funcion para la columna Pais
df_all['Pais'] = df_all['Pais'].apply(correct_word)

print(df_all.shape)


#Crear Colummna de Años
years = []
for index in range(len(df_all)):
    fundingDate = df_all.iloc[index]["Last Funding Date"]
    year = str(fundingDate)
    if year == 'nan' or year == 'year' or year == 'day':
        year = "0"
    years.append(int(year.split("-")[0]))
df_all["year"] = years

print(df_all.shape)

df_all.describe() #estadistica descriptiva de la variables

df_all.loc[(df_all.Pais == 'United'),'Pais']='Usa'


# #Perfilaje de Variables finales
from pandas_profiling import ProfileReport
ProfileReport(df_all)
profile = ProfileReport(df_all, title="Profiling Report",explorative=True)

profile = ProfileReport(df_all)
profile.to_file('profile_report.html')

# # #Correlacion de variables numericas
df_all.corr(method='pearson')
df_all.corr()

# # #Mapa de calor para las variables numericas y encontrar el grado de importancia de cada variable

sns.heatmap(df_all.corr(),annot=True,cmap="RdYlGn") 

# # # sns.heatmap(dataFrame.corr(method='spearman'));
# # #Veremos mejor las distribuciones de cada variable apartir del comportamiento de cada pais
sns.pairplot(df_all)

# dataFrame = df_all.fillna(0)


#Guarda el documeto
df_all.to_excel('./DF.xls',index=False)

#%%% 1 Cuánto capital se ha invertido en LaTAM durante el último año. Desagregue gráficamente por país. 

dfx1=df_all[df_all.columns[:]][(df_all["Headquarters Regions"] == 'Latin America') & (df_all["year"] == 2021)]

dfx1['Last Funding Amount Currency (in USD)']=dfx1['Last Funding Amount Currency (in USD)']/1e6

pd.crosstab(dfx1.Pais,dfx1["year"],aggfunc="sum",values=dfx1['Last Funding Amount Currency (in USD)']).plot(kind="bar")
plt.title('Pais Vs Last Funding Amount Currency (in USD)-2021', fontsize = 22)
plt.xlabel('Pais', fontsize = 20)
plt.ylabel('Last Funding Amount Currency in MM', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.show()
plt.savefig('Punto 1')

#%%% 2. Haga una comparación entre Colombia con cada uno de los otros países. Analice

dfx1=df_all[df_all.columns[:]][(df_all["Headquarters Regions"] == 'Latin America') & (df_all["year"] >= 2016)]

dfx1['Last Funding Amount Currency (in USD)']=dfx1['Last Funding Amount Currency (in USD)']/1e6

pd.crosstab(dfx1.Pais,dfx1["year"],aggfunc="sum",values=dfx1['Last Funding Amount Currency (in USD)']).plot(kind="bar")
plt.title('Pais Vs Last Funding Amount Currency (in USD)-(last 6 Year)', fontsize = 22)
plt.xlabel('Pais', fontsize = 20)
plt.ylabel('Last Funding Amount Currency in MM', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.show()
plt.savefig('Punto 1')


#Organizacion porRanking
df_all_sorted_by_CB_Rank = df_all.sort_values(["CB Rank (Company)"], ascending=True)
# ,aggfunc="sum",values = df_all_sorted_by_CB_Rank['Last Funding Amount Currency (in USD)']

# GRAFICA POR RANK

# pd.crosstab(df_all_sorted_by_CB_Rank .Pais,df_all_sorted_by_CB_Rank["CB Rank (Company)"]).plot(kind="bar")
# plt.title('Pais Vs CB Rank (Company', fontsize = 22)
# plt.xlabel('Pais', fontsize = 20)
# plt.ylabel('CB Rank (Company', fontsize = 20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=18)
# plt.show()
# plt.savefig('Punto 2a')

#Organizacion Revenue POR PAIS Y LFA
df_all_sorted_by_Revenue = df_all.sort_values(["Estimated Revenue Range"], ascending=True)

pd.crosstab(df_all_sorted_by_Revenue.Pais,df_all_sorted_by_Revenue["Estimated Revenue Range"],aggfunc="sum",\
            values=df_all_sorted_by_Revenue['Last Funding Amount Currency (in USD)']).plot(kind="bar")
plt.title('Pais Vs Estimated Revenue Range', fontsize = 22)
plt.xlabel('Pais', fontsize = 20)
plt.ylabel('Estimated Revenue Range', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.show()
plt.savefig('Punto 2b')

#### Grafica de Rangos de ingreso por Pais Normalizados

pd.crosstab(df_all_sorted_by_Revenue.Pais,df_all_sorted_by_Revenue['Estimated Revenue Range'],normalize = "index").plot(kind='bar')
plt.title('Normalized Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Percentage of Revenue Range', fontsize=20)

#################################################################
#### Grafica de Rangos de ingreso por Pais

pd.crosstab(df_all_sorted_by_Revenue.Pais,df_all_sorted_by_Revenue['Estimated Revenue Range']).plot(kind='bar')
plt.title('Revenue Range per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Revenue Range', fontsize=20)

#################################################################

################################################################
# LFA Normalizado per capita por pais y rango de ingreso de empresas

w=pd.crosstab(df_all_sorted_by_Revenue.Pais,df_all_sorted_by_Revenue['Estimated Revenue Range'],aggfunc="sum",values = df_all_sorted_by_Revenue['Last Funding Amount Currency (in USD)'])

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
################################################################

 #Organizacion Numero d eempleados
df_all_sorted_by_Employees = df_all.sort_values(["Number of Employees"], ascending=True)
    
pd.crosstab(df_all_sorted_by_Employees.Pais,df_all_sorted_by_Employees["Number of Employees"]).plot(kind='bar')
plt.title('Number of Employees per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Revenue Range', fontsize=20)
    
    
    #Organizacion Numero de Articulos
df_all["Number of Articles"] = pd.to_numeric(df_all["Number of Articles"])
    
df_all_sorted_by_Articles = df_all.sort_values(["Number of Articles"], ascending=True)
    
# Number of Articles
pd.crosstab(df_all_sorted_by_Articles.Pais,df_all_sorted_by_Articles['year'],aggfunc="sum",values=df_all_sorted_by_Articles['Number of Articles']).plot(kind='bar')
plt.title('Number of Articles - Total  per Country ', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Number of Articles - Sum', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
    
 #Organizacion Numero de Invesment 
df_all_sorted_by_Number_Investment = df_all.sort_values(["Number of Investments"], ascending=True)
    
    


#%%% 3. Cuáles son los fondos que más invierten en Colombia? Haga un análisis descriptivo de  cada uno de ellos.

#Organizacion Inversor Type
df_all_sorted_by_Investor_Type = df_all.sort_values(["Investor Type"], ascending=True)
df_all_sorted_by_Investor_Type =df_all_sorted_by_Investor_Type[df_all_sorted_by_Investor_Type.columns[:]][(df_all_sorted_by_Investor_Type["Pais"] == 'Colombia')]

pd.crosstab(df_all_sorted_by_Investor_Type.Pais,df_all_sorted_by_Investor_Type["Investor Type"]).plot(kind='bar')
plt.title('Colombia Vs Investor Type', fontsize=22)
plt.xlabel('Colombia', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)

# 

#%%% 3.1 ¿Cuál es la tesis de inversión de cada uno de estos fondos?

#Organizacion Investor Type
df_all_sorted_by_Investor_Type = df_all.sort_values(["Investor Type"], ascending=True)
df_all_sorted_by_Investor_Type = df_all_sorted_by_Investor_Type[df_all_sorted_by_Investor_Type.columns[:]][(df_all_sorted_by_Investor_Type["Pais"] == 'Colombia')]

pd.crosstab(df_all_sorted_by_Investor_Type.Pais,df_all_sorted_by_Investor_Type ["Investor Type"],aggfunc="sum",values=df_all_sorted_by_Investor_Type ['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Colombia Vs Investor Type-LFA', fontsize=22)
plt.xlabel('Colombia', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)

pd.crosstab(df_all_sorted_by_Investor_Type["Investor Type"],df_all_sorted_by_Investor_Type['Industry Groups']).plot(kind='bar')
plt.title('Investor Type Vs Industry Groups', fontsize=22)
plt.xlabel('Investor Type', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)



#%%% 4. Muestre gráficamente los exits de capital privado en Colombia por deal size


#Organizacion Investor Type
df_all_sorted_by_Number_of_Exits = df_all.sort_values(["Number of Exits"], ascending=True)
df_all_sorted_by_Number_of_Exits = df_all_sorted_by_Number_of_Exits[df_all_sorted_by_Number_of_Exits.columns[:]][(df_all_sorted_by_Number_of_Exits["Pais"] == 'Colombia')]

pd.crosstab(df_all_sorted_by_Number_of_Exits["Number of Exits"],df_all_sorted_by_Number_of_Exits ['Organization Name']).plot(kind='bar')
plt.title('Colombia Vs Number of Exits', fontsize=22)
plt.xlabel('Colombia', fontsize=20)
plt.ylabel('Frequency of Number of Exits', fontsize=20)


#%%% 5  Muestre el crecimiento porcentual mensual de ingresos por inversión en Colombia en  comparación con los demás países. 


dataFrame_1=pd.get_dummies(df_all, columns =["Estimated Revenue Range"],drop_first=True)
dataFrame_1.corr()
print(dataFrame_1.shape)

# #Guarda el documeto
# dataFrame_1.to_excel('./DF_2.xls',index=False)

# def growth():

#     population = [1, 3, 4, 7, 8, 12]

#     # new list for growth rates
#     growth_rate = []

#     # for population in list
#     for pop in range(1, len(population)):

#         gnumbers = ((population[pop] - population[pop-1]) * 100.0 / population[pop-1])
#         growth_rate.append(gnumbers)
#         print growth_rate

# growth(


#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_Less than $1M'],
                     aggfunc="sum",values=dataFrame_1['Estimated Revenue Range_Less than $1M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_Less than $1M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_Less than $1M'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

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

#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$1M to $10M'],
                     aggfunc="sum",values=dataFrame_1['Estimated Revenue Range_$1M to $10M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$1M to $10M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$1M to $10M'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $1M to $10M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge2= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge2=tabla_merge2.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge2=tabla_merge2[tabla_merge2.columns[:]][(tabla_merge2["year"] >= 2016) & (tabla_merge2["year"] <= 2020)]
tabla_merge2["Last Funding Amount Currency (in USD) $1M to $10M"]=tabla_merge2["Last Funding Amount Currency (in USD) $1M to $10M"]/1e6
tabla_merge2['Ratio']=tabla_merge2['Estimated Revenue Range_$1M to $10M']*5.5/tabla_merge2["Last Funding Amount Currency (in USD) $1M to $10M"]

tabla_merge2["Growth $1M to $10M"]=0.0
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
            
pd.crosstab(tabla_merge2.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge2["Growth $1M to $10M"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $1M to $10M / LFA < $1M to $10M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$10M to $50M'],
                     aggfunc="sum",values=dataFrame_1['Estimated Revenue Range_$10M to $50M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$10M to $50M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$10M to $50M'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $10M to $50M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge3= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge3=tabla_merge3.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge3=tabla_merge3[tabla_merge3.columns[:]][(tabla_merge3["year"] >= 2016) & (tabla_merge3["year"] <= 2020)]
tabla_merge3["Last Funding Amount Currency (in USD) $10M to $50M"]=tabla_merge3["Last Funding Amount Currency (in USD) $10M to $50M"]/1e6
tabla_merge3['Ratio']=tabla_merge3['Estimated Revenue Range_$10M to $50M']*30/tabla_merge3["Last Funding Amount Currency (in USD) $10M to $50M"]

tabla_merge3["Growth $10M to $50M"]=0.0
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
            
pd.crosstab(tabla_merge3.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge3["Growth $10M to $50M"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $10M to $50M / LFA < $10M to $50M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1["Estimated Revenue Range_$50M to $100M"],
                     aggfunc="sum",values=dataFrame_1['Estimated Revenue Range_$50M to $100M'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$50M to $100M'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$50M to $100M'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $50M to $100M'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge4= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge4=tabla_merge4.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)

tabla_merge4=tabla_merge4[tabla_merge4.columns[:]][(tabla_merge4["year"] >= 2016) & (tabla_merge4["year"] <= 2020)]
tabla_merge4["Last Funding Amount Currency (in USD) $50M to $100M"]=tabla_merge4["Last Funding Amount Currency (in USD) $50M to $100M"]/1e6
tabla_merge4['Ratio']=tabla_merge4['Estimated Revenue Range_$50M to $100M']*75/tabla_merge4["Last Funding Amount Currency (in USD) $50M to $100M"]

tabla_merge4["Growth $50M to $100M"]=0.0
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
            
pd.crosstab(tabla_merge4.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge4["Growth $50M to $100M"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $50M to $100M / LFA  $50M to $100M', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################

#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1["Estimated Revenue Range_$500M to $1B"],
                     aggfunc="sum",values=dataFrame_1['Estimated Revenue Range_$500M to $1B'])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$500M to $1B'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$500M to $1B'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $500M to $1B'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge5= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge5=tabla_merge5.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)


tabla_merge5 =tabla_merge5[tabla_merge5.columns[:]][(tabla_merge5["year"] >= 2016) & (tabla_merge5["year"] <= 2020)]
tabla_merge5["Last Funding Amount Currency (in USD) $500M to $1B"]=tabla_merge5["Last Funding Amount Currency (in USD) $500M to $1B"]/1e6
tabla_merge5['Ratio']=tabla_merge5['Estimated Revenue Range_$500M to $1B']*750/tabla_merge5["Last Funding Amount Currency (in USD) $500M to $1B"]

tabla_merge5["Growth $500M to $1B"]=0.0
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
            
pd.crosstab(tabla_merge5.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge5["Growth $500M to $1B"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $500M to $1B / LFA  $500M to $1B', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################
#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1["Estimated Revenue Range_$1B to $10B"],
                     aggfunc="sum",values=dataFrame_1["Estimated Revenue Range_$1B to $10B"])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$1B to $10B'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$1B to $10B'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $1B to $10B'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge6= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge6=tabla_merge6.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)


tabla_merge6 =tabla_merge6[tabla_merge6.columns[:]][(tabla_merge6["year"] >= 2016) & (tabla_merge6["year"] <= 2020)]
tabla_merge6["Last Funding Amount Currency (in USD) $1B to $10B"]=tabla_merge6["Last Funding Amount Currency (in USD) $1B to $10B"]/1e6
tabla_merge6['Ratio']=tabla_merge6['Estimated Revenue Range_$1B to $10B']*11/tabla_merge6["Last Funding Amount Currency (in USD) $1B to $10B"]

tabla_merge6["Growth $1B to $10B"]=0.0
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
            
pd.crosstab(tabla_merge6.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge6["Growth $1B to $10B"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $1B to $10B / LFA  $1B to $10B', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################
#################################################################################################
tabla1 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1["Estimated Revenue Range_$10B+"],
                     aggfunc="sum",values=dataFrame_1["Estimated Revenue Range_$10B+"])

tabla1 = tabla1.drop(tabla1.columns[0], axis=1)
tabla1.rename(columns={tabla1.columns[0]: 'Estimated Revenue Range_$10B+'}, inplace=True)
tabla1 = tabla1.fillna(0)
# tabla1=tabla1.reset_index()

tabla2 = pd.crosstab(index=[dataFrame_1['Pais'],dataFrame_1['year']],
                     columns=dataFrame_1['Estimated Revenue Range_$10B+'],
                     aggfunc="sum",values=dataFrame_1['Last Funding Amount Currency (in USD)'])

tabla2 = tabla2.fillna(0)
tabla2['Last Funding Amount Currency (in USD) $10B+'] = tabla2[tabla2.columns[1]]
# tabla2['Last Funding Amount Currency (in USD) < 1M'] = tabla2[tabla2.columns[0]]+tabla2[tabla2.columns[1]]
tabla2 = tabla2.drop(tabla2.columns[1], axis=1)
tabla2 = tabla2.drop(tabla2.columns[0], axis=1)

# tabla2=tabla2.reset_index()

tabla_merge7= pd.concat([tabla1, tabla2], axis=1).reindex(tabla1.index)
tabla_merge7=tabla_merge7.reset_index()

# result = pd.concat([df1, df4], axis=1).reindex(df1.index)
#          pd.concat([df1, df4.reindex(df1.index)], axis=1)


tabla_merge7 =tabla_merge7[tabla_merge7.columns[:]][(tabla_merge7["year"] >= 2016) & (tabla_merge7["year"] <= 2020)]
tabla_merge7["Last Funding Amount Currency (in USD) $10B+"]=tabla_merge7["Last Funding Amount Currency (in USD) $10B+"]/1e6
tabla_merge7['Ratio']=tabla_merge7['Estimated Revenue Range_$10B+']*10/tabla_merge7["Last Funding Amount Currency (in USD) $10B+"]

tabla_merge7["Growth $10B+"]=0.0
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
            
pd.crosstab(tabla_merge7.Pais,tabla_merge1.year
            ,aggfunc="mean",values=tabla_merge7["Growth $10B+"]).plot(kind='bar')
plt.title('Crecimiento mensual Nominal: Ingreso $10B+ / LFA  $10B+', fontsize=22)
plt.ylabel('Crecimiento mensual Nominal ultimo quinquenio', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#############################################################################################


#marge de Informacion y comparcion.





