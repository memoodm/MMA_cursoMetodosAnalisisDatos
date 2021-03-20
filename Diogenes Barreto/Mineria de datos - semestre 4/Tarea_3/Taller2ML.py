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

os.chdir("C:/Users/LENOVO/Documents/Clase4/")

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

#Pais

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

df_all = df_all.drop(['Organization Name URL'], axis=1)
df_all.shape

#Cracterizacion de las variables en el nuevo df_frame
from pandas_profiling import ProfileReport
ProfileReport(df_all)
profile = ProfileReport(df_all, title="Pandas Profiling Report",explorative=True)
#reporte = ProfileReport(df_all).to_html("report.html")
#profile.to_scv("Pandas Profiling Report.txt")

profile = ProfileReport(df_all)
profile.to_file('profile_report.html')

#Es una funcion para eliminar entradas nulas de acuerdo al porcentaje    
def cols_90per_nulls(df_all):
    count = 0
    cols_to_drop = {}
    for col in df_all.columns: 
        per_nulls = df_all[col].isna().sum()/len(df_all[col])
        if per_nulls >= 0.93:
            cols_to_drop[col] = per_nulls 
            # print(col, per_nulls)
            count+=1
        else:
            None
    
    print('Number of cols with >90% nulls:', count)
    return cols_to_drop

nulls_Colums = cols_90per_nulls(df_all)


dataFrame = df_all.drop(columns=nulls_Colums)#Elimina las columnas con un 75% nan
print(dataFrame.shape)



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

UsaquenDataframe = dataFrame[dataFrame["Ciudad"] == "Usaquen"]
print(len(UsaquenDataframe))
UsaquenDataframe



dataFrame.corr(method='pearson')
dataFrame.corr()
#Mapa de calor para las variables numericas y encontrar el grado de importancia de cada variable
sns.heatmap(dataFrame.corr(method='spearman'));
#Veremos mejor las distribuciones de cada variable apartir del comportamiento de cada pais
sns.pairplot(dataFrame)

# #Importamos el método Fit_everything 
# from reliability.Fitters import Fit_Everything 
# # Convertimos nuestros dataframes en arrays ya que Reliability tienen este requerimiento 
# porosidad= np.array(df['Last Funding Amount Currency (in USD)']) 
# # Pasamos los datos de porosidad a la función, con los parámetros de gráficos con valor False para que para solo nos 
# # entregue la tabla resumen de la evaluación de los modelos. 
# fit_porosidad = Fit_Everything(failures=porosidad,
# show_histogram_plot=False,
# show_probability_plot=False, show_PP_plot=False)



#%% Organizacion por sorted

#Organizacion porRanking

dataFrame_sorted_by_CB_Rank = dataFrame.sort_values(["CB Rank (Company)"], ascending=True)
# df_all_sorted_by_CB_Rank["CB Rank (Company)"].head(10).plot(kind="barh")
# plt.show()

print(dataFrame_sorted_by_CB_Rank.head(10))

#Organizacion Revenue

dataFrame_sorted_by_Revenue = dataFrame.sort_values(["Estimated Revenue Range"], ascending=True)

#Organizacion Numero d eempleados

dataFrame_sorted_by_Employees = dataFrame.sort_values(["Number of Employees"], ascending=True)

#Organizacion Numero de Articulos

dataFrame["Number of Articles"] = pd.to_numeric(dataFrame["Number of Articles"])
dataFrame_sorted_by_Articles = dataFrame.sort_values(["Number of Articles"], ascending=True)

#Organizacion Invesment stage

dataFrame_sorted_by_Investment_stage = dataFrame.sort_values(["Investment Stage"], ascending=True)

#Organizacion Numero de Invesment 
dataFrame_sorted_by_Number_Investment = dataFrame.sort_values(["Number of Investments"], ascending=True)


#Organizacion Numero de Funding Rounds 
dataFrame_sorted_by_Number_Fundings = dataFrame.sort_values(["Number of Funding Rounds"], ascending=True)


#Organizacion Last Founding Amount
dataFrame_sorted_by_Last_Fundings = dataFrame.sort_values(["Last Funding Amount"], ascending=True)

#Organizacion Last Founding Type
dataFrame_sorted_by_Last_Fundings_Type = dataFrame.sort_values(["Last Funding Type"], ascending=True)

#Organizacion Last Equity Founding Amount
dataFrame_sorted_by_Last_Equity_Fundings = dataFrame.sort_values(["Last Equity Funding Amount"], ascending=True)

#%% Graficas

#Compara df con organizacion Organizacion porRanking
pd.crosstab(dataFrame_sorted_by_CB_Rank["Pais"].head(50),dataFrame_sorted_by_CB_Rank["Organization Name"].head(50)).plot(kind="bar")
plt.title('Pais Vs Companies')
plt.xlabel("Pais")
plt.ylabel("Organization Name")
plt.savefig('Pais Vs Companies')

##Organizacion Revenue

pd.crosstab(dataFrame_sorted_by_Revenue.Pais,dataFrame_sorted_by_Revenue["Estimated Revenue Range"],aggfunc="sum",\
            values=dataFrame_sorted_by_Revenue['Last Funding Amount Currency (in USD)']).plot(kind="bar")
plt.savefig('Pais Vs Last Funding Amounts')

#Organizacion Numero de empleados
pd.crosstab(dataFrame_sorted_by_Employees.Pais, dataFrame_sorted_by_Employees["Number of Employees"],aggfunc="sum",\
            #values=dataFrame_sorted_by_Employees["Organization Name"]).plot(kind="bar")
    
pd.crosstab(dataFrame_sorted_by_Employees.Pais, dataFrame_sorted_by_Employees["Number of Employees"]).plot(kind="bar")
#pd.crosstab(df_all_sorted_by_Employees["Number of Employees"].head(10),df_all_sorted_by_Employees["Ciudad"].head(10)).plot(kind="bar")
plt.title('Pais Vs Employees')
plt.xlabel("Pais")
plt.ylabel("Number of Employees")
plt.savefig('Pais Vs Employees')


#Organizacion Invesment stage
pd.crosstab(dataFrame_sorted_by_Investment_stage.Pais, dataFrame_sorted_by_Investment_stage["Investment Stage"]).plot(kind="bar")
plt.title('Pais Vs Investment Stage')
plt.xlabel("Pais")
plt.ylabel("Investment Stage")
plt.savefig('Pais Vs Investment Stage')

#Organizacion Numero de Funding Rounds 

pd.crosstab(dataFrame_sorted_by_Number_Fundings.Pais, dataFrame_sorted_by_Number_Fundings["Number of Funding Rounds"],aggfunc="sum",\
            values = dataFrame_sorted_by_Number_Fundings["Pais"]).plot(kind="bar")
    
pd.crosstab(dataFrame_sorted_by_Number_Fundings.Pais, dataFrame_sorted_by_Number_Fundings["Number of Funding Rounds"]).plot(kind="bar")
plt.title('Pais Vs Number of Fundings Rounds')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig("Pais Vs Number of Fundings Rounds")

#Organizacion Last Founding Amount

pd.crosstab(dataFrame_sorted_by_Last_Fundings.Pais,dataFrame_sorted_by_Last_Fundings["Estimated Revenue Range"],aggfunc="sum",\
            values=dataFrame_sorted_by_Last_Fundings['Last Funding Amount Currency (in USD)']).plot(kind="bar")
plt.savefig('Pais Vs Last Funding Amounts')
   
# pd.crosstab(dataFrame_sorted_by_Last_Fundings.Pais, dataFrame_sorted_by_Last_Fundings["Last Funding Amount"]).plot(kind="bar")
# plt.title('Pais Vs Last Funding Amounts')
# plt.xlabel("Pais")
# plt.ylabel("Number of Fundings Rounds")
# plt.savefig('Pais Vs Last Funding Amounts')

#Organizacion Last Founding Type
pd.crosstab(dataFrame_sorted_by_Last_Fundings_Type.Pais, dataFrame_sorted_by_Last_Fundings_Type["Last Funding Type"]).plot(kind="bar")
plt.title('Pais Vs Last Funding Type')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig('Pais Vs Last Funding Type')

#Organizacion Last Equity Founding Amount
pd.crosstab(dataFrame_sorted_by_Last_Equity_Fundings.Pais, dataFrame_sorted_by_Last_Equity_Fundings["Last Equity Funding Amount"]).plot(kind="bar")
plt.title('Pais Vs Last Equity Funding Amount')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig('Pais Vs Last Equity Funding Amount')


# for i in datframe:
#     sns.set_theme(style="darkgrid")
#     i = sns.load_dataset("")
#     sns.displot(i, x="Headquarters Location", col="Last Funding Amount", row="sex", binwidth=3, height=3, facet_kws=dict(margin_titles=True)
    