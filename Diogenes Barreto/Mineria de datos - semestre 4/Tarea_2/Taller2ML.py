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
  

df_all=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
print(df_all.shape)

df_all = df_all.drop(['Organization Name URL'], axis=1)
# df_all = df_all.drop(["Website"], axis=1)
# df_all = df_all.drop(["Twitte"], axis=1)
# df_all = df_all.drop(["Facebook"], axis=1)
# df_all = df_all.drop(["LinkedIn"], axis=1)
df_all.shape

dfx=df_all["Headquarters Location"].str.split(",", n = 2, expand = True) 
####  adicionando las nuevas columnas a df original
df_all["Ciudad"]= dfx[0] 
df_all["Departamento"]= dfx[1] 
df_all["Pais"]= dfx[2]

#%% Organizacion por sorted

#Organizacion porRanking

df_all_sorted_by_CB_Rank = df_all.sort_values(["CB Rank (Company)"], ascending=True)
# df_all_sorted_by_CB_Rank["CB Rank (Company)"].head(10).plot(kind="barh")
# plt.show()

print(df_all_sorted_by_CB_Rank.head(10))

#Organizacion Revenue

df_all_sorted_by_Revenue = df_all.sort_values(["Estimated Revenue Range"], ascending=True)

#Organizacion Numero d eempleados

df_all_sorted_by_Employees = df_all.sort_values(["Number of Employees"], ascending=True)

#Organizacion Numero de Articulos

#df_all= df_all["Number of Articles"].replace({',' : '.','nan' : '0' })


df_all_sorted_by_Articles = df_all.sort_values(["Number of Articles"], ascending=True)

#Organizacion Invesment stage
df_all_sorted_by_Investment_stage = df_all.sort_values(["Investment Stage"], ascending=True)

#Organizacion Numero de Invesment 
df_all_sorted_by_Number_Investment = df_all.sort_values(["Number of Investments"], ascending=True)


#Organizacion Numero de Funding Rounds 
df_all_sorted_by_Number_Fundings = df_all.sort_values(["Number of Funding Rounds"], ascending=True)


#Organizacion Last Founding Amount
df_all_sorted_by_Last_Fundings = df_all.sort_values(["Last Funding Amount"], ascending=True)

#Organizacion Last Founding Type
df_all_sorted_by_Last_Fundings_Type = df_all.sort_values(["Last Funding Type"], ascending=True)

#Organizacion Last Equity Founding Amount
df_all_sorted_by_Last_Equity_Fundings = df_all.sort_values(["Last Equity Funding Amount"], ascending=True)

#%% Graficas

#Compara df con organizacion
pd.crosstab(df_all_sorted_by_CB_Rank["Pais"].head(50),df_all_sorted_by_CB_Rank["Organization Name"].head(50)).plot(kind="bar")
plt.title('Pais Vs Companies')
plt.xlabel("Pais")
plt.ylabel("Organization Name")
plt.savefig('Pais Vs Companies')

pd.crosstab(df_all_sorted_by_Revenue["Pais"].head(50),df_all_sorted_by_Revenue["Estimated Revenue Range"].head(50)).plot(kind="bar")
plt.title('Pais Vs Revenue')
plt.xlabel("Pais")
plt.ylabel("Revenue")
plt.savefig('Pais Vs Revenue')

pd.crosstab(df_all_sorted_by_Revenue.Pais, df_all_sorted_by_Revenue["Estimated Revenue Range"]).plot(kind="bar")
plt.title('Pais Vs Revenue')
plt.xlabel("Pais")
plt.ylabel("Revenue")
plt.savefig('Pais Vs Revenue')

pd.crosstab(df_all_sorted_by_Employees.Pais, df_all_sorted_by_Employees["Number of Employees"]).plot(kind="bar")
#pd.crosstab(df_all_sorted_by_Employees["Number of Employees"].head(10),df_all_sorted_by_Employees["Ciudad"].head(10)).plot(kind="bar")
plt.title('Pais Vs Employees')
plt.xlabel("Pais")
plt.ylabel("Number of Employees")
plt.savefig('Pais Vs Employees')

pd.crosstab(df_all_sorted_by_Investment_stage.Pais, df_all_sorted_by_Investment_stage["Investment Stage"]).plot(kind="bar")
plt.title('Pais Vs Investment Stage')
plt.xlabel("Pais")
plt.ylabel("Investment Stage")
plt.savefig('Pais Vs Investment Stage')

pd.crosstab(df_all_sorted_by_Number_Fundings.Pais, df_all_sorted_by_Number_Fundings["Number of Funding Rounds"]).plot(kind="bar")
plt.title('Pais Vs Number of Fundings Rounds')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig("Pais Vs Number of Fundings Rounds")

pd.crosstab(df_all_sorted_by_Last_Fundings.Pais, df_all_sorted_by_Last_Fundings["Last Funding Amount"]).plot(kind="bar")
plt.title('Pais Vs Last Funding Amounts')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig('Pais Vs Last Funding Amounts')

pd.crosstab(df_all_sorted_by_Last_Fundings_Type.Pais, df_all_sorted_by_Last_Fundings_Type["Last Funding Type"]).plot(kind="bar")
plt.title('Pais Vs Last Funding Type')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig('Pais Vs Last Funding Type')

pd.crosstab(df_all_sorted_by_Last_Equity_Fundings.Pais, df_all_sorted_by_Last_Equity_Fundings["Last Equity Funding Amount"]).plot(kind="bar")
plt.title('Pais Vs Last Equity Funding Amount')
plt.xlabel("Pais")
plt.ylabel("Number of Fundings Rounds")
plt.savefig('Pais Vs Last Equity Funding Amount')



sns.bartplot(x="Pais", y="Last Funding Amount", data=df_all, estimator=np.median)


data=[df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11]
for i in data:
    sns.set_theme(style="darkgrid")
    i = sns.load_dataset("")
    sns.displot(i, x="Headquarters Location", col="Last Funding Amount", row="sex", binwidth=3, height=3, facet_kws=dict(margin_titles=True)
    