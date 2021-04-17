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

from imblearn.over_sampling import SMOTE

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

df_marge3['Intersección'] = 0

#asigana 1 en las intesecciones y ceros a los no intersecciones.

for i in intersec:
   df_marge3.loc[df_marge3['Organization Name'] == i, ['Intersección']] = 1
#Guarda el documeto
df_marge3.to_csv('./dataall.csv',index=False)


#datos nulos
# Lista = [df_marge3]

# for i in Lista:
# print(i, i.isnull().sum(),)

# ## Envia el resultado de nulos a un archivo de texto
# with open("fileC.txt", "w") as output:
# output.write(str(Lista))



#Perfilaje de Variables
from pandas_profiling import ProfileReport
ProfileReport(df_marge3)
profile = ProfileReport(df_marge3, title="Profiling Report",explorative=True)

profile = ProfileReport(df_marge3)
profile.to_file('profile_report.html')



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
           & ((df_marge3['Organization Name'] != 'Qinaya')  & (df_marge3['Organization Name'] != 'Partsium'))].index)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Chile, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Las Vegas, Sucre, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Los Angeles, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Madrid, Distrito Especial, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Maryland, Cundinamarca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Miami, Magdalena, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='MÃ©xico, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='PanamÃ¡, Magdalena, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='PerÃº, Valle del Cauca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Perú, Valle del Cauca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Canadá, Cundinamarca, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Panamá, Magdalena, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='México, Huila, Colombia'].index, inplace =True)
df_marge3.drop(df_marge3[df_marge3['Headquarters Location']=='Andalucía, Valle del Cauca, Colombia'].index, inplace =True)
df_marge3.shape





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



#Es una funcion para eliminar entradas nulas de acuerdo al porcentaje    
def cols_90per_nulls(df_marge3):
    count = 0
    cols_to_drop = {}
    for col in df_marge3.columns: 
        per_nulls = df_marge3[col].isna().sum()/len(df_marge3[col])
        if per_nulls >= 0.8:
            cols_to_drop[col] = per_nulls 
            # print(col, per_nulls)
            count+=1
        else:
            None
    
    print('Number of cols with >80% nulls:', count)
    return cols_to_drop


nulls_Colums = cols_90per_nulls(df_marge3)

dataFrame=df_marge3.drop(columns = nulls_Colums)#Elimina las columnas con un 75% nan

print(dataFrame.shape)


#Eliminacion de Columnasde variables que no aportan con base al perfilamiento

dataFrame = dataFrame.drop(['Organization Name URL'], axis=1)
dataFrame = dataFrame.drop(['Contact Email'], axis=1) 
dataFrame = dataFrame.drop(['Phone Number'], axis=1)  
dataFrame= dataFrame.drop(['Full Description'], axis=1) 
dataFrame = dataFrame.drop(['Website'], axis = 1)
dataFrame = dataFrame.drop(['Twitter'], axis = 1)
dataFrame = dataFrame.drop(['Facebook'], axis = 1)
dataFrame = dataFrame.drop(['LinkedIn'], axis = 1)

dataFrame.shape
  
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

# #Correlacion de variables numericas
dataFrame.corr(method='pearson')
dataFrame.corr()
# #Mapa de calor para las variables numericas y encontrar el grado de importancia de cada variable

sns.heatmap(dataFrame.corr(),annot=True,cmap="RdYlGn") 

# # sns.heatmap(dataFrame.corr(method='spearman'));
# #Veremos mejor las distribuciones de cada variable apartir del comportamiento de cada pais
sns.pairplot(dataFrame)

dataFrame_2 = dataFrame.fillna(0)

#%%DATA REGRESION LOGISTICA

dataFrame_2 = dataFrame.fillna(0)

dataFrame_2 = dataFrame_2.drop(['Organization Name'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Industries'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Headquarters Location'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Description'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Headquarters Regions'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Operating Status'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Founded Date'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Founded Date Precision'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Company Type'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Industry Groups'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Founders'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Funding Date'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Funding Amount'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Funding Amount Currency'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Funding Type'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Equity Funding Amount'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Last Equity Funding Amount Currency'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Total Equity Funding Amount'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Total Equity Funding Amount Currency'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Total Funding Amount'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Total Funding Amount Currency'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Top 5 Investors'], axis=1)
dataFrame_2 = dataFrame_2.drop(['CB Rank (Organization)'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Ciudad'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Departamento'], axis=1)
dataFrame_2 = dataFrame_2.drop(['Pais'], axis=1)
dataFrame_2 = dataFrame_2.drop(['IPO Status'], axis=1)
dataFrame_2.shape

dataFrame_3=pd.get_dummies(dataFrame_2, columns =['Estimated Revenue Range', 'Number of Employees', 'Funding Status', 'Last Equity Funding Type'],drop_first=True)
dataFrame_3.corr()

dataFrame_3["CB Rank (Company)"] = dataFrame_3["CB Rank (Company)"].str.replace(r'\D', '')
dataFrame_3["CB Rank (Company)"] = pd.to_numeric(dataFrame_3["CB Rank (Company)"])


#%%  Regresion Logistica

Xtrain = dataFrame_3[[dataFrame_3.columns[0]]]

for i in range(1,(dataFrame_3.shape[1])):
    if dataFrame_3.columns[i] != "Intersección":
        Xtrain = Xtrain.join(dataFrame_3[[dataFrame_3.columns[i]]])

# Xtrain=Xtrain.dropna()
# Xtrain = Xtrain.fillna(0)


#Es una funcion para eliminar entradas nulas de acuerdo al porcentaje    

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
    
    print('Number of cols with > ', perc*100, '% nulls:', count)
    return cols_to_drop

# Columnas comas del 80% de datos nulos

dict_col_zeros=cols_90per_zeros(Xtrain, 0.95)

Xtrain = Xtrain.drop(columns=dict_col_zeros)
Xtrain.shape
ytrain = dataFrame_3[["Intersección"]]

Xtrain = Xtrain.drop(['Number of Employees_1-10'], axis=1)
Xtrain = Xtrain.drop(['Funding Status_Seed'], axis=1)
Xtrain = Xtrain.drop(['Last Equity Funding Type_Pre-Seed'], axis=1)
Xtrain = Xtrain.drop(['Last Equity Funding Type_Seed'], axis=1)


from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.3, random_state=0)

smt = SMOTE(random_state=0)  # Muestra de entrenamiento balanceado con variable binaria
data_X, data_y= smt.fit_sample(X_train, y_train) # en misma proporcion


sns.countplot(data_X= "Number Of Articles", data = data_y)
plt.title('')
plt.show()


import statsmodels.api as sm # libria para encontrar varias funciones de estimaciones de moleslos estadisticos#

log_reg = sm.Logit(data_y, data_X).fit(method='newton', maxiter=1000)

print(log_reg.summary())

# Xtest=Xtrain
# ytest = dataFrame_3[["Intersección"]]

yhat = log_reg.predict(X_test)
prediction = list(map(round, yhat))
print('Actual values', list(y_test.values)) 
print('Predictions :', prediction)

from sklearn.metrics import (confusion_matrix,  
                           accuracy_score)
cm = confusion_matrix(y_test, prediction)

print ("Confusion Matrix : \n", cm)
print('Test accuracy = ', accuracy_score(y_test, prediction))


# sns.countplot(x='Survived',data=data_y)
# plt.title('Not Survived vs Survived')
# plt.show()

# logreg = LogisticRegression(max_iter=500)
# logreg.fit(data_X, data_y.values.ravel())

# y_pred = logreg.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# from sklearn.metrics import confusion_matrix
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))





#%% PLOT ONE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(dataFrame['Organization Name'],dataFrame.Intersección).plot(kind='barh')
plt.title('Organization Name for 12 Company', fontsize = 22)
plt.xlabel('Organization Name', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Organization Name')

#%% PLOT THREE - DIAGRAMA DE BARRAS HORIZONTALES
pd.crosstab(dataFrame['Industries'],dataFrame.Intersección).plot(kind='barh')
plt.title('Industries for 12 Company', fontsize = 22)
plt.xlabel('Industries', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Industries')

#%%
pd.crosstab(dataFrame['Headquarters Location'],dataFrame.Intersección).plot(kind='barh')
plt.title('Headquarters Location for 12 Company', fontsize = 22)
plt.xlabel('Headquarters Location', fontsize = 20)
plt.ylabel('12 Company', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Headquarters Location')

#%%
pd.crosstab(dataFrame['Description'],dataFrame.Intersección).plot(kind='barh')
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

