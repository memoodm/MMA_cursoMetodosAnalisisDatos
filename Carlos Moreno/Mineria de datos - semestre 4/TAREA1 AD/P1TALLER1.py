import pandas as pd #importamos librerÃ­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from pandas_profiling import ProfileReport
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asign
excel_file = "Colombia-Feb21.xlsx" #asigno archivo
df = pd.read_excel(excel_file, header= 0)
df=df.replace({"BogotÃ¡, Distrito Especial, Colombia":"Bogotá","MedellÃ­n, Antioquia, Colombia":"Medellín", "UsaquÃ©n, Distrito Especial, Colombia":"Usaquén"})
df = df.drop(['Organization Name URL'], axis=1)
df.head()
df.count()
df.describe()
df1=df.dropna()
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Organization Name"], margins=True)
plot = (100 * df1['Headquarters Location'].value_counts() / len(df1['Organization Name'])).plot(kind='barh', title='Localización de Sedes %')
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Organization Name"])
plot = pd.crosstab(index=df1['Headquarters Location'].head(20),columns=df1['Organization Name']).plot(kind='bar')
