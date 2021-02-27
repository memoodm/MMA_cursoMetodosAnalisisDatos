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
df1['Last Funding Date'] = pd.to_datetime(df1['Last Funding Date'])
df1['year']= df1['Last Funding Date'].dt.year
tabla = pd.pivot_table(df1, 'Last Funding Amount','year','Headquarters Location', aggfunc=np.sum )
tabla.fillna(0, inplace=True)
tabla.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('LFA USD')

