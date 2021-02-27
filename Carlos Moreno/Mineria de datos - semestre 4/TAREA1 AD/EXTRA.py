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
fig, ax1 = plt.subplots(figsize=(7,5))
ax2=ax1.twinx()
sns.barplot(x='year', y='CB Rank (Company)', data=df1, hue='Industries',ax=ax1)
sns.lineplot(x='year',y='Headquarters Location', data=df1, hue='Industries', marker='d', ax=ax2)
plt.show()
