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
df1["Industries"] = np.where(df1["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df1["Industries"])
df1["Industries"].unique()
df1["Industries"] = np.where(df1["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Financial Services, FinTech, Lending, Online Portals, Small and Medium Businesses', 'Credit, Finance, Financial Services, FinTech', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Food and Beverage, Food Delivery, Restaurants, Retail Technology', 'Food Delivery, Restaurants, Waste Management', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='E-Commerce, Logistics, Software', 'Computer, SaaS, Software', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Credit, Finance, Financial Services, FinTech', 'Financial Services', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Finance, Financial Services, FinTech', 'Financial Services', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Financial Services, FinTech, Personal Finance', 'Financial Services', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Financial Services, FinTech, Payments', 'Financial Services', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Automotive, Transportation, Travel', 'Automotive', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Industrial, Mining, Oil and Gas', 'Oil and Gas', df1["Industries"])
df1["Industries"] = np.where(df1["Industries"] =='Energy, Energy Efficiency, Oil and Gas, Renewable Energy', 'Oil and Gas', df1["Industries"])
tabla = pd.pivot_table(df1, 'Last Funding Amount','Headquarters Location','Industries', aggfunc=np.sum)
tabla.fillna(0, inplace=True)
tabla.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Headquarters Location')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - Industies- Headquarters Location')
plt.show()
