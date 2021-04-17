import pandas as pd #importamos librerÃ­as para analisis de datos
import numpy as np #librerÃ­a para anÃ¡lisis numÃ©rico
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import plotnine as p9
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() #Asign
#%% Lectura de los datos
text_file = "Col.csv" 
Col=pd.read_csv(text_file)
Col=Col.drop(columns=["Organization Name URL","Website","Facebook","Twitter","Contact Email","Phone Number","LinkedIn"])
Col
excel_file = "tp.xlsx"
tp=pd.read_excel(excel_file)
excel_file = "Emp.xlsx"
Emp=pd.read_excel(excel_file)
#%% Unirlos en una sola data
Col["Organization Name"]=Col["Organization Name"].str.upper()#cambia los nombres de las empresasa a mayusculas
tp["Organization"]=tp["Organization"].str.upper()
Emp["Name"]=Emp["Name"].str.upper()
Col = Col.join(tp, rsuffix='_right')#se inserta la data de l top 100 a la derecha
print(Col)
Col = Col.join(Emp, rsuffix='_right')
print(Col)
Col["Bandera"]=0
print(Col)
#%% Intersección de los datos
df1=set(Col['Organization Name']).intersection(set(tp['Organization']))#se intersectan las dos para saber  que tienen de comun
df2=set(Col['Organization Name']).intersection(set(Emp['Name']))#se intersectan las dos para saber  que tienen de comun
df3=set(Emp['Name']).intersection(set(tp['Organization']))#se intersectan las dos para saber  que tienen de comun
df4=set(Col['Organization Name']).intersection(set(tp['Organization']).intersection(set(Emp['Name']))) #Bandera para los tres datos 
for n in df4:#se relializa un ciclo para colocar 1 en l bandera que representa las que estan las tres datas
    Col.loc[Col['Organization Name'] == n, ['Bandera']] = 1
#%% Gráfico 0
Col.plot.scatter(x='CB Rank (Company)', y='Bandera')
plt.title('Comparación', fontsize = 22)
plt.xlabel("CB Rank (Company)", fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 0')
#%% Grafico 1
Col.plot.scatter(x='BuiltWith - Active Tech Count', y='Bandera')
plt.title('Comparación', fontsize = 22)
plt.xlabel("BuiltWith - Active Tech Count", fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 1')
#%%Grafico 2
Col.plot.scatter(x='Last Funding Amount', y='Bandera')
plt.title('Comparación', fontsize = 22)
plt.xlabel("Last Funding Amount", fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 2')
#%%Grafico 3
Col.plot.scatter(x='Last Equity Funding Amount', y='Bandera')
plt.title('Comparación', fontsize = 22)
plt.xlabel("Last Equity Funding Amount", fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 3')
#%% Grafico 4
Col.plot.scatter(x='Total Funding Amount Currency (in USD)', y='Bandera')
plt.title('Comparación', fontsize = 22)
plt.xlabel("Total Funding Amount Currency (in USD)", fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 4')
#%% Grafico 5
plot = pd.crosstab(index=Col['Number of Employees'],columns=Col['Bandera']).apply(lambda r: r/r.sum() *100, axis=1).plot(kind='bar')
plt.savefig('Fig 5')
#%% Grafico 6
plot = pd.crosstab(index=Col['Number of Articles'],columns=Col['Bandera']).apply(lambda r: r/r.sum() *100, axis=1).plot(kind='bar')
plt.savefig('Fig 6')
Col.to_excel("Colombia.xls")
#%% Grafico 7
pd.crosstab(Col['Number of Employees'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Number of Employees', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 7')
#%% Gráfico 8
pd.crosstab(Col['Number of Articles'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Number of Employees', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 8')
#%% Gráfico 9
pd.crosstab(Col['Number of Lead Investors'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Number of Lead Investors', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 9')
#%% Gráfico 10
pd.crosstab(Col['Number of Investors'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Number of Investors', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 10')
#%% Gráfico 11
pd.crosstab(Col['Number of Funding Rounds'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Number of Funding Rounds', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 11')
#%% Gráfico 12
pd.crosstab(Col['BuiltWith - Active Tech Count'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('BuiltWith - Active Tech Count', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 12')
#%% Grafico 13
pd.crosstab(Col['G2 Stack - Total Products Active'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('G2 Stack - Total Products Active', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 13')
#%% Grafico 13
pd.crosstab(Col['Estimated Revenue Range'],Col.Bandera).plot(kind='bar')
plt.title('Comparación', fontsize = 22)
plt.xlabel('Estimated Revenue Range', fontsize = 20)
plt.ylabel('Bandera', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.savefig('Fig 14')
