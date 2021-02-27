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
df1=df.dropna()#borra los vacios
df1.iloc[:, 8]
a=df1.info
#sns.pairplot(df1)
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Organization Name"], margins=True)
#City = pd.crosstab(index=df1["Headquarters Location"],columns=df1["Organization Name"], margins=True)
#City.sort_values(["All"], ascending = False)
plot = (100 * df1['Headquarters Location'].value_counts() / len(df1['Organization Name'])).plot(kind='barh', title='Localización de Sedes %')
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Organization Name"])
plot = pd.crosstab(index=df1['Headquarters Location'],columns=df1['Organization Name']).plot(kind='bar')
df1.groupby(["Headquarters Location"])["Organization Name"].nunique().plot(kind='bar')
#sns.jointplot(df1['Headquarters Location'],df1['Organization Name'])
#plt.show()
#PUNTO 2
pd.crosstab(index=df1["Headquarters Location"],columns=df1["Last Funding Amount"].sum)
plot = pd.crosstab(index=df1['Headquarters Location'],columns=df1['Last Funding Amount']).plot(kind='bar')
df.plot.scatter(x="Headquarters Location", y="Organization Name")
#plt.title('ciudades por concentracion')
#plt.xlabel("Headquarters Location")
#plt.ylabel("Organization Name")
sns.swarmplot(x="Headquarters Location", y="Last Funding Amount", data=df1)#grafico de dispersión de las variables
df1.groupby(["Headquarters Location"])["Last Funding Amount"].nunique().plot(kind='barh')
df1.unstack()#se dan los valores de las dos variables
from seaborn import lmplot
lmplot('Headquarters Location', 'Last Funding Amount', data=df1, ci=None)#se observa la forma en que se acumulan los datos
lmplot(x="Headquarters Location", y="Last Funding Amount", data=df1)
#df1.groupby('Headquarters Location').count()["Last Funding Amount"].plot(kind='bar')
fig, ax = plt.subplots()
df1.plot(x = 'Headquarters Location', y = 'Last Funding Amount', ax = ax,kind="bar")
plt.show()
#punto 2
pkt_cnt = df1["Headquarters Location"].value_counts(sort=False).sort_index()
pkt_cnt.sort_values("Headquarters Location", inplace=True)
yearly_counts = df1.groupby(['Headquarters Location', 'Last Funding Amount'])['Last Funding Amount'].count()
yearly_counts
sns.catplot(x='Last Funding Amount', y='Headquarters Location', kind="bar", data=df1)
#este es el grafico de las dos varibles

#tercer punto
df1['Last Funding Date'] = pd.to_datetime(df1['Last Funding Date'])
df1['year']= df1['Last Funding Date'].dt.year
tabla2 = pd.pivot_table(df1, 'Last Funding Amount','year','Headquarters Location', aggfunc=np.sum )
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('year')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - year - Headquarters Location')
plt.show()


ids = df1['Industries'].unique()
for item in ids:
    df2 = df1[(df1['Industries']==item)]

df1.loc[df1['Industries'] == 'Accounting, Financial Services, Small and Medium Businesses']

df1.dtypes
df1['Industries'].nunique()

#cuarto punto

    
df2["Industries"] = np.where(df2["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df2["Industries"])

df2["Industries"].unique()# Permite Unir las categorias en una sola#

df2["Industries"] = np.where(df2["Industries"] =='Consumer Goods, E-Commerce, Food Delivery', 'Consumer Goods, E-Commerce, Pet, Retail', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Lending, Online Portals, Small and Medium Businesses', 'Credit, Finance, Financial Services, FinTech', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Food and Beverage, Food Delivery, Restaurants, Retail Technology', 'Food Delivery, Restaurants, Waste Management', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='E-Commerce, Logistics, Software', 'Computer, SaaS, Software', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Credit, Finance, Financial Services, FinTech', 'Financial Services', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Finance, Financial Services, FinTech', 'Financial Services', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Personal Finance', 'Financial Services', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Financial Services, FinTech, Payments', 'Financial Services', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Automotive, Transportation, Travel', 'Automotive', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Industrial, Mining, Oil and Gas', 'Oil and Gas', df2["Industries"])
df2["Industries"] = np.where(df2["Industries"] =='Energy, Energy Efficiency, Oil and Gas, Renewable Energy', 'Oil and Gas', df2["Industries"])

tabla3 = pd.pivot_table(df2, 'Last Funding Amount','Headquarters Location','Industries', aggfunc=np.sum)
tabla3.fillna(0, inplace=True)
tabla3.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Headquarters Location')
plt.ylabel('LFA USD')
plt.title('Last Funding Amount - Industies- Headquarters Location')
plt.show()


#Quinto punto
pkt_cnt = df1["Industries"].value_counts(sort=False).sort_index()
pkt_cnt.sort_values("Industries", inplace=True)
yearly_counts = df1.groupby(['Industries', 'Last Equity Funding Amount Currency (in USD)'])['Last Equity Funding Amount Currency (in USD)'].count()
yearly_counts
sns.catplot(x='Last Equity Funding Amount Currency (in USD)', y='Industries', kind="bar", data=df1)

lmplot('Industries', 'Last Equity Funding Amount Currency (in USD)', data=df1, ci=None)#se observa la forma en que se acumulan los datos
lmplot(x="Industries", y="Last Equity Funding Amount Currency (in USD)", data=df1)

pd.crosstab(index=df1["Industries"],columns=df1["Last Equity Funding Amount Currency (in USD)"].sum)
plot = pd.crosstab(index=df1['Industries'],columns=df1['Last Equity Funding Amount Currency (in USD)']).plot(kind='bar')
df.plot.scatter(x="Industries", y="Last Equity Funding Amount Currency (in USD)")

pd.crosstab(index=df1['Industries'],columns = df1['Last Equity Funding Amount Currency (in USD)'])
plot = pd.crosstab(index=df1['Industries'].head(20),columns = df1['Last Equity Funding Amount Currency (in USD)'].head(20)).plot(kind='bar')


#punto sexto
pd.value_counts(df1["year"]) 

#Grafico de lineas y barras en uno solo
fig, ax1 = plt.subplots(figsize=(7,5))
ax2=ax1.twinx()
sns.barplot(x='year', y='CB Rank (Company)', data=df1, hue='Industries',ax=ax1)
sns.lineplot(x='year',y='Headquarters Location', data=df1, hue='Industries', marker='d', ax=ax2)
plt.show()
