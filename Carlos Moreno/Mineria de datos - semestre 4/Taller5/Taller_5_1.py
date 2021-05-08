import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
os.chdir("C:\\Users\\Casa\\Documents\PROGRMASCARLOS")#ruta del archivo donde esta alojado
cwd=os.getcwd() 
text_file = "Col.csv" 
Col=pd.read_csv(text_file)
Col
text_file = "Arg.csv" 
Arg=pd.read_csv(text_file)
text_file = "Braz.csv" 
Braz=pd.read_csv(text_file)
text_file = "Chi.csv" 
Chi=pd.read_csv(text_file)
text_file = "Uru.csv" 
Uru=pd.read_csv(text_file)
text_file = "Mex.csv" 
Mex=pd.read_csv(text_file)
text_file = "Ale.csv" 
Ale=pd.read_csv(text_file)
text_file = "USA.csv" 
USA=pd.read_csv(text_file)
text_file = "Isra.csv" 
Isra=pd.read_csv(text_file)
text_file = "Suiz.csv" 
Suiz=pd.read_csv(text_file)
text_file = "Esp.csv" 
Esp=pd.read_csv(text_file)
print(Col.isnull().sum())
dt=[Col,Braz,Chi,Uru,Mex,Arg]
dt=pd.concat([Col,Braz,Chi,Uru,Mex,Arg])
dt=pd.concat([Col,Braz,Chi,Uru,Mex,Arg])
#separa la columna por departamento, ciudad,pais
df1 = dt['Headquarters Location'].str.split(",", n = 3, expand = True) 
dt["Ciudad"]= df1[0]
dt["Departamento"]= df1[1]
dt["Pais"]= df1[2]
#quita los espacios en blanco
def correct_word(word):
    new_word = word.split()[0]
    return new_word
dt['Pais'] = dt['Pais'].apply(correct_word)
dt=dt.drop(columns=["Organization Name URL","Website","Facebook","Twitter","Contact Email","Phone Number","LinkedIn"])
#%%
#PUNTO 1
dt['Last Funding Date'] = pd.to_datetime(dt['Last Funding Date'])
dt['year']= dt['Last Funding Date'].dt.year
df=dt[(dt["Headquarters Regions"]== 'Latin America') & (dt["year"] ==2021)]
df['Last Funding Amount Currency (in USD)']=df['Last Funding Amount Currency (in USD)']/1e6
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Capital Invertido por País en 2021', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Capital Invertido en Millones de Doláres', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
#%%
#Punto 2  (a)
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Number of Investors']).plot(kind='bar')
plt.title('Número de Inversoras por País', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Número de inversoras', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#(b)
df = df.fillna(0)
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Number of Acquisitions']).plot(kind='bar')
plt.title('Número de Adquisiciones  por País', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Número de Adquisiciones', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#(c)
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Total Equity Funding Amount']).plot(kind='bar')
plt.title('Monto Total de Financiamiento de Capital  por País', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Monto de Financiamiento de Capital', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
#(d)
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Number of Funding Rounds']).plot(kind='bar')
plt.title('Número de Rondas de Financiación  por País', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Número de Rondas de Financiación', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
#(e)
pd.crosstab(df.Pais,df['year'],aggfunc="sum",values=df['Number of Lead Investments']).plot(kind='bar')
plt.title('Número de inversiones de clientes potenciales por País', fontsize=22)
plt.xlabel('País', fontsize=20)
plt.ylabel('Número de Inversiones de Clientes Potenciales', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

#%% 
#PUNTO 3
df1=dt[dt['Pais'] == 'Colombia']
pd.crosstab(df1['Pais'],df1['Investor Type']).plot(kind='bar')
plt.title('Investor Type per Country', fontsize=22)
plt.xlabel('Country', fontsize=20)
plt.ylabel('Frequency of Investor Type', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

#PUNTO3.1

dt_sorted_by_Investor_Type = dt.sort_values(["Investor Type"], ascending=True)
dt_sorted_by_Investor_Type = dt_sorted_by_Investor_Type[dt_sorted_by_Investor_Type.columns[:]][(dt_sorted_by_Investor_Type["Pais"] == 'Colombia')]
pd.crosstab(dt_sorted_by_Investor_Type.Pais,dt_sorted_by_Investor_Type ["Investor Type"],aggfunc="sum",values=dt_sorted_by_Investor_Type ['Last Funding Amount Currency (in USD)']).plot(kind='bar')
plt.title('Colombia Vs Investor Type-LFA', fontsize=22)
#%%
#PUNTO 4
df2=dt[dt['Pais'] == 'Colombia']
pd.crosstab(df2['Number of Exits'],df2['Organization Name']).plot(kind='bar')
plt.title('Capital Privado en Colombia', fontsize=22)
plt.xlabel('Number of Exits', fontsize=20)
plt.ylabel('Organization Name', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)