import pandas as pd
import numpy as np #librería para análisis numérico
import matplotlib.pyplot as plt #Librería de visualización
import os #Librería para el sistema operativo
os.chdir('C:/Users/Personal/Documents')
cwd=os.getcwd()
excel_file = "Movies.xls"#
movies=pd.read_excel(excel_file)#guardar como data frame#
print(movies.head(15))#imprime los primeros 15 titulos
print(movies.tail(15))#imprime los ultimos 15 titulos
movies1=pd.read_excel(excel_file,sheet_name=0)#selecciona la primera hoja del excel movies 
movies2=pd.read_excel(excel_file,sheet_name=1)#selecciona la 2 hoja del excel movies 
movies3=pd.read_excel(excel_file,sheet_name=2)#selecciona la 3 hoja del excel movies 
movies = pd.concat([movies1, movies2, movies3])#une todas las hojas
movies.shape#imprime la dimension del df

#organizar el df

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending=False)#orden ascente del archivo#
print(sorted_by_gross.head(10))#imprime las primeras 15 del sorted_by gros , el cual ordena el arcivo de forma descendente

sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh")#impresion diagrama de barras horizontal de las primeras 10 peliculas
movies["IMDB Score"].plot(kind="hist")
movies.describe()#estadisticos
movies["IMDB Score"].mean()#media de la columna Score
movies["Gross Earnings"].mean()
movies["Gross Earnings"].corr (movies["Budget"])
movies["netEarnings"]=movies['Gross Earnings']-movies["Budget"]
print(movies["netEarnings"])
sorted_by_netEarnings=movies.sort_values(["netEarnings"],ascending=False)

#TAREA PARTE 
sorted_by_netEarnings["netEarnings"].head(10).plot(kind="barh")

movies1 = pd.read_excel(excel_file, sheet_name=0, index_col = 0)
movies2 = pd.read_excel(excel_file, sheet_name=1, index_col = 0)
movies3 = pd.read_excel(excel_file, sheet_name=2, index_col = 0)
movies = pd.concat([movies1,movies2,movies3])
#Coloca los espacios en blanco o fna como 0
movies.fillna(0, inplace=True)
sorted_by_netEarnings = movies.sort_values(['netEarnings'],ascending=False)
sorted_by_netEarnings["netEarnings"].head(10).plot(kind="barh",color="red")
#sorted_by_netEarnings["netEarnings"].head(10).plot(kind="barh")
plt.xlabel('Posición')
plt.ylabel('NetEarnings')
plt.title('Movies IMDB')
plt.show()


tabla1 = pd.pivot_table(movies, values=["netEarnings"], index=["Year"], aggfunc=np.sum)
tabla1.plot(kind="bar")
plt.xlabel('Year')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Year')
plt.show()




#tabla2 = pd.pivot_table(movies, values=["netEarnings"], index=["Country","Language"], aggfunc=np.sum)
tabla2 = pd.pivot_table(movies, 'netEarnings','Country','Language', aggfunc=np.sum )
tabla2.fillna(0, inplace=True)
tabla2.plot(kind="bar", stacked = 'True',alpha = 1.0 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country - Language')
plt.ylabel('netEarnings')
plt.title('Net_Earnings - Country - Language')
plt.show()



tabla3 = pd.pivot_table(movies, values=["Language"], index=["Country"], aggfunc="count")
tabla3.plot(kind="bar", stacked = 'True',alpha = 0.8 ,width = 1.0, figsize=(9,4))
plt.xlabel('Country')
plt.ylabel('Languaje')
plt.title('Country - Language')
plt.show()
