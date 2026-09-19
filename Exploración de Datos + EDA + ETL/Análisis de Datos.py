#!/usr/bin/env python
# coding: utf-8

# ## 1.	Titulo: Modelo de predicción de ocurrencia de accidentes de tránsito en Risaralda mediante datos históricos.

# ## 2.	Introducción:

# El presente trabajo se enfoca en la utilización de técnicas avanzadas de analítica de datos para abordar una problemática crítica y de gran relevancia social como lo son los accidentes de tránsito en la ciudad de Pereira, capital del departamento de Risaralda. Esta investigación surge de la necesidad de comprender mejor los factores que contribuyen a estos incidentes, que no solo representan un desafío para la seguridad vial sino que también tienen un profundo impacto humano y económico en la comunidad.
# 
# Con el objetivo de contribuir a la reducción de las muertes y lesiones graves derivadas de accidentes de tránsito, este estudio se propone identificar los puntos críticos donde estos incidentes son más frecuentes, así como los factores de riesgo que aumentan su probabilidad. Esta identificación se realizará mediante datos históricos  provenientes del Instituto Nacional de Medicina Legal y Ciencias Forenses.
# 

# ## 3.	Objetivos:

# •	Identificación de puntos críticos de accidentes de transito
# 
# •	Identificación de factores de influencia en accidentes de tránsito.
# 
# •	Proponer estrategias para la prevención de accidentes de tránsito en Risaralda.
# 

# ## 4.	Presnetacion detallada de la base de datos:

# Se cuenta con la base de datos de tipo CSV aportada por medicina legal con datos de accidentes fatales desde el año 2007 al 2021 (4.500 registros aprox.) en la cual contamos con datos asociados a la víctima de tipo cualitativas como sexo, estado civil, nivel académico, ocupación y cuantitativas como edad; adicionalmente contamos con datos cualitativos asociados a los hechos del accidente como fecha, hora, municipio, ubicación geográfica, tipo de la vía, case de accidente, tipo de vehículo, condición de lugar, entre otros.
# En algunos casos no se cuenta con datos como la hora de los hechos, estado de la via, tipo de vehículo (vehículo fantasma) o condición de la vía debido a que no hay testigos que pudieran aportar dicha información o también se pueden encontrar datos atípicos que podrían ser generados en la digitación del informe pericial como por ejemplo edades de mas de 100 años.
# 

# ## 5.	Metodología: 

# Se pretende hacer un análisis de información identificando factores de influencia en accidentes de transito, identificar si hay correlaciones entre datos al igual que identificar las ubicaciones mas frecuentes y sus posibles causas como puede ser hora, estado de la vía o tipo de vehículo.

# ## 6.	Resultados esperados: 

# Se espera poder proponer alertas para evitar accidentes fatales en puntos críticos o en condiciones específicas como climáticas, horas o días en particular.

# ## 7.	Descripción de la base de datos a trabajar

# La base de datos a trabajar cuenta con datos relacionados en accidentes de tránsito fatales en los departamentos de Risaralda, Caldas, Quindío y parte de norte del Valle (Cartago) ocurridos durante el periodo comprendido desde el año 2007 y 2021.
# 
# Es una base de datos estructurada y cuenta con datos etiquetados en formato .xlsx 
# 
# A continuación, se describen la tipología de algunas variables de interés:
# 
# •	Datos Cuantitativos (Discretos):  Edad calculada en años 
# 
# •	Datos Cualitativos (Ordinal): Departamento asociado a la necropsia; Municipio asociado a la necropsia; Sexo; Estado civil; nivel académico; Ocupación; Pertenencia grupal; Fecha de los hechos; Hora de los hechos; Zona del hecho (Latitud-Longitud); escenario de los hechos; causa de muerte definitiva; diagnostico topográfico;	mecanismo de muerte; clase de accidente de transporte; condición de la víctima; vehículo	tipo de servicio del vehículo; objeto de colisión; servicio objeto de colisión; vehículo fantasma; condición del lugar del at; estado de la vía.
# 

# ## 8.	Calidad de los datos

# Al revisar la calidad de los datos en la base de datos se identifica que en algunos casos se registraron en la etiqueta sexo “Femenino” y en otras solo “F” al igual que “Masculino” y “M” para lo cual se corrigieron las datos quedando solo valores “F” o “M” De igual manera ocurre en la etiqueta tipo de vehículo, donde se encuentran datos registrados como “Buseta” y “BUS” donde se decide unificarlos todos a “BUS”, lo cual se hace en Excel antes de convertir los datos a csv. 

# ## 9.	Datos Faltantes

# Se identificaron datos faltantes en la etiqueta fecha y hora de los hechos para lo cual se decide no eliminarlos dado que conocemos el año en que se realizo y los demás datos relevantes.

# ## 10.	Importar librerías.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import HeatMap
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

file_path = '/Users/andresbermudezrosero/Documents/Maestria Ing Electrica/I Semestre/Introduccion Analisis de datos/Proyecto/Datos  INML.csv'

df = pd.read_csv(file_path, sep=';', on_bad_lines='skip')
print(df.columns.tolist())


# El código importa varias bibliotecas de Python comunes para el análisis de datos y la visualización. Esto incluye:
# •	pandas y numpy para la manipulación y análisis de datos.
# 
# •	matplotlib.pyplot y seaborn para la visualización de datos.
# 
# •	datetime para trabajar con fechas y horas.
# 
# •	sklearn.preprocessing.LabelEncoder para la codificación de etiquetas (útil en el preprocesamiento de datos).
# 
# •	folium y sus plugins para crear mapas interactivos.
# 
# •	re para expresiones regulares.
# 
# •	geopandas para trabajar con datos geoespaciales.
# 
# •	osmnx para obtener datos de OpenStreetMap y trabajar con datos de redes urbanas.
# 
# •	shapely.geometry para manipulación y análisis de figuras geométricas.
# 

# ## Revisión de valores nulos

# In[2]:


sns.heatmap(df.isnull(), cbar=False)


# ## 11. Transformacion de columnas

# 11.1.	Conversión a numérico: El primer bucle Utiliza pd.to_numeric() para convertir los valores de estas columnas a números, lo que es útil si los datos se importaron como cadenas de texto o si hay valores no numéricos que deben ser manejados. El argumento.
# 
# 11.2.	Conversión a datetime: El segundo bucle for es para convertir los valores de estas columnas a objetos de fecha y hora de pandas.
# 
# 11.3.	Extracción del año: En la línea siguiente, se crea una nueva columna llamada 'AÑO DE RADICADO' extrayendo el año de la columna 'AÑO DE RADICADO' ya existente. Esto se hace con el atributo .dt.year que es parte de las funcionalidades de pandas para trabajar con fechas y horas.
# 
# 11.4.	Eliminación de columnas: el código elimina las siguientes columnas del DataFrame que no se necesitan para el análisis. 'NUMERO DE RADICADO', 'UNIDAD DE EDAD', 'CAUSA DE MUERTE DEFINITIVA', 'MECANISMO DE MUERTE', 'AÑO DE RADICADO', 'HORA DE LOS HECHOS', 'DIAGNOSTICO TOPOGRAFICO', 'DEPARTAMENTO ASOCIADO A LA NECROPSIA', y elimina las filas donde la fecha de los hechos es NaN.
# 

# In[3]:


for i in ['EDAD CALCULADA EN AÑOS','Latitude','Longitude']:
    df[i] = pd.to_numeric(df[i], errors='coerce')


# In[4]:


for i in ['FECHA DE LOS HECHOS', 'AÑO DE RADICADO']:
    df[i]=pd.to_datetime(df[i], errors='coerce')


# In[5]:


df['AÑO DE RADICADO']= df['AÑO DE RADICADO'].dt.year


# In[6]:


df['HORA_DE_LOS_HECHOS'] = pd.to_datetime(df['HORA DE LOS HECHOS'], errors='coerce')
df['hora'] = df['HORA_DE_LOS_HECHOS'].dt.hour


# In[7]:


df.drop(['NUMERO DE RADICADO', 'UNIDAD DE EDAD', 'CAUSA DE MUERTE DEFINITIVA', 'MCANISMO DE MUERTE',
'AÑO DE RADICADO', 'DIAGNOSTICO TOPOGRAFICO', 'DEPARTAMENTO ASOCIADO A LA NECROPSIA'], axis=1, inplace=True)
df.dropna(subset='FECHA DE LOS HECHOS', inplace=True)


# In[8]:


df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')


# ## 12. Descripción de variables 

# ## Numéricas

# El método .describe() es para obtener una comprensión de las características de las variables numéricas en un conjunto de datos, permitiendo identificar tendencias, anomalías, y la distribución general de los datos.
# 
# count: Muestra el número de observaciones no nulas para cada variable.
# mean: El valor medio de cada variable.
# std: La desviación estándar de cada variable, que es una medida de la cantidad de variación o dispersión de un conjunto de valores.
# min: El valor mínimo en cada variable.
# 25%: El percentil 25 o cuartil inferior, que indica que el 25% de los datos son menores que este valor.
# 50%: El percentil 50 o mediana, que es el valor medio de los datos.
# 75%: El percentil 75 o cuartil superior, que indica que el 75% de los datos son menores que este valor.
# max: El valor máximo en cada variable.
# 

# In[9]:


df.describe().T


# EDAD CALCULADA EN AÑOS: Representa la edad calculada de los sujetos o entidades dentro del conjunto de datos. La edad promedio es de aproximadamente 45 años, con una edad mínima de 0 y una máxima de 99 años.

# ## Categóricas

# In[10]:


df.describe(include=['O']).T


# Variables como 'VEHICULO', 'TIPO DE SERVICIO DEL VEHICULO', 'OBJETO DE COLISION', 'SERVICIO OBJETO DE COLISION', 'VEHICULO FANTASMA', 'CONDICION DEL LUGAR DEL AT', y 'ESTADO DE LA VIA AT' ofrecen información adicional sobre las circunstancias y los detalles de los incidentes reportados, como el tipo de vehículo involucrado, la condición de la carretera, entre otros.
# 
# En general, esta tabla proporciona una visión general valiosa de las características categóricas del conjunto de datos, identificando las categorías más comunes y la diversidad de los datos en cada campo. Esto es esencial para comprender la distribución de las variables categóricas y puede informar decisiones posteriores sobre la codificación de datos para el análisis y la modelización.
# 

# In[11]:


df.info()


# ## 13. Gráfico de densidad

# In[12]:


df_subset = df[['EDAD CALCULADA EN AÑOS', 'VEHICULO']]

pairplot_ = sns.pairplot(df_subset, hue='VEHICULO', height=5)  
plt.legend(loc='best', title='VEHICULO', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

plt.savefig(fname='Pairplot', dpi=300, format='png', bbox_inches='tight') 
plt.show()


# Eje Y: Representa la densidad estimada para cada valor de edad. Los picos más altos indican las edades más comunes dentro de las categorías de vehículos.
# Curvas de Densidad: Cada curva muestra la distribución de la edad para una categoría diferente de vehículo. Las áreas bajo las curvas están sombreadas, lo que indica la probabilidad de la edad dentro de esa categoría de vehículo. Un pico más alto y estrecho sugiere que la mayoría de las edades están concentradas alrededor de un valor central, mientras que una curva más plana indica una mayor dispersión de edades.
# 
# Tipos de Vehículos
# No Aplica: no es aplicable o no se reportó.
# Camioneta, Taxi, Automóvil, Moto Carro, Camión Furgón: Estos tipos de vehículos muestran distribuciones de edad similares con picos alrededor de la mediana de edad, lo que sugiere que son utilizados por un rango de edad más amplio.
# Bicicleta: La curva de bicicletas parece tener un pico más bajo, indicando una distribución más uniforme de edades entre los ciclistas.
# Bus, Motocicleta, Vehículo Articulado, Volqueta, Ambulancia, Microbús, Maquinaria Industrial, Tractocamión: Las curvas para estos vehículos tienen picos y formas variadas, lo que indica diferentes patrones de uso de edad para cada tipo de vehículo.
# 
# La superposición de las curvas de densidad de diferentes tipos de vehículos proporciona una indicación visual de cómo las víctimas se distribuye entre los diferentes tipos de vehículos implicados en los hechos reportados. Por ejemplo, si las curvas para "Automóvil" y "Motocicleta" son muy diferentes, esto podría sugerir diferencias demográficas entre los conductores de estos dos tipos de vehículos.
# 
# Es importante mencionar que la densidad de kernel es una estimación y la altura de las curvas no debe interpretarse como una probabilidad absoluta. Además, el gráfico utiliza un eje Y que está normalizado.
# 

# ## 14.	Gráficos boxplot

# ## Boxplot Sexo - Edad 

# In[13]:


sns.boxplot(x=df['SEXO'],y=df['EDAD CALCULADA EN AÑOS'])


# Eje X vemos las dos categorías de sexo. Y en el eje Y ("EDAD CALCULADA EN AÑOS") nos muestra la edad de las victimas. 
# Boxplot Azul (Masculino): Línea está alrededor de los 40 años, lo que significa que la mitad de la población masculina es más joven que 40 y la otra mitad es mayor.
# Boxplot Naranja (Femenino): Línea central al igual que en el boxplot azul, parece estar alrededor de los 40 años.
# 
# En general, el boxplot sugiere que no hay una diferencia significativa en la distribución de la edad entre las poblaciones masculina y femenina, ya que tanto la mediana como el rango intercuartílico son similares. Esto podría implicar que, para la variable de interés o el contexto de este conjunto de datos, la edad se distribuye de manera bastante uniforme entre hombres y mujeres.
# 

# ## Boxplot: Tipo de servicio de vehículo - Edad

# In[14]:


plt.figure(figsize=[15,5])
sns.boxplot(x=df['TIPO DE SERVICIO DEL VEHICULO'],y=df['EDAD CALCULADA EN AÑOS'])


# Los boxplots sugieren que hay una variabilidad en la edad asociada con diferentes tipos de servicios de vehículos, pero con medianas bastante consistentes cerca de los 40 años. Los valores atípicos pueden indicar excepciones individuales o errores en los datos. La categoría "Sin Información" parece tener la mayor variabilidad en la edad, mientras que las categorías "Oficial" y "Transporte Masivo" muestran la menor variabilidad, lo que podría reflejar las políticas de contratación o las demografías de uso específicas para estos tipos de servicios.

# ## Boxplot: Vehículo - Edad

# In[15]:


plt.figure(figsize=[15,5])
sns.boxplot(x=df['VEHICULO'],y=df['EDAD CALCULADA EN AÑOS'])


# La edad mediana para la mayoría de las categorías de vehículos es consistente, agrupándose alrededor de los 40 años. Las medianas consistentes sugieren que no hay una diferencia significativa en la edad promedio de las personas asociadas con cada tipo de vehículo. Sin embargo, algunos tipos de vehículos como la bicicleta parecen tener una población más joven en comparación con otros tipos de vehículos.

# ## 15 Gráfico de barras 

# ## Tipo de vehículo.

# In[16]:


plt.figure(figsize=[18,5])
df.groupby('VEHICULO')['VEHICULO'].count().plot(kind='bar')


# El gráfico de barras representa la cantidad de muertes por accidente de tránsito clasificadas por tipo de vehículo involucrado. Este es un análisis esencial para la seguridad vial y para entender qué tipos de vehículos están más frecuentemente asociados con accidentes fatales.
# 
# "NO APLICA": Esta categoría tiene la mayor cantidad de registros. Esto podría ser debido a registros donde el tipo de vehículo no fue especificado o no es relevante para el accidente. La alta frecuencia en esta categoría podría ser un indicador de que la recopilación de datos necesita mejoras para asegurar que se capture la información del tipo de vehículo en cada accidente.
# "TAXI" y "AUTOMÓVIL": Estos tipos de vehículos tienen una cantidad significativa de muertes asociadas, lo que puede reflejar su prevalencia en el tráfico o posibles problemas de seguridad específicos para estos tipos de vehículos.
# "BICICLETA", "BUS", "CAMIONETA": Estas categorías tienen menos muertes asociadas en comparación con "TAXI" y "AUTOMÓVIL", pero aún así representan una cantidad considerable. Esto podría reflejar tanto la frecuencia de uso como los riesgos inherentes asociados con estos tipos de vehículos.
# "MOTOCICLETA" y "MOTOCARRO": La frecuencia de muertes para estos vehículos es notablemente alta. Las motocicletas suelen estar involucradas en accidentes de tránsito con consecuencias graves debido a la menor protección que ofrecen en comparación con los vehículos cerrados.
# "TRACTOCAMIÓN", "AMBULANCIA", "MAQUINARIA INDUSTRIAL": Estas categorías tienen menos muertes asociadas en comparación con otras categorías, lo que podría ser debido a su uso menos frecuente en áreas urbanas o a regulaciones más estrictas sobre su operación.
# 
# Esta herramienta es  útil para los responsables de la formulación de políticas y las autoridades de tráfico, ya que destaca los tipos de vehículos que podrían necesitar atención adicional en términos de medidas de seguridad, campañas de concienciación y regulaciones de tráfico. Por ejemplo, si los "TAXIS" y "AUTOMÓVILES" son responsables de un número significativo de muertes, las autoridades podrían considerar campañas de seguridad específicas para conductores y pasajeros de estos vehículos, así como inspecciones de seguridad más rigurosas.
# 

# ## Recurrencia por horas del dia

# In[17]:


df_filtrado = df[df['hora'] != 0]

conteo_por_hora = df_filtrado.groupby('hora').size()

conteo_por_hora.plot(kind='bar')

plt.xlabel('Hora del día')
plt.ylabel('Número de accidentes')
plt.title('Accidentes de tránsito fatales por hora en Pereira')

plt.show()


# Hay un aumento notable en la cantidad de accidentes en las horas de la tarde y primeras horas de la noche, con un pico particularmente alto alrededor de las 18:00 (6 PM). Esto podría estar relacionado con la mayor cantidad de tráfico debido al horario de salida del trabajo y posiblemente también a una visibilidad reducida al anochecer. Se observa otro pico más pequeño en las horas de la mañana, alrededor de las 7:00 a 8:00 AM, lo cual puede estar relacionado con el tráfico de la hora pico matutina. Las horas de la madrugada muestran menos accidentes, lo cual podría estar relacionado con las carreteras más despejadas durante ese período. Sin embargo, los accidentes que ocurren en este tiempo pueden ser más graves debido a factores como la fatiga o la conducción bajo la influencia del alcohol.
# Después de la medianoche, el número de accidentes disminuye, pero hay un aumento nuevamente alrededor de las 23:00 (11 PM).
# Este tipo de información es crucial para las autoridades encargadas de la planificación de la seguridad vial, ya que indica los períodos de tiempo en los que deben aumentar la vigilancia, mejorar la iluminación de las calles o realizar campañas de concienciación sobre seguridad vial. Además, puede indicar la necesidad de revisar las condiciones de tráfico o los patrones de conducción durante las horas pico de accidentes.
# 

# ## 16. Matriz de correlacion

# In[18]:


categorical_cols = ['SEXO', 'ESTADO CIVIL','hora', 'NIVEL ACADEMICO', 'OCUPACION', 'FECHA DE LOS HECHOS', 'CLASE DE ACCIDENTE DE TRANSPORTE', 'CONDICION DE LA VICTIMA', 'VEHICULO', 'TIPO DE SERVICIO DEL VEHICULO', 'OBJETO DE COLISION', 'SERVICIO OBJETO DE COLISION', 'VEHICULO FANTASMA', 'CONDICION DEL LUGAR DEL AT', 'ESTADO DE LA VIA AT'] 
numerical_cols = ['EDAD CALCULADA EN AÑOS'] 


label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))


correlation_matrix = df[categorical_cols + numerical_cols].corr()


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()


# "SEXO" y "CONDICION DE LA VICTIMA": Hay una correlación negativa moderada de -0.27. Esto podría sugerir que hay una relación entre el sexo de la víctima y su condición en el accidente (por ejemplo, si eran el conductor, pasajero, etc.).
# 
# "NIVEL ACADEMICO" y "OCUPACION": Existe una correlación negativa notable de -0.19, lo que podría indicar una relación entre el nivel educativo y la ocupación de las víctimas.
# 
# "NIVEL ACADEMICO" y "EDAD CALCULADA EN AÑOS": Hay una correlación positiva fuerte de 0.73. Esto sugiere que hay una relación significativa entre la edad y el nivel educativo, lo que podría reflejar tendencias demográficas o socioeconómicas en el conjunto de datos.
# 
# "VEHICULO FANTASMA" y "SEXO": Existe una correlación positiva de 0.73, lo que podría indicar una relación entre el sexo de la víctima y la incidencia de accidentes donde el otro vehículo no se detuvo o no fue identificado.
# 
# "TIPO DE SERVICIO DEL VEHICULO" y "CONDICION DE LA VICTIMA": La correlación positiva de 0.39 indica una relación entre el tipo de servicio del vehículo y la condición de la víctima en el accidente.
# 

# ## 17. Base de datos - PEREIRA

# Dado que la base de datos es muy grande para algunos análisis se decide crear una base de datos alterna con datos solo de la ciudad de Pereira.

# In[19]:


bd_Pereira = df[df['MUNICIPIO DEL HECHO'] == 'PEREIRA']
bd_Pereira.describe().T


# In[20]:


bd_Pereira.describe(include=['O']).T


# In[21]:


bd_Pereira.head()


# Estos datos proporcionan una visión general de las circunstancias y características de los accidentes de tránsito fatales en la ciudad de Pereira.

# In[22]:


ox.config(use_cache=True, log_console=True)

# Obtener la geometría de la ciudad de Pereira
place_name = "Pereira, Risaralda, Colombia"
graph = ox.graph_from_place(place_name, network_type='drive')
fig, ax = ox.plot_graph(graph)

plt.tight_layout()
plt.show()


# ## 18. Mapa de calor - Pereira

# In[23]:


ox.settings.use_cache = True
ox.settings.log_console = True

place_name = "Pereira, Risaralda, Colombia"
graph = ox.graph_from_place(place_name, network_type='drive')

nodes, edges = ox.graph_to_gdfs(graph)

bd_Pereira['Longitude'] = bd_Pereira['Longitude'].apply(lambda x: x if -180 <= x <= 180 else x/10)
bd_Pereira['Latitude'] = bd_Pereira['Latitude'].apply(lambda x: x if -90 <= x <= 90 else x/10)

bd_Pereira.dropna(subset=['Latitude', 'Longitude'], inplace=True)

accident_counts = bd_Pereira.groupby(['Latitude', 'Longitude']).size().reset_index(name='counts')

gdf_accident_counts = gpd.GeoDataFrame(
    accident_counts, 
    geometry=gpd.points_from_xy(accident_counts.Longitude, accident_counts.Latitude)
)

gdf_accident_counts.set_crs(edges.crs, inplace=True)


fig, ax = plt.subplots(figsize=(12, 12))
edges.plot(ax=ax, linewidth=1, edgecolor='grey')

gdf_accident_counts.plot(ax=ax, color='red', markersize=gdf_accident_counts['counts'] * 10)

ax.set_xlim(edges.total_bounds[[0, 2]])
ax.set_ylim(edges.total_bounds[[1, 3]])

ax.set_aspect('equal', 'box')

plt.tight_layout()
plt.show()


# La distribución de los puntos a lo largo del mapa muestra cómo los accidentes de tránsito fatales están dispersos por toda la ciudad. Se puede observar que hay ciertas áreas con una mayor concentración de puntos, lo que indica una mayor frecuencia de accidentes.
# 
# Las áreas con los puntos más grandes, como en el centro del mapa, sugieren puntos críticos donde los accidentes son más comunes. Estas áreas podrían ser intersecciones peligrosas, zonas con alta densidad de tráfico o áreas con infraestructura vial inadecuada.
# 
# Para las autoridades de la ciudad y los planificadores urbanos, este tipo de visualización es extremadamente útil porque identifica las áreas que podrían beneficiarse de medidas de seguridad mejoradas, como señalización adicional, cambios en los límites de velocidad, mejor iluminación, o campañas de educación para conductores y peatones. Las áreas de alta concentración de accidentes podrían ser el enfoque de estudios más detallados para comprender las causas subyacentes y para desarrollar intervenciones específicas destinadas a reducir la cantidad y severidad de los accidentes de tránsito.
# 

# ## 19.	Conclusiones y recomendaciones.

# Se identificaron correlaciones clave que pueden ser utilizadas para prevenir accidentes, como la relación entre el sexo de la víctima y la condición en el accidente, y la relación entre el nivel educativo y la ocupación.
# 
# Los mapas de calor muestran áreas de alta frecuencia de accidentes, lo que puede guiar a las autoridades para mejorar la seguridad vial en estas zonas.
# 
# Los resultados sugieren la necesidad de campañas de seguridad específicas para conductores y pasajeros de vehículos con alta incidencia en accidentes, así como inspecciones de seguridad más rigurosas.
# 
# Desarrollar modelos predictivos más sofisticados utilizando técnicas avanzadas de análisis de datos para mejorar la precisión en la predicción de accidentes.
# 
