# UNIVERSIDAD ICESI
# MAESTRIA EN IA APLICADA Y CIENCIA DE DATOS
# MATERIA: ANALISIS DE DAATOS 1
# PROFESOR: JOSE ARMANDO ORDOÑEZ CORDOBA
# ALUMNO: RUBEN DARIO SABOGAL URBANO
# FECHA: AGOSTO DE 2025 - CALI

# TALLER 2 - ANALISIS UNIVARIADO DE UNA VARIABLE EN UN CONJUNTO DE DATOS
# ARCHIVO: "autism_screening"



## Importar las librerias necesarias para visualizacion de datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer Datos usando read_csv()function
df=pd.read_csv('autism_screening.csv')

# Display el Data_Frame "autism_screning" o Prueba de Deteccion de Autismo(cribado) 
df

# Encontrar los tipos de Datos de las Variables en el DataFrame
# La propiedad .dtypes se utiliza para conocer los tipos de datos de las variables del conjunto de datos

df.dtypes

# Código Python que calcula las estadísticas descriptivas para la variable A9_Score

# GENERAR ESTADISTICAS DESCRIPTIVAS PARA LA VARIABLE ESCOGIDA

import pandas as pd

# Cargar datos
df = pd.read_csv('autism_screening.csv')

# Variable a analizar
var = 'A9_Score'

# Calcular estadísticas descriptivas
media = df[var].mean()
mediana = df[var].median()
moda = df[var].mode().iloc[0] if not df[var].mode().empty else None
std_dev = df[var].std()

# Detectar valores atípicos usando el método de IQR
Q1 = df[var].quantile(0.25)
Q3 = df[var].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]

# Resultados
resultados = {
    'media': media,
    'mediana': mediana,
    'moda': moda,
    'desviacion_estandar': std_dev,
    'limite_inferior_outliers': lower_bound,
    'limite_superior_outliers': upper_bound,
    'valores_atipicos': outliers.tolist()
}

print(resultados)

# Visualizar la Distribuccion de la Variable usando Histogramas, Boxplots, u otras graficas adecuadas

# HISTOGRAMA A9_SCORE

sns.displot(df['A9_Score'])
plt.xlabel('A9_Score')
plt.ylabel('Frecuencia')
plt.title('Histograma de A9_Score')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_csv('autism_screening.csv')

var = 'A9_Score'
data = df[var].dropna()

# Configuración general para los gráficos
plt.figure(figsize=(16, 20))

# 1 - Histograma
plt.subplot(5, 2, 1)
sns.histplot(data, kde=False, bins=10, color='skyblue')
plt.title('Histograma de A9_Score')
plt.xlabel('A9_Score')
plt.ylabel('Frecuencia')

# La función seaborn sns.distplot() también se puede utilizar para trazar un histograma.

sns.displot(df['A9_Score'])
plt.xlabel('A9_Score')
plt.ylabel('Frecuencia')
plt.title('Histograma de A9_Score')
plt.show()

# 2 - Boxplot
plt.subplot(5, 2, 2)
sns.boxplot(x=data, color='lightgreen')
plt.title('Boxplot de A9_Score')

# 3 - Diagrama de dispersión univariable
plt.subplot(5, 2, 3)
plt.scatter(range(len(data)), data, color='purple')
plt.title('Diagrama de dispersión univariable (A9_Score)')
plt.xlabel('Índice')
plt.ylabel('A9_Score')

# 4 - Traza de líneas (con marcadores)
plt.subplot(5, 2, 4)
plt.plot(range(len(data)), data, marker='o', color='orange')
plt.title('Traza de líneas con marcadores (A9_Score)')
plt.xlabel('Índice')
plt.ylabel('A9_Score')

# 5 - Parcela de franjas (Strip plot)
plt.subplot(5, 2, 5)
sns.stripplot(x=data, color='red', jitter=True, size=4 )
plt.title('Parcela de franjas (Strip plot) de A9_Score')

# 6 - Trama del enjambre (Swarm plot)
plt.subplot(5, 2, 6)
sns.swarmplot(x=data, color='blue', size=3)
plt.title('Trama del enjambre (Swarm plot) de A9_Score')

# 7 - Gráfica de densidad
plt.subplot(5, 2, 7)
sns.kdeplot(data, fill=True, color='darkgreen')
plt.title('Gráfica de densidad de A9_Score')

# 8 - Parcelas de alfombras (Heatmap de conteos)
counts = data.value_counts().sort_index()
plt.subplot(5, 2, 8)
sns.heatmap(counts.to_frame().T, cmap='YlGnBu', cbar=False, annot=True)
plt.title('Parcelas de alfombras (Heatmap) de A9_Score')
plt.yticks([])

# 9 - Diagrama de caja alternativo - BoxPlot
plt.subplot(5, 2, 9)
sns.boxplot(x=data, color='cyan')
plt.title('Diagrama de caja (Boxplot) alternativo de A9_Score')

# 10 - Parámetros de violín (Violin plot)
plt.subplot(5, 2, 10)
sns.violinplot(x=data, color='pink')
plt.title('Diagrama de violín (Violin plot) de A9_Score')
plt.tight_layout()
plt.show()


