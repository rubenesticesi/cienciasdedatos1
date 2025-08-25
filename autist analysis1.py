# UNIVERSIDAD ICESI
# MAESTRIA EN IA APLICADA Y CIENCIA DE DATOS
# MATERIA: ANALISIS DE DAATOS 1
# PROFESOR: JOSE ARMANDO ORDOÑEZ CORDOBA
# ALUMNO: RUBEN DARIO SABOGAL URBANO
# FECHA: AGOSTO DE 2025 - CALI


# Paso 1: Importar las librerías necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configurar el estilo de las visualizaciones para un mejor aspecto
sns.set_theme(style="whitegrid")

# Paso 2: Cargar el conjunto de datos
# Se reemplazan los valores '?' por NaN (Not a Number) para que Pandas los reconozca como datos faltantes.
df = pd.read_csv('autism_screening.csv', na_values='?')

# Paso 3: Inspección inicial de los datos
print("--- Información general del dataset ---")
df.info()
print("\n--- Primeras 5 filas del dataset ---")
print(df.head())


# Paso 4: Análisis Univariado de la variable 'A9_Score'
print("\n--- Análisis de la variable 'A9_Score' ---")

# Contar la frecuencia de cada respuesta (0 para 'No', 1 para 'Sí')
a9_counts = df['A9_Score'].value_counts()
print("\nDistribución de respuestas para A9_Score:")
print(a9_counts)

# Crear una visualización para la distribución de A9_Score
plt.figure(figsize=(10, 6))
sns.countplot(x='A9_Score', data=df, palette='viridis')
plt.title('Distribución de Respuestas para la Pregunta A9', fontsize=16)
plt.xlabel('Puntuación en A9 (0 = No, 1 = Sí)', fontsize=12)
plt.ylabel('Cantidad de Encuestados', fontsize=12)
plt.xticks([0, 1], ['No', 'Sí'])
plt.show()


# Paso 5: Relación entre 'A9_Score' y la variable objetivo ('Class/ASD')
# Este es un análisis bivariado, pero es crucial para entender la importancia de A9_Score.
print("\n--- Relación entre A9_Score y el Diagnóstico de TEA (Class/ASD) ---")

# Crear una tabla de contingencia para ver la relación
crosstab_a9 = pd.crosstab(df['A9_Score'], df['Class/ASD'])
print("\nTabla de Contingencia:")
print(crosstab_a9)

# Crear una visualización para la tasa de diagnóstico por respuesta en A9
plt.figure(figsize=(10, 6))
sns.countplot(x='A9_Score', hue='Class/ASD', data=df, palette='plasma')
plt.title('Diagnóstico de TEA según la Respuesta en A9', fontsize=16)
plt.xlabel('Puntuación en A9 (0 = No, 1 = Sí)', fontsize=12)
plt.ylabel('Cantidad de Encuestados', fontsize=12)
plt.xticks([0, 1], ['No', 'Sí'])
plt.legend(title='Clasificación TEA')
plt.show()