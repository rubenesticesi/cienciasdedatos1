import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración de visualización
plt.style.use('seaborn-whitegrid')
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# Cargar el dataset
df = pd.read_csv('autism_screening.csv', na_values='?')

# 1. Información general del dataset
print("="*70)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*70)
print(f"Forma del dataset: {df.shape}")
print(f"Número de variables: {df.shape[1]}")
print(f"Número de observaciones: {df.shape[0]}")
print("\nTipos de datos por variable:")
print(df.dtypes.value_counts())
print("\nInformación detallada del dataset:")
print(df.info())
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# 2. Análisis univariado para variables numéricas
print("\n" + "="*70)
print("ANÁLISIS UNIVARIADO DE VARIABLES NUMÉRICAS")
print("="*70)

# Identificar variables numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nVariables numéricas identificadas ({len(numeric_cols)}): {numeric_cols}")

# Función para análisis univariado de variables numéricas
def univariate_numeric_analysis(column):
    print(f"\n--- Análisis de {column} ---")
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(f"Media: {df[column].mean():.4f}")
    print(f"Mediana: {df[column].median():.4f}")
    print(f"Moda: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
    print(f"Desviación estándar: {df[column].std():.4f}")
    print(f"Rango intercuartílico (IQR): {df[column].quantile(0.75) - df[column].quantile(0.25):.4f}")
    print(f"Valor mínimo: {df[column].min():.4f}")
    print(f"Valor máximo: {df[column].max():.4f}")
    print(f"Rango: {df[column].max() - df[column].min():.4f}")
    print(f"Coeficiente de asimetría: {df[column].skew():.4f}")
    print(f"Coeficiente de curtosis: {df[column].kurtosis():.4f}")
    
    # Detección de outliers usando IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    print(f"\nNúmero de outliers detectados: {len(outliers)} ({len(outliers)/len(df)*100:.2f}% del total)")
    
    # Visualización
    plt.figure(figsize=(14, 10))
    
    # Histograma con curva de densidad
    plt.subplot(2, 2, 1)
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.axvline(df[column].mean(), color='r', linestyle='--', label='Media')
    plt.axvline(df[column].median(), color='g', linestyle='-', label='Mediana')
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.legend()
    
    # Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(f'Boxplot de {column}')
    plt.xlabel(column)
    
    # Q-Q plot para verificar normalidad
    plt.subplot(2, 2, 3)
    stats.probplot(df[column].dropna(), dist="norm", plot=plt)
    plt.title(f'Q-Q Plot de {column}')
    
    # Gráfico de densidad
    plt.subplot(2, 2, 4)
    sns.kdeplot(df[column], color='purple', fill=True)
    plt.title(f'Curva de Densidad de {column}')
    plt.xlabel(column)
    plt.ylabel('Densidad')
    
    plt.tight_layout()
    plt.savefig(f'distribucion_{column}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualización de {column} generada: 'distribucion_{column}.png'")

# Aplicar análisis a todas las variables numéricas
for col in numeric_cols:
    univariate_numeric_analysis(col)

# 3. Análisis univariado para variables categóricas
print("\n" + "="*70)
print("ANÁLISIS UNIVARIADO DE VARIABLES CATEGÓRICAS")
print("="*70)

# Identificar variables categóricas
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nVariables categóricas identificadas ({len(categorical_cols)}): {categorical_cols}")

# Función para análisis univariado de variables categóricas
def univariate_categorical_analysis(column):
    print(f"\n--- Análisis de {column} ---")
    
    # Estadísticas descriptivas
    print("\nConteo de categorías:")
    print(df[column].value_counts())
    
    print("\nPorcentaje por categoría:")
    print(df[column].value_counts(normalize=True) * 100)
    
    print(f"\nNúmero de categorías únicas: {df[column].nunique()}")
    print(f"Moda: {df[column].mode().iloc[0] if not df[column].mode().empty else 'N/A'}")
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    # Gráfico de barras
    plt.subplot(1, 2, 1)
    ax = sns.countplot(x=df[column], order=df[column].value_counts().index)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    
    # Añadir porcentajes en las barras
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height/len(df)*100:.1f}%', 
                   (p.get_x() + p.get_width()/2., height),
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    # Gráfico de pastel
    plt.subplot(1, 2, 2)
    counts = df[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Proporción de {column}')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'distribucion_{column}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualización de {column} generada: 'distribucion_{column}.png'")

# Aplicar análisis a todas las variables categóricas
for col in categorical_cols:
    univariate_categorical_analysis(col)