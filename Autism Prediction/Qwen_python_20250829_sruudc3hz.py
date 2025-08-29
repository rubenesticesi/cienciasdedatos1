print("\n" + "="*70)
print("ANÁLISIS BIVARIADO")
print("="*70)

# Variable objetivo
target = 'Class/ASD'
print(f"Variable objetivo identificada: {target}")

# 1. Relación entre variables numéricas y la variable objetivo
print("\n" + "-"*50)
print("RELACIÓN ENTRE VARIABLES NUMÉRICAS Y LA VARIABLE OBJETIVO")
print("-"*50)

def bivariate_numeric_vs_target(numeric_col, target_col):
    print(f"\n--- Análisis de {numeric_col} vs {target_col} ---")
    
    # Tabla de resumen por clase
    print("\nEstadísticas por clase:")
    print(df.groupby(target_col)[numeric_col].describe())
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    # Boxplot por clase
    plt.subplot(1, 2, 1)
    sns.boxplot(x=target_col, y=numeric_col, data=df)
    plt.title(f'Distribución de {numeric_col} por {target_col}')
    plt.xlabel(target_col)
    plt.ylabel(numeric_col)
    
    # Histograma superpuesto
    plt.subplot(1, 2, 2)
    for class_value in df[target_col].unique():
        subset = df[df[target_col] == class_value]
        sns.kdeplot(subset[numeric_col], label=class_value, fill=True, alpha=0.5)
    
    plt.title(f'Distribución de {numeric_col} por {target_col}')
    plt.xlabel(numeric_col)
    plt.ylabel('Densidad')
    plt.legend(title=target_col)
    
    plt.tight_layout()
    plt.savefig(f'relacion_{numeric_col}_vs_{target_col}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualización de relación generada: 'relacion_{numeric_col}_vs_{target_col}.png'")

# Aplicar análisis a todas las variables numéricas
for col in numeric_cols:
    if col != target:
        bivariate_numeric_vs_target(col, target)

# 2. Relación entre variables categóricas y la variable objetivo
print("\n" + "-"*50)
print("RELACIÓN ENTRE VARIABLES CATEGÓRICAS Y LA VARIABLE OBJETIVO")
print("-"*50)

def bivariate_categorical_vs_target(categorical_col, target_col):
    print(f"\n--- Análisis de {categorical_col} vs {target_col} ---")
    
    # Tabla de contingencia
    print("\nTabla de contingencia:")
    contingency_table = pd.crosstab(df[categorical_col], df[target_col])
    print(contingency_table)
    
    # Porcentajes por categoría
    print("\nPorcentajes por categoría:")
    print(pd.crosstab(df[categorical_col], df[target_col], normalize='index') * 100)
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    # Gráfico de barras apiladas
    plt.subplot(1, 2, 1)
    sns.countplot(x=categorical_col, hue=target_col, data=df)
    plt.title(f'Distribución de {categorical_col} por {target_col}')
    plt.xlabel(categorical_col)
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    
    # Gráfico de porcentajes
    plt.subplot(1, 2, 2)
    pd.crosstab(df[categorical_col], df[target_col], normalize='index').plot(
        kind='bar', stacked=True, colormap='viridis', ax=plt.gca()
    )
    plt.title(f'Porcentaje de {target_col} por {categorical_col}')
    plt.xlabel(categorical_col)
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45)
    plt.legend(title=target_col)
    
    plt.tight_layout()
    plt.savefig(f'relacion_{categorical_col}_vs_{target_col}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualización de relación generada: 'relacion_{categorical_col}_vs_{target_col}.png'")

# Aplicar análisis a todas las variables categóricas
for col in categorical_cols:
    if col != target:
        bivariate_categorical_vs_target(col, target)

# 3. Matriz de correlación para variables numéricas
print("\n" + "-"*50)
print("MATRIZ DE CORRELACIÓN")
print("-"*50)

plt.figure(figsize=(16, 12))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
print("✓ Matriz de correlación generada: 'matriz_correlacion.png'")

# 4. Análisis de importancia por información mutua
print("\n" + "-"*50)
print("ANÁLISIS DE IMPORTANCIA POR INFORMACIÓN MUTUA")
print("-"*50)

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Codificar la variable objetivo
le = LabelEncoder()
y = le.fit_transform(df[target])

# Preparar variables predictoras
X = df.drop(columns=[target])
X = pd.get_dummies(X, drop_first=True)

# Calcular información mutua
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, name="Información Mutua", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# Mostrar las 15 variables más importantes
print("\nLas 15 variables con mayor información mutua:")
print(mi_scores.head(15))

# Visualizar
plt.figure(figsize=(14, 10))
mi_scores.head(15).sort_values().plot(kind='barh', color='skyblue')
plt.title('Información Mutua de las Variables con respecto a la Clase ASD')
plt.xlabel('Información Mutua')
plt.ylabel('Variables')
plt.tight_layout()
plt.savefig('informacion_mutua.png', dpi=300, bbox_inches='tight')
print("✓ Análisis de información mutua generado: 'informacion_mutua.png'")