print("\n" + "="*70)
print("ESTADÍSTICAS DESCRIPTIVAS Y GRÁFICOS CLAVE")
print("="*70)

# 1. Estadísticas descriptivas generales
print("\n" + "-"*50)
print("ESTADÍSTICAS DESCRIPTIVAS GENERALES")
print("-"*50)

# Estadísticas para variables numéricas
print("\nEstadísticas descriptivas para variables numéricas:")
print(df[numeric_cols].describe().T)

# Estadísticas para variables categóricas
print("\nEstadísticas para variables categóricas:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
    print(f"Porcentaje de valores faltantes: {df[col].isnull().mean()*100:.2f}%")

# 2. Análisis detallado de variables clave
print("\n" + "-"*50)
print("ANÁLISIS DETALLADO DE VARIABLES CLAVE")
print("-"*50)

# Variables clave identificadas (según el documento)
key_variables = ['A9_Score', 'A1_Score', 'A2_Score', 'age', 'gender', 'ethnicity', 'Class/ASD']

# Función para análisis detallado de variables clave
def detailed_analysis(variable, target='Class/ASD'):
    print(f"\n--- Análisis detallado de {variable} ---")
    
    # Si es variable numérica
    if variable in numeric_cols:
        # Estadísticas por clase
        stats_by_class = df.groupby(target)[variable].agg(['mean', 'median', 'std', 'count'])
        print(f"\nEstadísticas de {variable} por clase:")
        print(stats_by_class)
        
        # Test estadístico (t-test si 2 clases)
        if df[target].nunique() == 2:
            class_0 = df[df[target] == df[target].unique()[0]][variable].dropna()
            class_1 = df[df[target] == df[target].unique()[1]][variable].dropna()
            t_stat, p_value = stats.ttest_ind(class_0, class_1, nan_policy='omit')
            print(f"\nResultado del t-test: t={t_stat:.4f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print("  * Diferencia estadísticamente significativa entre clases (p < 0.05)")
            else:
                print("  * No hay diferencia estadísticamente significativa entre clases")
    
    # Si es variable categórica
    else:
        # Tabla de contingencia
        contingency_table = pd.crosstab(df[variable], df[target])
        print(f"\nTabla de contingencia ({variable} vs {target}):")
        print(contingency_table)
        
        # Test chi-cuadrado
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nResultado del test chi-cuadrado: chi2={chi2:.4f}, p-value={p:.4f}")
        if p < 0.05:
            print("  * Relación estadísticamente significativa entre variables (p < 0.05)")
        else:
            print("  * No hay relación estadísticamente significativa entre variables")
    
    # Visualización
    plt.figure(figsize=(14, 6))
    
    if variable in numeric_cols:
        # Boxplot por clase
        plt.subplot(1, 2, 1)
        sns.boxplot(x=target, y=variable, data=df)
        plt.title(f'{variable} por {target}')
        
        # Histograma superpuesto
        plt.subplot(1, 2, 2)
        for class_value in df[target].unique():
            subset = df[df[target] == class_value]
            sns.kdeplot(subset[variable], label=class_value, fill=True, alpha=0.5)
        plt.title(f'Distribución de {variable} por {target}')
        plt.legend(title=target)
    else:
        # Gráfico de barras apiladas
        plt.subplot(1, 2, 1)
        sns.countplot(x=variable, hue=target, data=df)
        plt.title(f'{variable} por {target}')
        plt.xticks(rotation=45)
        
        # Porcentajes
        plt.subplot(1, 2, 2)
        pd.crosstab(df[variable], df[target], normalize='index').plot(
            kind='bar', stacked=True, colormap='viridis', ax=plt.gca()
        )
        plt.title(f'Porcentaje de {target} por {variable}')
        plt.xticks(rotation=45)
        plt.legend(title=target)
    
    plt.tight_layout()
    plt.savefig(f'analisis_detallado_{variable}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Análisis detallado de {variable} generado: 'analisis_detallado_{variable}.png'")

# Aplicar análisis a variables clave
for var in key_variables:
    if var in df.columns and var != target:
        detailed_analysis(var)

# 3. Análisis específico de A9_Score (variable más relevante)
print("\n" + "-"*50)
print("ANÁLISIS ESPECÍFICO DE A9_SCORE (VARIABLE MÁS RELEVANTE)")
print("-"*50)

# Contar frecuencia de respuestas
a9_counts = df['A9_Score'].value_counts()
print("\nDistribución de respuestas para A9_Score:")
print(a9_counts)

# Análisis cruzado con la variable objetivo
a9_asd = pd.crosstab(df['A9_Score'], df['Class/ASD'], margins=True)
a9_asd_pct = pd.crosstab(df['A9_Score'], df['Class/ASD'], normalize='index') * 100

print("\nTabla cruzada A9_Score vs Class/ASD:")
print(a9_asd)
print("\nPorcentajes por A9_Score:")
print(a9_asd_pct)

# Calcular la probabilidad condicional
for a9_value in df['A9_Score'].unique():
    asd_yes_pct = a9_asd_pct.loc[a9_value, 'YES'] if 'YES' in a9_asd_pct.columns else 0
    print(f"- Probabilidad de TEA dado A9_Score={a9_value}: {asd_yes_pct:.2f}%")

# Visualización específica para A9_Score
plt.figure(figsize=(14, 6))

# Distribución de A9_Score
plt.subplot(1, 2, 1)
sns.countplot(x='A9_Score', data=df)
plt.title('Distribución de A9_Score')
plt.xlabel('A9_Score')
plt.ylabel('Frecuencia')

# Añadir porcentajes
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().annotate(f'{height/len(df)*100:.1f}%', 
                      (p.get_x() + p.get_width()/2., height),
                      ha='center', va='center', 
                      xytext=(0, 5), 
                      textcoords='offset points')

# Relación con Class/ASD
plt.subplot(1, 2, 2)
sns.countplot(x='A9_Score', hue='Class/ASD', data=df)
plt.title('Relación entre A9_Score y Class/ASD')
plt.xlabel('A9_Score')
plt.ylabel('Frecuencia')

# Añadir porcentajes por grupo
for container in plt.gca().containers:
    labels = [f"{h/total:.0%}" if (total := sum([h.get_height() for h in container])) > 0 else "" 
              for h in container]
    plt.bar_label(container, labels=labels, label_type='edge')

plt.tight_layout()
plt.savefig('analisis_a9_score.png', dpi=300, bbox_inches='tight')
print("✓ Análisis específico de A9_Score generado: 'analisis_a9_score.png'")