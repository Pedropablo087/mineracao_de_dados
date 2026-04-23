import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv.xls')  
print("Primeiras linhas do dataset:")
print(df.head())

sns.pairplot(df, hue='Name', diag_kind='kde', palette='Set2')
plt.suptitle('Pairplot do dataset Iris', y=1.02)
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Name', palette='Set2')
plt.title('Quantidade de amostras por espécie')
plt.xlabel('Espécie')
plt.ylabel('Contagem')
plt.tight_layout()
plt.show()

df_melted = df.melt(id_vars=['Name'], var_name='Característica', value_name='Valor (cm)')

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_melted,
    x='Característica',
    y='Valor (cm)',
    hue='Name',
    ci='sd',                
    capsize=0.15,           
    errwidth=1.5,           
    palette='viridis'
)
plt.title('Médias das características por espécie (com desvio padrão)')
plt.xlabel('Característica')
plt.ylabel('Valor médio (cm)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()