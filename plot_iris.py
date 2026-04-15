import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('iris.csv.xls')  
print(df.head())

sns.pairplot(df, hue='Name', diag_kind='kde')
plt.suptitle('Pairplot do dataset Iris', y=1.02)

plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Name',palette='Set2')
plt.title('Quantidade de amostras por espécie')
plt.xlabel('Espécie')
plt.ylabel('Contagem')

medias = df.groupby('Name').mean()
medias.plot(kind='bar', figsize=(8,5), colormap='viridis')
plt.title('Médias das características por espécie')
plt.xlabel('Espécie')
plt.ylabel('Valor médio (cm)')
plt.xticks(rotation=0)
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()
plt.show()
