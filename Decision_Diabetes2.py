import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 1. Carregar e preparar os dados do diabetes
# ------------------------------------------------------------
df = pd.read_csv('diabetes.csv.xls')

cols_com_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_com_zero] = df[cols_com_zero].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_com_zero] = imputer.fit_transform(df[cols_com_zero])

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[feature_cols]
y = df['Outcome']

def calcular_metricas(y_true, y_pred):
    return {
        'Acurácia': metrics.accuracy_score(y_true, y_pred),
        'Precisão': metrics.precision_score(y_true, y_pred, zero_division=0),
        'Recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'F1-score': metrics.f1_score(y_true, y_pred, zero_division=0)
    }

def adicionar_rotulos(bars, ax):
    """Adiciona rótulos com valor percentual acima das barras."""
    for bar in bars:
        height = bar.get_height()
        # Formata como porcentagem com 1 casa decimal
        label = f'{height*100:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=8, color='black')

# ------------------------------------------------------------
# Experimento 1: critério
# ------------------------------------------------------------
criterions = ['gini', 'entropy', 'log_loss']
resultados_crit = {c: None for c in criterions}

for crit in criterions:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(criterion=crit, splitter='best', random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados_crit[crit] = calcular_metricas(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
metricas_nomes = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
x = np.arange(len(metricas_nomes))
largura = 0.25
for i, crit in enumerate(criterions):
    valores = [resultados_crit[crit][m] for m in metricas_nomes]
    bars = ax.bar(x + i*largura, valores, largura, label=crit.capitalize())
    adicionar_rotulos(bars, ax)

ax.set_ylabel('Valor')
ax.set_title('Desempenho por Critério (splitter=best, test_size=0.2)')
ax.set_xticks(x + largura)
ax.set_xticklabels(metricas_nomes)
ax.legend()
ax.set_ylim(0, 1.1)  # Aumentar um pouco para dar espaço aos rótulos
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Experimento 2: splitter
# ------------------------------------------------------------
splitters = ['best', 'random']
resultados_split = {s: None for s in splitters}

for split in splitters:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeClassifier(criterion='gini', splitter=split, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados_split[split] = calcular_metricas(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
largura = 0.35
for i, split in enumerate(splitters):
    valores = [resultados_split[split][m] for m in metricas_nomes]
    bars = ax.bar(x + i*largura, valores, largura, label=split.capitalize())
    adicionar_rotulos(bars, ax)

ax.set_ylabel('Valor')
ax.set_title('Desempenho por Splitter (criterion=gini, test_size=0.2)')
ax.set_xticks(x + largura/2)
ax.set_xticklabels(metricas_nomes)
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Experimento 3: tamanho do teste
# ------------------------------------------------------------
test_sizes = [0.2, 0.25, 0.3, 0.35, 0.4]
resultados_ts = {ts: None for ts in test_sizes}

for ts in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
    modelo = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados_ts[ts] = calcular_metricas(y_test, y_pred)

fig, ax = plt.subplots(figsize=(12, 6))
largura = 0.15
for i, ts in enumerate(test_sizes):
    valores = [resultados_ts[ts][m] for m in metricas_nomes]
    bars = ax.bar(x + i*largura, valores, largura, label=f'test={ts}')
    adicionar_rotulos(bars, ax)

ax.set_ylabel('Valor')
ax.set_title('Desempenho por Tamanho do Teste (criterion=gini, splitter=best)')
ax.set_xticks(x + largura * 2)
ax.set_xticklabels(metricas_nomes)
ax.legend(loc='lower left', ncol=2, fontsize=8)
ax.set_ylim(0, 1.2)  # Mais espaço para os rótulos
plt.tight_layout()
plt.show()