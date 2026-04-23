from matplotlib import pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#selecionar as features do csv original do iris e carregar data set
iris = pd.read_csv('iris.csv.xls')
nome_coluna = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Name']
iris.head()
features_coluna = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
X = iris[features_coluna]
Y = iris['Name']
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X,Y, test_size=0.3, random_state=1)
arvore = DecisionTreeClassifier()
arvore = arvore.fit(X_treino, Y_treino)
Y_pred = arvore.predict(X_teste)
print('Acuracia: ', metrics.accuracy_score(Y_teste, Y_pred)) 
plt.show()

