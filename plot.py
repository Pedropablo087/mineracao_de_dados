import matplotlib.pyplot as plt
#import random as rand

#aleatorio = rand.randint (0, 9)
x = [1, 2, 3, 4]
y = [2, 5, 7, 10]
yerro = [0.3, 0.5, 1, 0.1]
plt.errorbar(x, y, yerr=yerro, fmt='-o', ecolor='black', capsize=3)
plt.ylabel('números aleatórios')
plt.xlabel('ordem dos numeros')
plt.title('plot da aula de hoje')
#print(aleatorio)

plt.show()
