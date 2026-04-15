import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()
candidato_a=[30,32,40,37,38,50,48]
candidato_b=[22,25,30,31,33,37,40]

meses=[i for i in range (len(candidato_a))]

erro_a=[voto*0.01 for voto in candidato_a]
erro_b=[voto*0.01 for voto in candidato_b]

plt.grid(True)

ax.errorbar(meses,
            candidato_a,
            yerr=erro_a,
            label='candidato A',
            marker='o',
            capsize=5)
ax.errorbar(meses,
            candidato_b,
            yerr=erro_b,
            label='candidato B'
            ,marker='o',
            capsize=5)

plt.ylabel('Nº de votos')
plt.xlabel('Pesquisa')
plt.title('Votos para eleição')
plt.show()