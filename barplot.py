import matplotlib.pyplot as plt
import numpy as np

fig, ax=plt.subplots()

frutas=['maca','pera','uva']
quantidade=[10,20,50]
bar_label=['maca','pera','uva']
bar_colors=['tab:red', 'tab:orange','tab:green']
erro=[i*0.1 for i in quantidade]

ax.bar(frutas,quantidade,label=bar_label,color=bar_colors,edgecolor='black',linewidth=2,yerr=erro,capsize=12)
plt.xlabel('Variedade das frutas',fontsize=14)
plt.ylabel('Quantidade das frutas',fontsize=14)
plt.xticks(fontsize=14)
plt.title('Frutas disponiveis hoje na barraca')
plt.yticks(np.arange(0,50,5),fontsize=14)
plt.grid(True)

plt.show()