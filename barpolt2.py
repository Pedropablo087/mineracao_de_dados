import matplotlib.pyplot as plt
fig, ax = plt.subplots()
frutas = ['Maçã', 'Uva', 'Pera']
quantidade = [10, 23, 39]
bar_colors = ['tab:red', 'tab:orange', 'tab:green']
bar_label = ['Maçã', 'Uva', 'Pera']
erro = [1, 2, 3]
ax.bar(frutas, quantidade, label=bar_label, color=bar_colors, edgecolor='black', linewidth=3, yerr=erro, capsize=10)
#plt.errorbar(frutas, quantidade, yerr=erro, fmt='o', capsize=10)
plt.xlabel('Variedade de Frutas', fontsize=14)
plt.ylabel('Quantidade de Frutas', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.title('Banca do Seu Zé', fontsize=16)
plt.show()