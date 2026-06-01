import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


furos_exemplo = np.arange(1, 107)
duracao_pct = (furos_exemplo / 106) * 100
erro_mse_exemplo = np.exp(duracao_pct / 25) * 0.005 + np.random.normal(0, 0.02, 106)
erro_mse_exemplo = np.clip(erro_mse_exemplo, 0.001, None)
threshold_exemplo = 0.12 # Exemplo de corte de 99.5%

fig, ax = plt.subplots(figsize=(6.5, 3.8))

# 1. Plotar os pontos de erro de cada furo
ax.scatter(duracao_pct, erro_mse_exemplo, color='#7209b7', alpha=0.7, edgecolors='none', label='Erro de Reconstrução ($MSE$)')
ax.plot(duracao_pct, erro_mse_exemplo, color='#7209b7', alpha=0.3, linestyle='-')

# 2. Plotar a linha horizontal do Threshold de 99.5%
ax.axhline(y=threshold_exemplo, color='red', linestyle='--', linewidth=1.2, label='Limiar Estatístico (Threshold 99.5%)')

# 3. Adicionar a linha vertical indicando onde a escuta humana mudou (Furo 85 ~ 80%)
ax.axvline(x=80, color='cyan', linestyle=':', linewidth=1.2)
ax.text(81, threshold_exemplo * 2.5, 'Percepção Humana\n(Fase Crítica)', color='darkcyan', fontsize=8, weight='bold')

# Estética
ax.set_title("Evolução do Erro de Reconstrução (LSTM-AE) - Broca 05", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Ciclo de Vida da Ferramenta (Duração %)", fontsize=10)
ax.set_ylabel("Erro de Reconstrução Médio ($MSE$)", fontsize=10)
ax.set_yscale('log') # Escala logarítmica destaca o início do desgaste
ax.legend(loc='upper left', fontsize=9, frameon=True)
plt.grid(True, which="both", linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig("tendencia_erro_mse_drill05.pdf", dpi=600, bbox_inches='tight')
plt.savefig("tendencia_erro_mse_drill05.png", dpi=300, bbox_inches='tight')
plt.show()