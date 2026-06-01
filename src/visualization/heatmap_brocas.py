import os
import re  # ADICIONADO: Correção do NameError
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOCALIZAR E CARREGAR O ARQUIVO EXPORTADO PELO SEU SCRIPT
nome_csv = "resultados_xgboost_hibrido.csv"
try:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_csv)
except NameError:
    csv_path = nome_csv

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"O arquivo {csv_path} não foi localizado. Rode o script do XGBoost primeiro!")

df = pd.read_csv(csv_path)

# 2. PIVOTAR OS DADOS PARA O FORMATO DA MATRIZ (Linhas = Brocas, Colunas = Furos)
df['drill_short'] = df['drill'].apply(lambda x: re.search(r"drill_4mm_(\d+)", x.lower()).group(0) if re.search(r"drill_4mm_(\d+)", x.lower()) else x)

heatmap_truth = df.pivot(index="drill_short", columns="hole", values="label")
heatmap_score = df.pivot(index="drill_short", columns="hole", values="score")
heatmap_alert = df.pivot(index="drill_short", columns="hole", values="prediction")

# Ordenar as brocas em ordem alfabética/numérica crescente
heatmap_truth = heatmap_truth.sort_index(ascending=True)
heatmap_score = heatmap_score.sort_index(ascending=True)
heatmap_alert = heatmap_alert.sort_index(ascending=True)

# ==============================================================================
# 3. CONFIGURAR E GERAR OS TRÊS HEATMAPS LADO A LADO (PADRÃO ACADÊMICO)
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9.5,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)

kwargs_base = {
    "xticklabels": 5,
    "yticklabels": True,
    "linewidths": 0.1,
    "linecolor": "#f0f0f0",
    "rasterized": True
}

# --- Heatmap 1: Ground Truth (Verdade-Terreno de 20%) ---
sns.heatmap(
    heatmap_truth,
    ax=axes[0],
    cmap="Blues",
    cbar_kws={'label': 'Fase Crítica Definida (1 = Últimos 20%)', 'orientation': 'horizontal', 'pad': 0.15, 'shrink': 0.8},
    **kwargs_base
)
axes[0].set_title("1. Ground Truth (Alvo Ideal)", fontsize=11, weight='bold', pad=8)
axes[0].set_ylabel("Identificação da Broca (Drill)", fontsize=10)
axes[0].set_xlabel("Sequência de Furos (Hole)", fontsize=10)

# --- Heatmap 2: XGBoost Score (Probabilidade Contínua do Modelo) ---
sns.heatmap(
    heatmap_score,
    ax=axes[1],
    cmap="YlOrRd",
    vmin=0, vmax=1,
    cbar_kws={'label': 'Confidence / Probability Score', 'orientation': 'horizontal', 'pad': 0.15, 'shrink': 0.8},
    **kwargs_base
)
axes[1].set_title("2. XGBoost Score (Evolução Contínua)", fontsize=11, weight='bold', pad=8)
axes[1].set_xlabel("Sequência de Furos (Hole)", fontsize=10)
axes[1].set_ylabel("")

# --- Heatmap 3: Alertas Finais (Sistema com Persistência Janela=2) ---
sns.heatmap(
    heatmap_alert,
    ax=axes[2],
    cmap="Oranges",
    cbar_kws={'label': 'Disparo do Alarme (1 = Manutenção)', 'orientation': 'horizontal', 'pad': 0.15, 'shrink': 0.8},
    **kwargs_base
)
axes[2].set_title("3. Alertas Finais (97.5% + Janela=2)", fontsize=11, weight='bold', pad=8)
axes[2].set_xlabel("Sequência de Furos (Hole)", fontsize=10)
axes[2].set_ylabel("")

plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8.5)
plt.tight_layout()

plt.savefig("analise_painel_heatmaps_xgboost.pdf", dpi=600, bbox_inches='tight')
plt.savefig("analise_painel_heatmaps_xgboost.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n[PRONTO] O painel de mapas de calor comparativos foi gerado com sucesso!")