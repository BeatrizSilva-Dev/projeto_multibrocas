import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ==============================================================================
# LEITURA E PADRONIZAÇÃO DOS CSVs
# ==============================================================================
df_xgb  = pd.read_csv("resultados_xgboost_hibrido.csv")
df_mlp  = pd.read_csv("resultados_autoencoder_hibrido.csv")
df_lstm = pd.read_csv("resultados_LSTM_ae.csv")

for df in [df_xgb, df_mlp, df_lstm]:
    df["drill"] = df["drill"].str.lower().str.strip()

# Configuração visual padrão acadêmico
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ==============================================================================
# FUNÇÃO PARA GERAR A MATRIZ DE ESTADOS CATEGÓRICOS (101 pontos em %)
# ==============================================================================
def gerar_matriz_estados_categoricos(df, coluna_pred="prediction"):
    NORM_POINTS = 101
    matriz_estados = []
    nomes = []

    drills = sorted(df["drill"].unique())

    for drill in drills:
        temp = df[df["drill"] == drill].sort_values("hole").reset_index(drop=True)
        n = len(temp)
        if n < 5: continue

        estados_furo = np.zeros(n)

        for i in range(n):
            pred = int(temp.loc[i, coluna_pred])
            label = int(temp.loc[i, "label"]) # Lendo direto a coluna corrigida do CSV!

            if label == 0 and pred == 0:
                estados_furo[i] = 0.0 # Nominal State (Cinza)
            elif label == 0 and pred == 1:
                estados_furo[i] = 1.0 # False/Early Alarm (Laranja Claro)
            elif label == 1 and pred == 1:
                estados_furo[i] = 2.0 # True Alarm (Vermelho Vivo)
            elif label == 1 and pred == 0:
                estados_furo[i] = 3.0 # Undetected Critical Wear (Amarelo Claro)

        x_original = np.linspace(0, 100, n)
        x_target = np.linspace(0, 100, NORM_POINTS)

        # Interpolação por vizinho mais próximo para preservar os IDs categóricos
        estado_interp = np.round(np.interp(x_target, x_original, estados_furo))

        matriz_estados.append(estado_interp)
        nomes.append(drill)

    return np.array(matriz_estados), nomes

# Gerar as matrizes de estados
matriz_xgb, nomes = gerar_matriz_estados_categoricos(df_xgb)
matriz_mlp, _     = gerar_matriz_estados_categoricos(df_mlp)
matriz_lstm, _    = gerar_matriz_estados_categoricos(df_lstm)

# ==============================================================================
# FUNÇÃO DE PLOTAGEM COM SUA NOVA PALETA DE CORES DEFINITIVA
# ==============================================================================
def plotar_heatmap_categorico(matriz, nomes, titulo, arquivo_saida):
    # 1. Proporção vertical otimizada para uma única coluna IEEE
    fig, ax = plt.subplots(figsize=(4.8, 5.2))

    cores_ihm = ['#e6e6e6', '#ffb366', '#d73027', '#ffeb99']
    cmap_categorico = ListedColormap(cores_ihm)

    sns.heatmap(
        matriz,
        cmap=cmap_categorico,
        vmin=0, vmax=3,
        linewidths=0.5,
        linecolor="#f0f0f0",
        cbar=False,
        ax=ax
    )

    ax.axvline(x=80, color="black", linestyle="--", linewidth=1.5)

    drill_ids = [re.search(r"drill_4mm_(\d+)", nome.lower()).group(1).zfill(2) if re.search(r"drill_4mm_(\d+)", nome.lower()) else nome for nome in nomes]
    ax.set_yticks(np.arange(len(drill_ids)) + 0.5)
    ax.set_yticklabels(drill_ids, rotation=0, fontsize=8)

    ax.set_xticks(np.arange(0, 101, 20)) # Mudado para passos de 20% para não sobrepor números
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize=9)

   # ax.set_title(titulo, fontsize=11, weight="bold", pad=10)
    ax.set_xlabel("Tool Life (%)", fontsize=10, weight="bold")
    ax.set_ylabel("Drill Number", fontsize=10, weight="bold")

    # 2. Quebra de linha (\n) nos textos para estreitar a legenda
    patch_normal     = mpatches.Patch(color='#e6e6e6', label='Nominal State\n(No Alarm)')
    patch_false      = mpatches.Patch(color='#ffb366', label='False/Early\nAlarm')
    patch_true       = mpatches.Patch(color='#d73027', label='True Alarm\n(Critical Phase)')
    patch_undetected = mpatches.Patch(color='#ffeb99', label='Undetected\nCritical Wear')

    # 3. Legenda posicionada abaixo do gráfico em matriz 2x2 para economizar espaço lateral
    ax.legend(
        handles=[patch_normal, patch_false, patch_true, patch_undetected],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22), # Mudado de -0.15 para -0.22 para dar espaço ao eixo X
        ncol=2,
        borderaxespad=0.,
        fontsize=8.5,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(arquivo_saida + ".pdf", dpi=600, bbox_inches="tight")
    plt.savefig(arquivo_saida + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# ==============================================================================
# GERAÇÃO DOS TRÊS HEATMAPS INDUSTRIAIS ATUALIZADOS
# ==============================================================================
plotar_heatmap_categorico(matriz_xgb, nomes, "XGBoost - Análise de Risco Baseada em Escuta Humana", "heatmap_risk_xgboost")
plotar_heatmap_categorico(matriz_mlp, nomes, "MLP-AE - Análise de Risco Baseada em Escuta Humana", "heatmap_risk_mlp")
plotar_heatmap_categorico(matriz_lstm, nomes, "LSTM-AE - Análise de Risco Baseada em Escuta Humana", "heatmap_risk_lstm")

print("\n[SUCESSO] Heatmaps gerados com o degradê de risco e grade visível!")