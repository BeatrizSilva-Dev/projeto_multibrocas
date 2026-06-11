import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


SAVE_PATH = r"C:\...\plots\plots_todos_testes_hibrido"
N_NORMAL = 5

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

print("Carregando tabelas estáveis do framework...")
df_xgb = pd.read_csv("resultados_xgboost_hibrido.csv")
df_mlp = pd.read_csv("resultados_autoencoder_hibrido.csv")
df_lstm = pd.read_csv("resultados_LSTM_ae.csv")

lista_brocas = sorted(df_xgb['drill'].unique())
print(f"Gerando gráficos de monitoramento para {len(lista_brocas)} brocas...")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# LOOP DE PLOTAGEM FURO A FURO POR UNIDADE DE FERRAMENTA
for drill in lista_brocas:
    sub_xgb = df_xgb[df_xgb['drill'] == drill].sort_values('hole').reset_index(drop=True)
    sub_mlp = df_mlp[df_mlp['drill'] == drill].sort_values('hole').reset_index(drop=True)
    sub_lstm = df_lstm[df_lstm['drill'] == drill].sort_values('hole').reset_index(drop=True)

    furos = sub_xgb['hole'].values
    n_holes = len(furos)

    # Extração dos thresholds 
    thresh_xgb = max(np.percentile(sub_xgb['score'].iloc[:N_NORMAL], 97.5), 0.1)
    thresh_mlp = np.percentile(sub_mlp['hybrid_mse'].iloc[:N_NORMAL], 99.5)
    thresh_lstm = np.percentile(sub_lstm['hybrid_mse'].iloc[:N_NORMAL], 99.5)

    fig, ax1 = plt.subplots(figsize=(4.5, 3.5))

    idx_falha = int(n_holes * 0.8)
    ax1.axvspan(furos[idx_falha], furos[-1], color='red', alpha=0.06, label='Critical Phase (20%)')

    # Plota as curvas de erro dos Autoencoders (Eixo Esquerdo)
    ax1.plot(furos, sub_lstm['hybrid_mse'], color='#7209b7', marker='o', markersize=2.5,
             linewidth=1.2, label='LSTM-AE MSE')
    ax1.plot(furos, sub_mlp['hybrid_mse'], color='#2ec4b6', marker='v', markersize=2.5,
             linewidth=1.0, linestyle='--', label='MLP-AE MSE')

    # Linhas de Threshold dos Autoencoders
    ax1.axhline(y=thresh_lstm, color='#7209b7', linestyle=':', linewidth=0.9, alpha=0.8,
                label=f'Thresh LSTM ({thresh_lstm:.2f})')
    ax1.axhline(y=thresh_mlp, color='#2ec4b6', linestyle=':', linewidth=0.9, alpha=0.8,
                label=f'Thresh MLP ({thresh_mlp:.2f})')

    ax1.set_xlabel('Hole Sequence (Chronological Process)', labelpad=2)
    ax1.set_ylabel('Reconstruction Error (MSE)', color='#7209b7', labelpad=2)
    ax1.tick_params(axis='y', labelcolor='#7209b7')

    ax1.set_yscale("log")         # Ativa o log para envelopar os erros sem achatar as curvas
    ax1.set_ylim(10**-2, 10**5)   # Limite estrito clássico que protege todo o ecossistema

    # Configuração de Probabilidade do XGBoost (Eixo Direito - Sempre Fixo)
    ax2 = ax1.twinx()
    # Configuração de Probabilidade do XGBoost (Eixo Direito - Sempre Fixo)
    ax2.set_ylim(-0.05, 1.05) # Fornece o zoom máximo em 50 para as brocas mais estáveis

    # O MaxNLocator garante que o Matplotlib escolha divisões bonitas e estritamente inteiras
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Criação do Eixo Gêmeo Direito para o Score de Probabilidade do XGBoost
    ax2.plot(furos, sub_xgb['score'], color='#e67e22', linestyle='-.', marker='s',
             markersize=2, linewidth=1.0, label='XGB Conf.')

    ax2.axhline(y=thresh_xgb, color='#e67e22', linestyle=':', linewidth=0.9, alpha=0.8,
                label=f'Thresh XGB ({thresh_xgb:.2f})')

    ax2.set_ylabel('XGBoost Confidence / Probability', color='#e67e22', labelpad=2)
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.set_ylim(-0.05, 1.05)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    all_lines = lines_1 + lines_2
    all_labels = labels_1 + labels_2

    ax1.legend(
        all_lines, all_labels,
        bbox_to_anchor=(0.01, 0.99),
        loc='upper left',
        ncol=2,
        fontsize=5.8,
        frameon=True,
        framealpha=0.9,
        borderpad=0.2,
        labelspacing=0.15,
        columnspacing=0.4,
        handletextpad=0.2,
        handlelength=1.2
    )

    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()

    pdf_filename = f"{drill}_multimodel_monitoring.pdf"
    png_filename = f"{drill}_multimodel_monitoring.png"

    plt.savefig(os.path.join(SAVE_PATH, pdf_filename), dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(os.path.join(SAVE_PATH, png_filename), dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

print(f"\nGráficos de monitoramento gerados sem erros! Salvos em: {SAVE_PATH}")