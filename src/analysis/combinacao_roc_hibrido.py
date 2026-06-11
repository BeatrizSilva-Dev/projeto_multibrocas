import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Configurações de estilo estrito para a IEEE (Times New Roman)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_hybrid_benchmark_roc():
    # Tamanho ideal clássico para ocupar uma coluna de artigo de duas colunas
    plt.figure(figsize=(4.2, 3.8))

    try:
        # Carrega os vetores de dados estáveis salvos pelos scripts individuais
        lstm_data = pd.read_csv("roc_lstm_hibrido_data.csv")
        mlp_data = pd.read_csv("roc_mlp_hibrido_data.csv")
        xgb_data = pd.read_csv("roc_xgboost_hibrido_data.csv")

        # Calcula a área sob a curva baseada na integração trapezoidal direta dos vetores
        auc_lstm = auc(lstm_data['fpr'], lstm_data['tpr'])
        auc_mlp = auc(mlp_data['fpr'], mlp_data['tpr'])
        auc_xgb = auc(xgb_data['fpr'], xgb_data['tpr'])

        # ➔ CORREÇÃO VISUAL CRÍTICA: Aplicação das cores oficiais trancadas e estilos de linha
        # LSTM-AE recebe linha contínua em destaque
        plt.plot(lstm_data['fpr'], lstm_data['tpr'],
                 color='#7209b7', linestyle='-', linewidth=1.8,
                 label=f'LSTM-AE Hybrid (AUC = {auc_lstm:.2f})')

        plt.plot(mlp_data['fpr'], mlp_data['tpr'],
                 color='#2ec4b6', linestyle='--', linewidth=1.4,
                 label=f'MLP-AE Hybrid (AUC = {auc_mlp:.2f})')

        plt.plot(xgb_data['fpr'], xgb_data['tpr'],
                 color='#e67e22', linestyle='-.', linewidth=1.4,
                 label=f'XGBoost Hybrid (AUC = {auc_xgb:.2f})')

        # Linha diagonal cinza indicando a linha de base aleatória (Classificador Aleatório)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)

        # Formatação rígida padrão IEEE
        plt.xlabel("False Positive Rate (FPR)", fontweight='bold')
        plt.ylabel("True Positive Rate (TPR)", fontweight='bold')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        # Posiciona a legenda no canto inferior direito de forma limpa e com borda discreta
        plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()

        # Exportação dupla em alta resolução (Vetor PDF para escrita + Imagem PNG para slides)
        plt.savefig("combined_roc_hybrid_benchmark.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.savefig("combined_roc_hybrid_benchmark.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.show()
        print("\n[SUCESSO] Gráfico comparativo de curvas ROC exportado com consistência visual!")

    except FileNotFoundError as e:
        print(f"\n[ERRO] Arquivo não encontrado: {e.filename}")
        print("Certifique-se de que executou os scripts individuais do LSTM, MLP e XGBoost antes para gerar as tabelas de ROC.")

if __name__ == "__main__":
    plot_hybrid_benchmark_roc()