import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_hybrid_benchmark_roc():
    plt.figure(figsize=(4, 3.5)) 

    try:
        lstm_data = pd.read_csv("roc_lstm_hibrido_data.csv")
        mlp_data = pd.read_csv("roc_mlp_hibrido_data.csv")
        xgb_data = pd.read_csv("roc_xgboost_hibrido_data.csv")

        auc_lstm = auc(lstm_data['fpr'], lstm_data['tpr'])
        auc_mlp = auc(mlp_data['fpr'], mlp_data['tpr'])
        auc_xgb = auc(xgb_data['fpr'], xgb_data['tpr'])

        # PLOTA AS LINHAS
        plt.plot(lstm_data['fpr'], lstm_data['tpr'],
                 color='darkviolet', linestyle='-', linewidth=1.6,
                 label=f'LSTM-AE Hybrid (AUC = {auc_lstm:.2f})')

        plt.plot(mlp_data['fpr'], mlp_data['tpr'],
                 color='seagreen', linestyle='--', linewidth=1.3,
                 label=f'MLP-AE Hybrid (AUC = {auc_mlp:.2f})')

        plt.plot(xgb_data['fpr'], xgb_data['tpr'],
                 color='darkorange', linestyle='-.', linewidth=1.3,
                 label=f'XGBoost Hybrid (AUC = {auc_xgb:.2f})')

        # Linha diagonal de referência aleatória 
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1)

        plt.xlabel("False Positive Rate", fontweight='bold')
        plt.ylabel("True Positive Rate", fontweight='bold')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()

        plt.savefig("combined_roc_hybrid_benchmark.pdf", dpi=600, bbox_inches='tight')
        plt.show()
        print("[SUCESSO] Gráfico híbrido exportado: 'combined_roc_hybrid_benchmark.pdf'")

    except FileNotFoundError as e:
        print(f"\n[ERRO] Arquivo não encontrado: {e.filename}")
        print("Certifique-se de que rodou os scripts do LSTM, MLP e XGBoost na versão HÍBRIDA antes.")

if __name__ == "__main__":
    plot_hybrid_benchmark_roc()