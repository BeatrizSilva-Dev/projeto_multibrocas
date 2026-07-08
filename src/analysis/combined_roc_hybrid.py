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
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def plot_all_9_curves_roc():
    plt.figure(figsize=(4.8, 4.4))

    try:
        lstm_hyb = pd.read_csv("hybrid_lstm_roc_data.csv")
        mlp_hyb  = pd.read_csv("hybrid_mlp_roc_data.csv")
        xgb_hyb  = pd.read_csv("hybrid_xgboost_roc_data.csv")

        lstm_ult = pd.read_csv("ultrasonic_lstm_roc_data.csv")
        mlp_ult  = pd.read_csv("ultrasonic_mlp_roc_data.csv")
        xgb_ult  = pd.read_csv("ultrasonic_xgboost_roc_data.csv")

        lstm_aud = pd.read_csv("common_lstm_roc_data.csv")
        mlp_aud  = pd.read_csv("common_mlp_roc_data.csv")
        xgb_aud  = pd.read_csv("common_xgboost_roc_data.csv")

        plt.plot(lstm_hyb['fpr'], lstm_hyb['tpr'], color='#7209b7', linestyle='-', linewidth=1.6,
                 label=f'LSTM-AE Hybrid (AUC = {auc(lstm_hyb["fpr"], lstm_hyb["tpr"]):.2f})')
        plt.plot(mlp_hyb['fpr'], mlp_hyb['tpr'], color='#2ec4b6', linestyle='-', linewidth=1.2,
                 label=f'MLP-AE Hybrid (AUC = {auc(mlp_hyb["fpr"], mlp_hyb["tpr"]):.2f})')
        plt.plot(xgb_hyb['fpr'], xgb_hyb['tpr'], color='#e67e22', linestyle='-', linewidth=1.2,
                 label=f'XGBoost Hybrid (AUC = {auc(xgb_hyb["fpr"], xgb_hyb["tpr"]):.2f})')

        plt.plot(lstm_ult['fpr'], lstm_ult['tpr'], color='#7209b7', linestyle='--', linewidth=1.2, alpha=0.7,
                 label=f'LSTM-AE Ultrasonic (AUC = {auc(lstm_ult["fpr"], lstm_ult["tpr"]):.2f})')
        plt.plot(mlp_ult['fpr'], mlp_ult['tpr'], color='#2ec4b6', linestyle='--', linewidth=1.0, alpha=0.7,
                 label=f'MLP-AE Ultrasonic (AUC = {auc(mlp_ult["fpr"], mlp_ult["tpr"]):.2f})')
        plt.plot(xgb_ult['fpr'], xgb_ult['tpr'], color='#e67e22', linestyle='--', linewidth=1.0, alpha=0.7,
                 label=f'XGBoost Ultrasonic (AUC = {auc(xgb_ult["fpr"], xgb_ult["tpr"]):.2f})')

        plt.plot(lstm_aud['fpr'], lstm_aud['tpr'], color='#7209b7', linestyle=':', linewidth=1.2, alpha=0.7,
                 label=f'LSTM-AE Audible (AUC = {auc(lstm_aud["fpr"], lstm_aud["tpr"]):.2f})')
        plt.plot(mlp_aud['fpr'], mlp_aud['tpr'], color='#2ec4b6', linestyle=':', linewidth=1.0, alpha=0.7,
                 label=f'MLP-AE Audible (AUC = {auc(mlp_aud["fpr"], mlp_aud["tpr"]):.2f})')
        plt.plot(xgb_aud['fpr'], xgb_aud['tpr'], color='#e67e22', linestyle=':', linewidth=1.0, alpha=0.7,
                 label=f'XGBoost Audible (AUC = {auc(xgb_aud["fpr"], xgb_aud["tpr"]):.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=0.8)

        plt.xlabel("False Positive Rate (FPR)", fontweight='bold')
        plt.ylabel("True Positive Rate (TPR)", fontweight='bold')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])

        plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none', framealpha=0.85)
        plt.grid(True, linestyle=':', alpha=0.3)

        plt.tight_layout()

        plt.savefig("combined_roc_9_curves_benchmark.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.savefig("combined_roc_9_curves_benchmark.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.show()
        print("\n[SUCCESS] Master plot with 9 ROC curves successfully generated!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing data file: {e.filename}")

if __name__ == "__main__":
    plot_all_9_curves_roc()