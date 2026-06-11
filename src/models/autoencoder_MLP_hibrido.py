import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, roc_auc_score, roc_curve

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

if not os.path.exists("resultados_autoencoder_hibrido.csv"):
    raise FileNotFoundError("Gere o arquivo 'resultados_autoencoder_hibrido.csv' executando o Mestre primeiro!")

df_mlp = pd.read_csv("resultados_autoencoder_hibrido.csv")

# CÁLCULO DOS RESULTADOS REGIONAIS E LEAD-TIMES
regional_results = []
lead_times = []

for drill in df_mlp['drill'].unique():
    sub_mlp = df_mlp[df_mlp['drill'] == drill].sort_values('hole').reset_index(drop=True)
    n_holes = len(sub_mlp)

    # Resgata o lead-time real do arquivo unificado
    alert_indices = np.where(sub_mlp['prediction'] == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    # Fatiamento prognóstico regional (Fase Saudável 50% vs Fase de Falha 80%)
    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    regional_results.append({'y_true': 0, 'y_pred': 1 if np.any(sub_mlp['prediction'].iloc[:idx_50] == 1) else 0})
    regional_results.append({'y_true': 1, 'y_pred': 1 if np.any(sub_mlp['prediction'].iloc[idx_80:] == 1) else 0})

df_res = pd.DataFrame(regional_results)
cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

# EXTRAÇÃO DAS MÉTRICAS OPERACIONAIS
f1 = f1_score(df_res['y_true'], df_res['y_pred'])
recall = recall_score(df_res['y_true'], df_res['y_pred'])
acc = accuracy_score(df_res['y_true'], df_res['y_pred'])
auc = roc_auc_score(df_mlp['label'], df_mlp['hybrid_mse'])

print("\nRESULTADOS CONSOLIDADOS MLP-AE")
print(f"F1-score: {f1:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")
print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")

# 4. PLOTAGEM DA MATRIZ DE CONFUSÃO 
plt.figure(figsize=(3.5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=['No Alert', 'Alert'],
            yticklabels=['Normal', 'Anomaly'])

plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("Ground Truth Label", fontweight='bold')
plt.tight_layout()
plt.savefig("matriz_autoencoder_hibrido.pdf", dpi=600, bbox_inches='tight')
plt.savefig("matriz_autoencoder_hibrido.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. PLOTAGEM DA CURVA ROC 
fpr, tpr, _ = roc_curve(df_mlp['label'], df_mlp['hybrid_mse'])

df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
df_roc.to_csv("roc_mlp_hibrido_data.csv", index=False)

plt.figure(figsize=(3.5, 3))
plt.plot(fpr, tpr, color='darkgreen', linewidth=1.5, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)

plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig("roc_autoencoder_hibrido.pdf", dpi=600, bbox_inches='tight')
plt.savefig("roc_autoencoder_hibrido.png", dpi=300, bbox_inches='tight')
plt.show()

print("Relatórios individuais do MLP Autoencoder exportados com consistência absoluta!")