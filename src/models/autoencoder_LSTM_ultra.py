import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, roc_auc_score, roc_curve

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

arquivo_ultrasonic = "resultados_lstm_ultrassonico.csv"
if not os.path.exists(arquivo_ultrasonic):
    raise FileNotFoundError(f"Gere o arquivo '{arquivo_ultrasonic}' executando o pipeline ultrassônico primeiro!")

df_lstm = pd.read_csv(arquivo_ultrasonic)

# 2. CÁLCULO DOS RESULTADOS REGIONAIS E LEAD-TIMES 
regional_results = []
lead_times = []

for drill in df_lstm['drill'].unique():
    sub_lstm = df_lstm[df_lstm['drill'] == drill].sort_values('hole').reset_index(drop=True)
    n_holes = len(sub_lstm)

    # Resgata o lead-time real do arquivo unificado do canal ultrassônico com janela=8
    alert_indices = np.where(sub_lstm['prediction'] == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    # Avaliação prognóstica regional (Fase Saudável 50% vs Fase de Anomalia Crítica 80%)
    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    regional_results.append({'y_true': 0, 'y_pred': int(np.any(sub_lstm['prediction'].iloc[:idx_50] == 1))})
    regional_results.append({'y_true': 1, 'y_pred': int(np.any(sub_lstm['prediction'].iloc[idx_80:] == 1))})

df_res = pd.DataFrame(regional_results)
cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

# 3. EXTRAÇÃO DAS MÉTRICAS OPERACIONAIS 
f1 = f1_score(df_res['y_true'], df_res['y_pred'])
recall = recall_score(df_res['y_true'], df_res['y_pred'])
acc = accuracy_score(df_res['y_true'], df_res['y_pred'])
auc = roc_auc_score(df_lstm['label'], df_lstm['hybrid_mse'])

print("\nRESULTADOS CONSOLIDADOS LSTM-AE (MICROFONE ULTRASSÔNICO)")
print(f"F1-score: {f1:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")
print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")

# 4. PLOTAGEM DA MATRIZ DE CONFUSÃO 
cmap_roxo = LinearSegmentedColormap.from_list("CustomPurple", ["#ffffff", "#7209b7"])

plt.figure(figsize=(3.5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_roxo, cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=['No Alert', 'Alert'],
            yticklabels=['Normal', 'Anomaly'])

plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("Ground Truth Label", fontweight='bold')
plt.tight_layout()

plt.savefig("matriz_lstm_ultrasonic.pdf", dpi=600, bbox_inches='tight')
plt.savefig("matriz_lstm_ultrasonic.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. PLOTAGEM DA CURVA ROC
fpr, tpr, _ = roc_curve(df_lstm['label'], df_lstm['hybrid_mse'])

df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
df_roc.to_csv("roc_lstm_ultrasonic_data.csv", index=False)

plt.figure(figsize=(3.5, 3))
plt.plot(fpr, tpr, color='#7209b7', linewidth=1.5, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)

plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

plt.savefig("roc_autoencoder_LSTM_ultrasonic.pdf", dpi=600, bbox_inches='tight')
plt.savefig("roc_autoencoder_LSTM_ultrasonic.png", dpi=300, bbox_inches='tight')
plt.show()

print("Relatórios e plots do LSTM-AE Ultrassônico gerados com consistência absoluta!")