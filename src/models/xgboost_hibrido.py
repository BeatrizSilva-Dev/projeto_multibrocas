import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, roc_auc_score, roc_curve

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

if not os.path.exists("resultados_xgboost_hibrido.csv"):
    raise FileNotFoundError("Gere o arquivo 'resultados_xgboost_hibrido.csv' rodando o Mestre primeiro!")

df_xgb = pd.read_csv("resultados_xgboost_hibrido.csv")

# CÁLCULO DOS RESULTADOS REGIONAIS E LEAD-TIMES
regional_results = []
lead_times = []

for drill in df_xgb['drill'].unique():
    sub_xgb = df_xgb[df_xgb['drill'] == drill].sort_values('hole').reset_index(drop=True)
    n_holes = len(sub_xgb)

    # Recuperação do cálculo do Lead-Time 
    alert_indices = np.where(sub_xgb['prediction'] == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    regional_results.append({'y_true': 0, 'y_pred': 1 if np.any(sub_xgb['prediction'].iloc[:idx_50] == 1) else 0})
    regional_results.append({'y_true': 1, 'y_pred': 1 if np.any(sub_xgb['prediction'].iloc[idx_80:] == 1) else 0})

df_res = pd.DataFrame(regional_results)
cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

# MÉTRICAS DO XGBOOST
f1 = f1_score(df_res['y_true'], df_res['y_pred'])
recall = recall_score(df_res['y_true'], df_res['y_pred'])
accuracy = accuracy_score(df_res['y_true'], df_res['y_pred'])
auc_real = roc_auc_score(df_xgb['label'], df_xgb['score'])

print(f"\nMÉTRICAS REAIS DO XGBOOST HÍBRIDO")
print(f"F1-score: {f1:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC:      {auc_real:.4f}")
print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")

# 4. PLOTAGEM DA MATRIZ DE CONFUSÃO 
fig, ax = plt.subplots(figsize=(3.5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=['No Alert', 'Alert'],
            yticklabels=['Normal (50%)', 'Anomaly (20%)'],
            ax=ax)

ax.set_ylabel('Ground Truth', fontweight='bold', fontsize=9)
ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
plt.setp(ax.get_yticklabels(), rotation=90, va="center")
plt.tight_layout(pad=0.1)

plt.savefig("matriz_xgboost_hibrido_final.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.savefig("matriz_xgboost_hibrido_final.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()


# PLOTAGEM DA CURVA ROC DO XGBOOST
fpr, tpr, _ = roc_curve(df_xgb['label'], df_xgb['score'])

df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
df_roc.to_csv("roc_xgboost_hibrido_data.csv", index=False)

fig, ax = plt.subplots(figsize=(3.5, 3))
ax.plot(fpr, tpr, color='#e67e22', linewidth=2, label=f'AUC = {auc_real:.2f}')
ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.7)

ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=9)
ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=9)
ax.set_title('ROC Curve - XGBoost Híbrido', fontweight='bold', fontsize=10, pad=10)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, linestyle=':', alpha=0.3)
plt.tight_layout()

plt.savefig("roc_xgboost_hibrido.pdf", dpi=600, bbox_inches='tight')
plt.savefig("roc_xgboost_hibrido.png", dpi=300, bbox_inches='tight')
plt.show()

print("Gráficos e métricas individuais do XGBoost atualizados sem risco de retreino!")