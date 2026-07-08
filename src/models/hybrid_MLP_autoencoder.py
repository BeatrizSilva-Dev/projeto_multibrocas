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

hybrid_file = "results_autoencoder_hybrid.csv"
if not os.path.exists(hybrid_file):
    raise FileNotFoundError(f"Generate the file '{hybrid_file}' by running the pipeline first")

df_mlp = pd.read_csv(hybrid_file)

regional_results = []
lead_times = []

for drill in df_mlp['drill'].unique():
    sub_mlp = df_mlp[df_mlp['drill'] == drill].sort_values('hole').reset_index(drop=True)
    n_holes = len(sub_mlp)

    alert_indices = np.where(sub_mlp['prediction'] == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    regional_results.append({'y_true': 0, 'y_pred': 1 if np.any(sub_mlp['prediction'].iloc[:idx_50] == 1) else 0})
    regional_results.append({'y_true': 1, 'y_pred': 1 if np.any(sub_mlp['prediction'].iloc[idx_80:] == 1) else 0})

df_res = pd.DataFrame(regional_results)
cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

f1 = f1_score(df_res['y_true'], df_res['y_pred'])
recall = recall_score(df_res['y_true'], df_res['y_pred'])
acc = accuracy_score(df_res['y_true'], df_res['y_pred'])
auc = roc_auc_score(df_mlp['label'], df_mlp['hybrid_mse'])

print(f"F1-score: {f1:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")
print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")

plt.figure(figsize=(3.5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
            xticklabels=['No Alert', 'Alert'],
            yticklabels=['Normal', 'Anomaly'])

plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("Ground Truth Label", fontweight='bold')
plt.tight_layout()
plt.savefig("matrix_autoencoder_hybrid.pdf", dpi=600, bbox_inches='tight')
plt.savefig("matrix_autoencoder_hybrid.png", dpi=300, bbox_inches='tight')
plt.show()
