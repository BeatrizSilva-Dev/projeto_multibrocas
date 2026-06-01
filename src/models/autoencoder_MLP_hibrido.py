import os
import re
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# CONFIGURAÇÕES
ROOT_DATASET = r"C:\...\data\segmented"

MIC_A = "reg_mics"
MIC_B = "ultrasonic_mics"
CANAL_ALVO = "4"

N_NORMAL = 5
N_MFCC = 20

# FUNÇÕES AUXILIARES
def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)[0]

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(rms),
        np.std(rms)
    ])

# 1. CARREGAMENTO DOS DADOS 
drills_data = {}

print("Extraindo features híbridas...")

for drill_folder in os.listdir(ROOT_DATASET):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path):
        continue

    dict_a = {}
    dict_b = {}

    for root, _, files in os.walk(path):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue

            if f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower():
                hole = extract_hole_number(f)
                if hole is None:
                    continue

                full_path = os.path.join(root, f)

                if MIC_A in root.lower():
                    dict_a[hole] = full_path
                elif MIC_B in root.lower():
                    dict_b[hole] = full_path

    # Furos com dados nos dois sensores
    common_holes = sorted(list(set(dict_a.keys()) & set(dict_b.keys())))

    if len(common_holes) <= N_NORMAL + 2:
        continue

    # remove último furo (falha total)
    common_holes = common_holes[:-1]

    X = []
    for hole in common_holes:
        try:
            y_a, sr_a = librosa.load(dict_a[hole], sr=None)
            y_b, sr_b = librosa.load(dict_b[hole], sr=None)

            feat_a = extract_features(y_a, sr_a)
            feat_b = extract_features(y_b, sr_b)

            X.append(np.hstack([feat_a, feat_b]))  # 92 features

        except:
            continue

    if len(X) > N_NORMAL:
        drills_data[drill_folder] = np.array(X)

all_scores = []
all_labels = []


# 2. VALIDAÇÃO LODO 
regional_results = []
export_data = []
lead_times = []  # Armazena o lead-time de cada ferramenta

print(f"\nValidando em {len(drills_data)} brocas...")

for drill_name, X in drills_data.items():

    n_holes = len(X)

    # NORMALIZAÇÃO SEM VAZAMENTO (usa só os primeiros furos)
    scaler = StandardScaler()
    scaler.fit(X[:N_NORMAL])
    X_scaled = scaler.transform(X)

    # AUTOENCODER
    ae = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32),
        max_iter=1000,
        random_state=42
    )

    ae.fit(X_scaled[:N_NORMAL], X_scaled[:N_NORMAL])

    # RECONSTRUÇÃO
    recon = ae.predict(X_scaled)
    errors = np.mean((X_scaled - recon) ** 2, axis=1)
    all_scores.extend(errors)

    # THRESHOLD 
    threshold = np.percentile(errors[:N_NORMAL], 99.5)

    # DETECÇÃO
    flags = (errors > threshold).astype(int)

    # PERSISTÊNCIA
    preds = np.zeros(n_holes)
    janela = 10

    for i in range(janela - 1, n_holes):
        if np.all(flags[i - (janela - 1):i + 1] == 1):
            preds[i] = 1

    # CÁLCULO DO LEAD-TIME POR BROCA 
    alert_indices = np.where(preds == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1  # Converte para índice base 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    # AVALIAÇÃO REGIONAL
    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)

    alerta_normal = int(np.any(preds[:idx_50]))
    alerta_anomalia = int(np.any(preds[idx_80:]))

    regional_results.append({'y_true': 0, 'y_pred': alerta_normal})
    regional_results.append({'y_true': 1, 'y_pred': alerta_anomalia})

    # EXPORT
    labels = [1 if (i / n_holes) >= 0.8 else 0 for i in range(n_holes)]
    all_labels.extend(labels)

    for i in range(n_holes):
        export_data.append({
            'drill': drill_name,
            'hole': i + 1,
            'hybrid_mse': errors[i],
            'adaptive_threshold': threshold,
            'prediction': preds[i],
            'label': labels[i],
            'drill_lead_time': drill_lead_time  
        })

# 3. RESULTADOS
if regional_results:

    df_res = pd.DataFrame(regional_results)

    f1 = f1_score(df_res['y_true'], df_res['y_pred'])
    recall = recall_score(df_res['y_true'], df_res['y_pred'])
    acc = accuracy_score(df_res['y_true'], df_res['y_pred'])
    auc = roc_auc_score(all_labels, all_scores)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc.to_csv("roc_mlp_hibrido_data.csv", index=False)

    print(f"F1-score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")


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

    #MATRIZ DE CONFUSÃO
    cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])
    plt.figure(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['No Alert', 'Alert'],
                yticklabels=['Normal', 'Anomaly'])

    plt.xlabel("Predicted Label", fontweight='bold')
    plt.ylabel("Ground Truth Label", fontweight='bold')

    plt.tight_layout()
    plt.savefig("matriz_autoencoder_hibrido.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    #CURVA ROC 
    plt.figure(figsize=(3.5, 3))
    plt.plot(fpr, tpr, color='darkgreen', linewidth=1.5, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=1)

    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.legend(loc="lower right")

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("roc_autoencoder_hibrido.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    df_export = pd.DataFrame(export_data)
    df_export.to_csv("resultados_autoencoder_hibrido.csv", index=False)
    print("\n CSV e PDFs (vetoriais) exportados com sucesso.")

else:
    print("Nenhum dado processado.")