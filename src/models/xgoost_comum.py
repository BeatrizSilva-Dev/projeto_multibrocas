import os
import re
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, recall_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# CONFIGURAÇÕES
ROOT_DATASET = r"C:\...\data\segmented"

MIC_A = "reg_mics"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048)
    rms = librosa.feature.rms(y=y)[0]
    return np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(rms), np.std(rms)])

# 1. CARREGAMENTO
drills_data = {}

print("Iniciando extração do microfone comum (reg_mics)...")
for drill_folder in os.listdir(ROOT_DATASET):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path): continue

    dict_a = {}

    for root, _, fs in os.walk(path):
        if MIC_A not in root.lower(): continue

        for f in fs:
            if not f.lower().endswith(".wav"): continue
            if f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower():
                hole = extract_hole_number(f)
                if hole is None: continue
                dict_a[hole] = os.path.join(root, f)

    holes_sorted = sorted(dict_a.keys())
    if len(holes_sorted) < 12: continue

    holes_sorted = holes_sorted[:-1]

    X_drill, y_drill = [], []
    n = len(holes_sorted)
    for i, hole in enumerate(holes_sorted):
        try:
            y_load, sr_load = librosa.load(dict_a[hole], sr=None)
            feat = extract_features(y_load, sr_load)

            X_drill.append(feat)
            y_drill.append(1 if (i/n) >= 0.8 else 0)
        except: continue

    if len(X_drill) > 0:
        drills_data[drill_folder] = {"X": np.array(X_drill), "y": np.array(y_drill)}

all_probs = []
all_labels = []
regional_results = []
export_data = []
lead_times = []  


# 2. VALIDAÇÃO LODO (LEAVE-ONE-DRILL-OUT)
print(f"Validando modelo Audível em {len(drills_data)} brocas...")

for drill_test in drills_data:
    X_train, y_train = [], []
    for drill_name, data in drills_data.items():
        if drill_name != drill_test:
            X_train.append(data["X"])
            y_train.append(data["y"])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = drills_data[drill_test]["X"]
    y_test = drills_data[drill_test]["y"]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        random_state=42, objective='binary:logistic', scale_pos_weight=8
    )
    model.fit(X_train_sc, y_train)

    probs = model.predict_proba(X_test_sc)[:, 1]
    n_holes = len(probs)
    all_probs.extend(probs)
    all_labels.extend(y_test)

    thresh_final = max(np.percentile(probs[:N_NORMAL], 97.5), 0.1)
    furos_acima = (probs > thresh_final).astype(int)

    preds_persistentes = np.zeros(n_holes)
    janela = 2
    for i in range(janela - 1, n_holes):
        if np.all(furos_acima[i-(janela-1) : i+1] == 1):
            preds_persistentes[i] = 1

    # Cálculo do lead-time por broca
    alert_indices = np.where(preds_persistentes == 1)[0]
    if len(alert_indices) > 0:
        first_alert_hole = alert_indices[0] + 1
        drill_lead_time = n_holes - first_alert_hole
    else:
        drill_lead_time = 0
    lead_times.append(drill_lead_time)

    idx_50 = int(n_holes * 0.5)
    idx_80 = int(n_holes * 0.8)
    regional_results.append({'y_true': 0, 'y_pred': 1 if np.any(preds_persistentes[:idx_50]) else 0})
    regional_results.append({'y_true': 1, 'y_pred': 1 if np.any(preds_persistentes[idx_80:]) else 0})

    for i in range(n_holes):
        export_data.append({
            'drill': drill_test,
            'hole': i + 1,
            'prob_xgb': probs[i],
            'label_real': int(y_test[i]),
            'threshold_xgb': thresh_final,
            'prediction': int(preds_persistentes[i]),
            'drill_lead_time': drill_lead_time
        })


# 3. MÉTRICAS E PLOTAGEM
if regional_results:
    df_res = pd.DataFrame(regional_results)
    cm = confusion_matrix(df_res['y_true'], df_res['y_pred'])

    f1 = f1_score(df_res['y_true'], df_res['y_pred'])
    recall = recall_score(df_res['y_true'], df_res['y_pred'])
    accuracy = accuracy_score(df_res['y_true'], df_res['y_pred'])
    auc_real = roc_auc_score(all_labels, all_probs)

    # Exportação dedicada para a curva ROC combinada
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    df_roc.to_csv("roc_xgboost_comum_data.csv", index=False)

    print(f"F1-score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc_real:.4f}")
    print(f"Mean Lead-Time Window: {np.mean(lead_times):.2f} holes of anticipation")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})

    # Matriz de Confusão 
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=['No Alert', 'Alert'],
                yticklabels=['Normal (50%)', 'Anomaly (20%)'], ax=ax)

    ax.set_ylabel('Ground Truth', fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout(pad=0.1)
    plt.savefig("matriz_xgboost_comum.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    # Curva ROC
    plt.figure(figsize=(3.5, 3))
    plt.plot(fpr, tpr, label=f'AUC = {auc_real:.2f}', color='blue')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_xgboost_comum.pdf", dpi=600, bbox_inches='tight')
    plt.show()

    df_export = pd.DataFrame(export_data)
    nome_csv = "resultados_xgboost_comum.csv"
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), nome_csv)
    df_export.to_csv(csv_path, index=False)
    print(f"\n[SUCESSO] Dados detalhados exportados para: {csv_path}")