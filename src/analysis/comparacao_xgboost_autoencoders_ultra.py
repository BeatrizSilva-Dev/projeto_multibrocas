import os
import re
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# CONFIGURAÇÕES
DRILL_PATH = r"C:\...\data\segmented"
SAVE_PATH = r"C:\...\src\plots_testes_ultrassonico"
CANAL_ALVO = "4"

MIC_B = "ultrasonic_mics" 
N_NORMAL = 5
N_MFCC = 20
SEQ_LEN = 10

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def extract_features_46d(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        rms = librosa.feature.rms(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        return np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth)
        ])
    except:
        return None

def create_sequences(X, seq_len):
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

# 1. CARREGAMENTO 
processed_data = []
print("Iniciando carregamento do Sensor Ultrassônico...")

pastas = [p for p in os.listdir(DRILL_PATH) if "drill_4mm" in p.lower()]

for pasta in pastas:
    drill_match = re.search(r"drill_4mm_(\d+)", pasta.lower())
    if drill_match:
        drill_num = int(drill_match.group(1))
        caminho_pasta = os.path.join(DRILL_PATH, pasta)

        dict_ultrasonic = {}

        for root, _, files in os.walk(caminho_pasta):
            # Filtra apenas se estiver na pasta ultrasonic_mics
            if MIC_B in root.lower():
                for f in files:
                    if f.lower().endswith(".wav") and (f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower()):
                        h_match = re.search(r"hole(\d+)", f.lower())
                        if h_match:
                            hole_idx = int(h_match.group(1))
                            dict_ultrasonic[hole_idx] = os.path.join(root, f)

        holes = sorted(list(dict_ultrasonic.keys()))

        if len(holes) > N_NORMAL + 2:
            holes = holes[:-1]  # Remove último furo
            n_total = len(holes)
            ponto_corte = int(n_total * 0.8)

            print(f"Lendo Drill {drill_num:02d} - {n_total} furos encontrados (Ultrassom).")
            for i, h_idx in enumerate(holes):
                feat = extract_features_46d(dict_ultrasonic[h_idx])

                if feat is not None:
                    processed_data.append({
                        'drill': drill_num,
                        'hole': h_idx,
                        'features': feat, # Apenas 46 dimensões
                        'label': 1 if i >= ponto_corte else 0
                    })

df = pd.DataFrame(processed_data)
lista_brocas = sorted(df['drill'].unique())

# 2. PROCESSAMENTO E PLOTAGEM
for TEST_DRILL in lista_brocas:
    df_train = df[df['drill'] != TEST_DRILL]
    df_test = df[df['drill'] == TEST_DRILL].sort_values('hole')

    scaler = StandardScaler()
    X_train_raw = np.stack(df_train['features'].values)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(np.stack(df_test['features'].values))

    n_features = X_train_scaled.shape[1]
    furos = df_test['hole'].values
    n_holes = len(X_test_scaled)

    # MLP-AE
    X_ae_train = []
    for d in df_train['drill'].unique():
        subset = df_train[(df_train['drill'] == d) & (df_train['hole'] <= N_NORMAL)]
        if not subset.empty:
            X_ae_train.append(np.stack(subset['features'].values))

    X_ae_train_scaled = scaler.transform(np.vstack(X_ae_train))
    mlp_ae = MLPRegressor(hidden_layer_sizes=(32,16,32), activation="relu", max_iter=1000, random_state=42)
    mlp_ae.fit(X_ae_train_scaled, X_ae_train_scaled)
    mlp_errors = np.mean((X_test_scaled - mlp_ae.predict(X_test_scaled))**2, axis=1)

    # LSTM-AE 
    X_train_seq_list = []
    for d in df_train['drill'].unique():
        subset = df_train[df_train['drill'] == d].sort_values('hole')
        if len(subset) >= SEQ_LEN:
            subset_scaled = scaler.transform(np.stack(subset['features'].values))
            seqs = create_sequences(subset_scaled, SEQ_LEN)
            X_train_seq_list.append(seqs[:max(1, N_NORMAL - SEQ_LEN + 1)])

    X_train_seq = np.vstack(X_train_seq_list)
    inputs = Input(shape=(SEQ_LEN, n_features))
    encoded = LSTM(32, activation='relu')(inputs)
    decoded = RepeatVector(SEQ_LEN)(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    lstm_ae = Model(inputs, outputs)
    lstm_ae.compile(optimizer='adam', loss='mse')
    lstm_ae.fit(X_train_seq, X_train_seq, epochs=30, batch_size=8, verbose=0)

    padding_holes = np.repeat(X_test_scaled[0:1], SEQ_LEN - 1, axis=0)
    padded_test = np.vstack([padding_holes, X_test_scaled])
    X_test_seq = create_sequences(padded_test, SEQ_LEN)
    lstm_recon = lstm_ae.predict(X_test_seq, verbose=0)
    lstm_errors = np.mean((X_test_seq - lstm_recon) ** 2, axis=(1, 2))

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=4)
    xgb.fit(X_train_scaled, df_train['label'].values)
    xgb_probs = xgb.predict_proba(X_test_scaled)[:, 1]


    # PLOT 
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})
    fig, ax1 = plt.subplots(figsize=(4.0, 3.2))

    idx_falha = int(len(furos) * 0.8)
    ax1.axvspan(furos[idx_falha], furos[-1], color='red', alpha=0.08, label='Critical Phase (20%)')

    ax1.plot(furos, lstm_errors, color='#00a8cc', marker='o', markersize=3, linewidth=1.3, label='LSTM-AE MSE')
    ax1.plot(furos, mlp_errors, color='#2ca02c', marker='v', markersize=3, linewidth=1.1, linestyle='--', label='MLP-AE MSE')

    ax1.set_xlabel('Hole Sequence', fontweight='bold', labelpad=2)
    ax1.set_ylabel('Reconstruction Error (MSE)', color='#003f5c', fontweight='bold', labelpad=2)
    ax1.tick_params(axis='y', labelcolor='#003f5c')

    ax2 = ax1.twinx()
    ax2.plot(furos, xgb_probs, color='#ff7a00', linestyle='-.', marker='s', markersize=2.5, linewidth=1.1, label='XGB Conf.')
    
    ax2.set_ylabel('Confidence / Probability Score', color='#ff7a00', fontweight='bold', labelpad=2)
    ax2.tick_params(axis='y', labelcolor='#ff7a00')
    ax2.set_ylim(-0.05, 1.05)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=7.5, frameon=True)

    plt.title(f"Drill {TEST_DRILL:02d} - Ultrasonic Sensor")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_PATH, f"drill_{TEST_DRILL:02d}_ultrasonic.png"), dpi=600, bbox_inches='tight')
    plt.close()

print(f"\n[SUCESSO] Monitoramento concluído para o Sensor Ultrassônico! Gráficos em: {SAVE_PATH}")