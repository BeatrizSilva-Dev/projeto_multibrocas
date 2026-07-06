import os
import re
import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

try:
    tf.keras.utils.set_random_seed(42)
except:
    pass

tf.get_logger().setLevel('ERROR')
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras import backend as K

ROOT_DATASET = r"C:\Users\...\segmented"
FILE_ESCUTA = "escuta_manual_projeto_completo.csv"
MIC_A = "reg_mics"
MIC_B = "ultrasonic_mics"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20
SEQ_LEN = 10

if not os.path.exists(FILE_ESCUTA):
    raise FileNotFoundError(f"Gere o arquivo {FILE_ESCUTA} primeiro antes de rodar!")

df_esc = pd.read_csv(FILE_ESCUTA)

def limpar_nome_broca(nome):
    match = re.search(r"drill_4mm_(\d+)", str(nome).lower())
    return match.group(0) if match else str(nome).lower().strip()

df_esc['drill'] = df_esc['drill'].apply(limpar_nome_broca)

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rms = librosa.feature.rms(y=y)[0]
    return np.hstack([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(rms), np.std(rms)])

def create_sequences(X, seq_len):
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

drills_data = {}
print("Carregando base híbrida para alinhamento multimodelo...")
for drill_folder in os.listdir(ROOT_DATASET):
    path = os.path.join(ROOT_DATASET, drill_folder)
    if not os.path.isdir(path): continue
    dict_a, dict_b = {}, {}
    for root, _, fs in os.walk(path):
        for f in fs:
            if not f.lower().endswith(".wav"): continue
            if f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower():
                hole = extract_hole_number(f)
                if hole is None: continue
                if MIC_A in root.lower(): dict_a[hole] = os.path.join(root, f)
                elif MIC_B in root.lower(): dict_b[hole] = os.path.join(root, f)
    common_holes = sorted(list(set(dict_a.keys()) & set(dict_b.keys())))
    if len(common_holes) <= N_NORMAL + 2: continue
    common_holes = common_holes[:-1]
    X_drill, y_drill, hole_indices = [], [], []
    for i, hole in enumerate(common_holes):
        try:
            y_a, sr_a = librosa.load(dict_a[hole], sr=None)
            y_b, sr_b = librosa.load(dict_b[hole], sr=None)
            feat_a = extract_features(y_a, sr_a)
            feat_b = extract_features(y_b, sr_b)
            X_drill.append(np.hstack([feat_a, feat_b]))
            y_drill.append(1 if (i/len(common_holes)) >= 0.8 else 0)
            hole_indices.append(hole)
        except: continue
    if len(X_drill) > N_NORMAL:
        chave_limpa = limpar_nome_broca(drill_folder)
        drills_data[chave_limpa] = {"X": np.array(X_drill), "y": np.array(y_drill), "holes": hole_indices}

registros_comparacao = []
dados_mlp_contínuos = []
dados_lstm_contínuos = []
dados_xgb_contínuos = []


for drill_test in drills_data:
    K.clear_session()

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, y_train = [], []
    for drill_name, data in drills_data.items():
        if drill_name != drill_test:
            X_train.append(data["X"])
            y_train.append(data["y"])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    X_test = drills_data[drill_test]["X"]
    furos_teste = drills_data[drill_test]["holes"]
    n_holes = len(X_test)
    y_test = drills_data[drill_test]["y"]

    scaler_global = StandardScaler()
    X_train_sc = scaler_global.fit_transform(X_train)
    X_test_sc_xgb = scaler_global.transform(X_test)
    n_features = X_train_sc.shape[1]

    #  MLP Autoencoder 
    scaler_mlp = StandardScaler()
    scaler_mlp.fit(X_test[:N_NORMAL])
    X_test_sc_mlp = scaler_mlp.transform(X_test)

    mlp_ae = MLPRegressor(hidden_layer_sizes=(32,16,32), activation="relu", max_iter=1000, random_state=42)
    mlp_ae.fit(X_test_sc_mlp[:N_NORMAL], X_test_sc_mlp[:N_NORMAL])
    mlp_test_errors = np.mean((X_test_sc_mlp - mlp_ae.predict(X_test_sc_mlp))**2, axis=1)
    thresh_mlp = np.percentile(mlp_test_errors[:N_NORMAL], 99.5)
    furos_acima_mlp = (mlp_test_errors > thresh_mlp).astype(int)

    # LSTM Autoencoder 
    scaler_lstm = StandardScaler()
    scaler_lstm.fit(X_test[:N_NORMAL])
    X_test_sc_lstm = scaler_lstm.transform(X_test)
    X_seq_lstm = create_sequences(X_test_sc_lstm, SEQ_LEN)

    lstm_test_errors = np.zeros(n_holes)
    furos_acima_lstm = np.zeros(n_holes)

    if len(X_seq_lstm) >= 5:
        X_train_seq_lstm = X_seq_lstm[:max(1, N_NORMAL - SEQ_LEN + 1)]

        inputs = Input(shape=(SEQ_LEN, n_features))
        encoded = LSTM(32, activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(inputs)
        decoded = RepeatVector(SEQ_LEN)(encoded)
        decoded = LSTM(32, activation='relu',
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42), return_sequences=True)(decoded)
        outputs = TimeDistributed(Dense(n_features, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))(decoded)

        lstm_ae = Model(inputs, outputs)
        lstm_ae.compile(optimizer='adam', loss='mse')
        lstm_ae.fit(X_train_seq_lstm, X_train_seq_lstm, epochs=50, batch_size=8, shuffle=False, verbose=0)

        recon_lstm = lstm_ae.predict(X_seq_lstm, verbose=0)
        errors_seq_lstm = np.mean((X_seq_lstm - recon_lstm) ** 2, axis=(1, 2))

        lstm_test_errors[SEQ_LEN-1:] = errors_seq_lstm

        valid_baseline = lstm_test_errors[SEQ_LEN-1:max(N_NORMAL, SEQ_LEN)]
        if len(valid_baseline) == 0:
            valid_baseline = errors_seq_lstm[:1]
        thresh_lstm = np.percentile(valid_baseline, 99.5)
        furos_acima_lstm = (lstm_test_errors > thresh_lstm).astype(int)

    # XGBoost 
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                        random_state=42, objective='binary:logistic', scale_pos_weight=8)
    xgb.fit(X_train_sc, y_train)
    xgb_probs = xgb.predict_proba(X_test_sc_xgb)[:, 1]

    thresh_xgb = max(np.percentile(xgb_probs[:N_NORMAL], 97.5), 0.1)
    furos_acima_xgb = (xgb_probs > thresh_xgb).astype(int)

    # FILTROS DE JANELA DE PERSISTÊNCIA CRONOLÓGICA INDIVIDUAIS
    pred_xgb = np.zeros(n_holes)
    pred_mlp = np.zeros(n_holes)
    pred_lstm = np.zeros(n_holes)

    for i in range(1, n_holes):
        # XGBoost (Janela = 2)
        if furos_acima_xgb[i] == 1 and furos_acima_xgb[i-1] == 1: pred_xgb[i] = 1

    # MLP-AE (Janela = 10)
    janela_mlp = 10
    for i in range(janela_mlp - 1, n_holes):
        if np.all(furos_acima_mlp[i - (janela_mlp - 1):i + 1] == 1): pred_mlp[i] = 1

    # LSTM-AE (Janela = 8)
    janela_lstm = 8
    for i in range(janela_lstm - 1, n_holes):
        if np.all(furos_acima_lstm[i - (janela_lstm - 1):i + 1] == 1): pred_lstm[i] = 1

    for i in range(n_holes):
        h_furo = furos_teste[i]
        dados_mlp_contínuos.append({
            "drill": drill_test, "hole": h_furo, "hybrid_mse": float(mlp_test_errors[i]), "prediction": int(pred_mlp[i]), "label": int(y_test[i])
        })
        dados_lstm_contínuos.append({
            "drill": drill_test, "hole": h_furo, "hybrid_mse": float(lstm_test_errors[i]), "prediction": int(pred_lstm[i]), "label": int(y_test[i])
        })
        dados_xgb_contínuos.append({
            "drill": drill_test, "hole": h_furo, "score": float(xgb_probs[i]), "prediction": int(pred_xgb[i]), "label": int(y_test[i])
        })

    for i in range(n_holes):
        registros_comparacao.append({
            "drill": drill_test, "hole": furos_teste[i],
            "alarm_xgb": int(pred_xgb[i]), "alarm_mlp": int(pred_mlp[i]), "alarm_lstm": int(pred_lstm[i])
        })

df_models = pd.DataFrame(registros_comparacao)
df_final = pd.merge(df_models, df_esc, on=['drill', 'hole'])

if df_final.empty:
    raise ValueError("ERRO CRÍTICO: O cruzamento de dados resultou vazio.")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Agrupamento e cálculo da média percentual
grouped = df_final.groupby('human_severity_score')[['alarm_xgb', 'alarm_mlp', 'alarm_lstm']].mean() * 100

grouped.columns = ['Supervised (XGBoost)', 'Semi-Supervised (MLP-AE)', 'Semi-Supervised (LSTM-AE)']

labels_severidade = [
    'Level 0:\nNormal',
    'Level 1:\nIncipient Anom.',
    'Level 2:\nSevere Pre-Fail.',
    'Level 3:\nCritical Fail.'
]
grouped.index = [labels_severidade[i] for i in grouped.index]

fig, ax = plt.subplots(figsize=(3.5, 3.8))

cores_originais = ['#e67e22', '#008000', '#7209b7']
grouped.plot(kind='bar', ax=ax, color=cores_originais, edgecolor='black', linewidth=0.5, width=0.75)

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        texto_porcentagem = f"{height:.1f}%"
        ax.text(p.get_x() + p.get_width()/2., height + 3.0, texto_porcentagem,
                ha='center', va='bottom', fontsize=8, weight='bold', rotation=90)
    elif height == 0:
        ax.text(p.get_x() + p.get_width()/2., 3.0, "0.0%",
                ha='center', va='bottom', fontsize=8, weight='bold', rotation=90)

ax.set_ylim(0, 140)

ax.set_ylabel("Alarm Sensitivity, S_alarm (%) ", fontsize=9)
ax.set_xlabel("Operational Acoustic Severity Levels", fontsize=9, labelpad=6)
#ax.set_title("Multi-Model Sensitivity Evaluation", fontsize=10, weight='bold', pad=10)

plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
plt.setp(ax.get_yticklabels(), fontsize=8)


ax.legend(loc='upper left', fontsize=7.5, frameon=True)
plt.grid(True, axis='y', linestyle=':', alpha=0.4)

fig.subplots_adjust(bottom=0.20, left=0.15, right=0.95, top=0.90)

plt.savefig("comparacao_multimodelo_severidade_audio.pdf", dpi=600)
plt.savefig("comparacao_multimodelo_severidade_audio.png", dpi=300)
plt.show()
plt.close()



df_mlp_real = pd.DataFrame(dados_mlp_contínuos)
df_mlp_real.to_csv("resultados_autoencoder_hibrido.csv", index=False)

df_lstm_real = pd.DataFrame(dados_lstm_contínuos)
df_lstm_real.to_csv("resultados_LSTM_ae.csv", index=False)

df_xgb_real = pd.DataFrame(dados_xgb_contínuos)
df_xgb_real.to_csv("resultados_xgboost_hibrido.csv", index=False)

print("\nFramework integrado, sincronizado e travado com sucesso!")