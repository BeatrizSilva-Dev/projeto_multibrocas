import os
import re
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
import seaborn as sns
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# CONFIGURAÇÕES
ROOT_DATASET = r"C:\...\data\segmented"
FILE_ESCUTA = "escuta_manual_projeto_completo.csv"
MIC_A = "reg_mics"
MIC_B = "ultrasonic_mics"
CANAL_ALVO = "4"
N_NORMAL = 5
N_MFCC = 20
SEQ_LEN = 10
tf.random.set_seed(42)

if not os.path.exists(FILE_ESCUTA):
    raise FileNotFoundError(f"Gere o arquivo {FILE_ESCUTA} primeiro antes de rodar a comparação unificada!")

df_esc = pd.read_csv(FILE_ESCUTA)

def limpar_nome_broca(nome):
    match = re.search(r"drill_4mm_(\d+)", str(nome).lower())
    return match.group(0) if match else str(nome).lower().strip()

df_esc['drill'] = df_esc['drill'].apply(limpar_nome_broca)

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048)
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
    if len(common_holes) < 12: continue
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
    if len(X_drill) > 0:
        # Salva a chave do dicionário já com o nome limpo e padronizado!
        chave_limpa = limpar_nome_broca(drill_folder)
        drills_data[chave_limpa] = {"X": np.array(X_drill), "y": np.array(y_drill), "holes": hole_indices}

# 2. EXECUÇÃO DO LOOP LODO PARA EXTRAÇÃO DOS ALARMES DOS 3 MODELOS
registros_comparacao = []
print("Executando inferência combinada (XGBoost + MLP-AE + LSTM-AE)...")

for drill_test in drills_data:
    tf.keras.backend.clear_session()

    X_train, y_train, X_ae_train = [], [], []
    for drill_name, data in drills_data.items():
        if drill_name != drill_test:
            X_train.append(data["X"])
            y_train.append(data["y"])
            X_ae_train.append(data["X"][:N_NORMAL])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_ae_train_scaled = np.vstack(X_ae_train)

    X_test = drills_data[drill_test]["X"]
    furos_teste = drills_data[drill_test]["holes"]
    n_holes = len(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_ae_train_scaled = scaler.transform(X_ae_train_scaled)
    X_test_sc = scaler.transform(X_test)
    n_features = X_train_sc.shape[1]

    # MLP Autoencoder
    mlp_ae = MLPRegressor(hidden_layer_sizes=(32,16,32), activation="relu", max_iter=1000, random_state=42)
    mlp_ae.fit(X_ae_train_scaled, X_ae_train_scaled)
    mlp_test_errors = np.mean((X_test_sc - mlp_ae.predict(X_test_sc))**2, axis=1)
    thresh_mlp = np.percentile(np.mean((X_ae_train_scaled - mlp_ae.predict(X_ae_train_scaled))**2, axis=1), 99.5)

    # LSTM Autoencoder
    X_train_seq_list = []
    for drill_name, data in drills_data.items():
        if drill_name != drill_test and len(data["X"]) >= SEQ_LEN:
            seqs = create_sequences(scaler.transform(data["X"]), SEQ_LEN)
            X_train_seq_list.append(seqs[:max(1, N_NORMAL - SEQ_LEN + 1)])
    X_train_seq = np.vstack(X_train_seq_list)

    inputs = Input(shape=(SEQ_LEN, n_features))
    encoded = LSTM(32, activation='relu')(inputs)
    decoded = RepeatVector(SEQ_LEN)(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    lstm_ae = Model(inputs, outputs)
    lstm_ae.compile(optimizer='adam', loss='mse')
    lstm_ae.fit(X_train_seq, X_train_seq, epochs=15, batch_size=8, verbose=0)

    padding_holes = np.repeat(X_test_sc[0:1], SEQ_LEN - 1, axis=0)
    padded_test = np.vstack([padding_holes, X_test_sc])
    X_test_seq = create_sequences(padded_test, SEQ_LEN)
    lstm_test_errors = np.mean((X_test_seq - lstm_ae.predict(X_test_seq, verbose=0)) ** 2, axis=(1, 2))
    thresh_lstm = np.percentile(np.mean((X_train_seq - lstm_ae.predict(X_train_seq, verbose=0))**2, axis=(1, 2)), 99.5)

    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=3, scale_pos_weight=8, random_state=42)
    xgb.fit(X_train_sc, y_train)
    xgb_probs = xgb.predict_proba(X_test_sc)[:, 1]
    thresh_xgb = np.percentile(xgb.predict_proba(X_train_sc)[:, 1], 97.5)

    # Predições com Janela de Persistência 
    pred_xgb = np.zeros(n_holes)
    pred_mlp = np.zeros(n_holes)
    pred_lstm = np.zeros(n_holes)

    for i in range(1, n_holes):
        if xgb_probs[i] > thresh_xgb and xgb_probs[i-1] > thresh_xgb: pred_xgb[i:] = 1; break
    for i in range(1, n_holes):
        if mlp_test_errors[i] > thresh_mlp and mlp_test_errors[i-1] > thresh_mlp: pred_mlp[i:] = 1; break
    for i in range(1, n_holes):
        if lstm_test_errors[i] > thresh_lstm and lstm_test_errors[i-1] > thresh_lstm: pred_lstm[i:] = 1; break

    for i in range(n_holes):
        registros_comparacao.append({
            "drill": drill_test, "hole": furos_teste[i],
            "alarm_xgb": int(pred_xgb[i]), "alarm_mlp": int(pred_mlp[i]), "alarm_lstm": int(pred_lstm[i])
        })

# 3. MERGE ROBUSTO COM A PLANILHA DE ESCUTA MANUAL
df_models = pd.DataFrame(registros_comparacao)
df_final = pd.merge(df_models, df_esc, on=['drill', 'hole'])

if df_final.empty:
    raise ValueError("ERRO CRÍTICO: O cruzamento de dados resultou vazio. Verifique se os números dos furos batem entre os arquivos.")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Agrupar dados para calcular a taxa de ativação por nível de severidade de áudio
grouped = df_final.groupby('human_severity_score')[['alarm_xgb', 'alarm_mlp', 'alarm_lstm']].mean() * 100
grouped.columns = ['Supervisionado (XGBoost)', 'Semi-Supervisionado (MLP-AE)', 'Semi-Supervisionado (LSTM-AE)']

labels_severidade = [
    '0: Normal\n(Som Limpo)',
    '1: Anomalia Leve\n(Início do Guizo)',
    '2: Pré-Falha Severa\n(Chiado/Rádio)',
    '3: Falha Crítica\n(Travamento)'
]

# Certifica-se de reindexar apenas para os índices que realmente existem no agrupamento
grouped.index = [labels_severidade[i] for i in grouped.index]

fig, ax = plt.subplots(figsize=(7.5, 4.5))

grouped.plot(
    kind='bar',
    ax=ax,
    color=['#e67e22', '#2ec4b6', '#7209b7'],
    edgecolor='black',
    linewidth=0.6,
    width=0.7
)

# Adicionar valores em cima das barras
for p in ax.patches:
    height = p.get_height()
    if height >= 0:
        ax.text(
            p.get_x() + p.get_width()/2.,
            height + 1.5,
            f'{height:.0f}%',
            ha='center',
            va='bottom',
            fontsize=8,
            weight='bold'
        )

ax.set_ylabel("Furos com Alarme Ativo (% de Sensibilidade)", fontsize=10)
ax.set_xlabel("Categorização da Percepção Acústica Humana (Escuta Manual)", fontsize=10, labelpad=8)
ax.set_title("Sensibilidade Comparativa Multimodelo por Severidade do Áudio", fontsize=11, weight='bold', pad=12)
ax.set_ylim(0, 115)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
ax.legend(loc='upper left', fontsize=9, frameon=True)
plt.grid(True, axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("comparacao_multimodelo_severidade_audio.pdf", dpi=600, bbox_inches='tight')
plt.savefig("comparacao_multimodelo_severidade_audio.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n[SUCESSO] Gráfico de comparação tri-modelo gerado e salvo com alta resolução!")