import os
import re
import numpy as np
import librosa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DATASET = r"C:\...\data\segmented"
MIC_A = "reg_mics" 
MIC_B = "ultrasonic_mics" 
CANAL_ALVO = "4"
N_MFCC = 20

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

# COMBINAR TODAS AS BROCAS EM UMA ESCALA DE VIDA ÚTIL NORMALIZADA
MFCC_ALL = []

print("Processando todas as brocas (Fusão Híbrida Completa)...")

for pasta in os.listdir(ROOT_DATASET):
    dict_a = {}
    dict_b = {}

    pasta_path = os.path.join(ROOT_DATASET, pasta)
    if not os.path.isdir(pasta_path):
        continue

    for root, _, fs in os.walk(pasta_path):
        for f in fs:
            if not f.lower().endswith(".wav"):
                continue

            if f"ch{CANAL_ALVO}" not in f.lower() and f"tr{CANAL_ALVO}" not in f.lower():
                continue

            hole = extract_hole_number(f)
            if hole is None:
                continue

            full_path = os.path.join(root, f)
            if MIC_A in root.lower():
                dict_a[hole] = full_path
            elif MIC_B in root.lower():
                dict_b[hole] = full_path

    # Cruza os furos existentes nos dois microfones para garantir a paridade híbrida
    holes = sorted(set(dict_a.keys()) & set(dict_b.keys()))

    if len(holes) < 10:
        print(f"DESCARTADA -> {pasta} | Furos hibrido = {len(holes)}")
        continue

    mfcc_drill = []
    for hole in holes:
        try:
            y_a, sr_a = librosa.load(dict_a[hole], sr=None)
            y_b, sr_b = librosa.load(dict_b[hole], sr=None)

            mfcc_a = librosa.feature.mfcc(y=y_a, sr=sr_a, n_mfcc=N_MFCC, n_fft=2048)
            mfcc_b = librosa.feature.mfcc(y=y_b, sr=sr_b, n_mfcc=N_MFCC, n_fft=2048)

            # Fusão híbrida pela média dos coeficientes espectrais clipados
            energia_hibrida = np.clip(
                (np.mean(mfcc_a, axis=1) + np.mean(mfcc_b, axis=1)) / 2,
                -80, 50
            )
            mfcc_drill.append(energia_hibrida)
        except:
            continue

    mfcc_drill = np.array(mfcc_drill)
    if len(mfcc_drill) == 0:
        continue

    # Interpolação para mapear o ciclo de vida dinâmico em uma matriz percentual de 100 pontos
    original_x = np.linspace(0, 100, len(mfcc_drill))
    target_x = np.linspace(0, 100, 100)

    interp_matrix = []
    for banda in range(N_MFCC):
        interp_band = np.interp(target_x, original_x, mfcc_drill[:, banda])
        interp_matrix.append(interp_band)

    MFCC_ALL.append(np.array(interp_matrix))

if len(MFCC_ALL) == 0:
    raise ValueError("Nenhuma broca válida encontrada.")

MFCC_ALL = np.array(MFCC_ALL)
matriz_media = np.mean(MFCC_ALL, axis=0)
matriz_std = np.std(MFCC_ALL, axis=0)

print(f"\nBrocas processadas final: {len(MFCC_ALL)}")
print(f"MFCC Matrix Shape: {matriz_media.shape}")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(7, 4.8))

# Inverte o eixo Y para colocar a Banda 1 na base e a Banda 20 no topo do gráfico
yticklabels = [f"Banda MFCC {i}" for i in range(N_MFCC, 0, -1)]
matriz_energia_plot = np.flipud(matriz_media)

sns.heatmap(
    matriz_energia_plot,
    cmap="coolwarm",
    vmin=-40, vmax=20,
    xticklabels=10,
    yticklabels=yticklabels,
    cbar_kws={'label': 'Energia Relativa Espectral (dB)'},
    ax=ax
)

furo_critico_idx = 80
ax.axvline(x=furo_critico_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.9)


ax.text(furo_critico_idx + 1.5, 2, 'Fase Crítica', color='black', fontsize=9, weight='bold')

ax.set_title("Average MFCC Energy Evolution Across All Drills (Hybrid Fusion)", fontsize=11, pad=12, weight='bold')
ax.set_xlabel("Tool Life (%)", fontsize=10)
ax.set_ylabel("Representação de Faixas de Frequência (MFCCs)", fontsize=10)

plt.tight_layout()
plt.savefig("heatmap_espectral_drill_hibrido.pdf", dpi=600, bbox_inches='tight')
plt.savefig("heatmap_espectral_drill_hibrido.png", dpi=300, bbox_inches='tight')
plt.show()

print("O mapa de calor espectral completo foi gerado e integrado!")