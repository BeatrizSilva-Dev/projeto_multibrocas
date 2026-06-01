import os
import re
import numpy as np
import librosa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT_DATASET = r"C:\Users\beatr\OneDrive\Desktop\Projeto_Brocas_AE\data\segmented"
MIC_A = "reg_mics"  # Foco no microfone audível para o mapeamento espectral
CANAL_ALVO = "4"
N_MFCC = 20

# Definimos o número da broca alvo de forma numérica para fazer a busca flexível
NUMERO_BROCA_ALVO = 5

def extract_hole_number(filename):
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

# ==============================================================================
# 1. LOCALIZAR A PASTA DA BROCA DE FORMA FLEXÍVEL (IGUAL AO SEU SCRIPT DO XGBOOST)
# ==============================================================================
pasta_localizada = None
nome_pasta_real = ""

for pasta in os.listdir(ROOT_DATASET):
    if "drill_4mm" in pasta.lower():
        match = re.search(r"drill_4mm_(\d+)", pasta.lower())
        if match and int(match.group(1)) == NUMERO_BROCA_ALVO:
            pasta_localizada = os.path.join(ROOT_DATASET, pasta)
            nome_pasta_real = pasta
            break

if pasta_localizada is None:
    raise FileNotFoundError(f"Não foi possível encontrar a pasta para a Broca {NUMERO_BROCA_ALVO:02d} em {ROOT_DATASET}")

# 2. COLETAR OS ARQUIVOS WAV DE FORMA ROBUSTA
dict_arquivos = {}

for root, _, fs in os.walk(pasta_localizada):
    if MIC_A in root.lower():
        for f in fs:
            if f.lower().endswith(".wav") and (f"ch{CANAL_ALVO}" in f.lower() or f"tr{CANAL_ALVO}" in f.lower()):
                hole = extract_hole_number(f)
                if hole is None: continue
                dict_arquivos[hole] = os.path.join(root, f)

furos_ordenados = sorted(dict_arquivos.keys())

# Validação de segurança para garantir que os arquivos foram carregados
if not furos_ordenados:
    raise ValueError(f"Nenhum arquivo WAV correspondente ao canal {CANAL_ALVO} foi encontrado na pasta {pasta_localizada}. Verifique os subdiretórios.")

# ==============================================================================
# 3. EXTRAIR A MATRIZ DE ENERGIA (MFCCs) FURO A FURO
# ==============================================================================
matriz_energia = []

print(f"Processando análise espectral para a {nome_pasta_real} ({len(furos_ordenados)} furos localizados)...")
for hole in furos_ordenados:
    try:
        y, sr = librosa.load(dict_arquivos[hole], sr=None)

        # Extração dos MFCCs (20 faixas de representação espectral)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048)

        # Média de energia de cada coeficiente para o furo completo
        mfcc_medio = np.mean(mfcc, axis=1)
        mfcc_medio = np.clip(mfcc_medio, -80, 50)
        matriz_energia.append(mfcc_medio)
    except Exception as e:
        print(f"Aviso: Erro ao processar o furo {hole}: {e}. Pulando...")
        continue

# Converter para Array NumPy e transpor para ter as Frequências nas linhas e Furos nas colunas
matriz_energia = np.array(matriz_energia).T

# Trava de segurança final para o Seaborn
if matriz_energia.size == 0 or matriz_energia.ndim < 2:
    raise ValueError("A matriz de energia gerada está vazia ou corrompida. Não há dados para plotar.")

# ==============================================================================
# 4. CONFIGURAR A ESTÉTICA ACADÊMICA IGUAL AO GRÁFICO DO SÉRGIO
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Criar labels para o eixo Y (de MFCC 1 a MFCC 20)
yticklabels = [f"Banda MFCC {i}" for i in range(N_MFCC, 0, -1)]
matriz_energia_plot = np.flipud(matriz_energia) # Joga as frequências mais altas para o topo da matriz

# Gerar o Heatmap de Energia
sns.heatmap(
    matriz_energia_plot,
    cmap="coolwarm",
    #"RdBu_r",
    #"viridis",
    vmin=-40, vmax=20, # CORREÇÃO DE CONTRASTE: Foca na faixa útil do seu sensor
    xticklabels=5,
    yticklabels=yticklabels,
    cbar_kws={'label': 'Energia Relativa Espectral (dB)'},
    ax=ax
)

# Adicionar a linha vertical tracejada indicando o início da fase crítica (80% da vida da broca)
furo_critico_idx = int(matriz_energia_plot.shape[1] * 0.8)
ax.axvline(x=furo_critico_idx, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)

# Texto de marcação igual ao "possível pré-falha" da imagem dele
ax.text(furo_critico_idx + 1, N_MFCC - 2, 'Fase Crítica (Pré-Falha)', color='cyan', fontsize=9, weight='bold')

ax.set_title(f"Mapeamento de Energia por Faixa de Frequência e Vida da Ferramenta ({nome_pasta_real})", fontsize=11, pad=10)
ax.set_xlabel("Sequência Cronológica de Furos (Hole Sequence)", fontsize=10)
ax.set_ylabel("Representação de Faixas de Frequência (MFCCs)", fontsize=10)

plt.tight_layout()
plt.savefig(f"heatmap_espectral_drill_{NUMERO_BROCA_ALVO:02d}.pdf", dpi=600, bbox_inches='tight')
plt.savefig(f"heatmap_espectral_drill_{NUMERO_BROCA_ALVO:02d}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[SUCESSO] O mapa de calor espectral da Broca {NUMERO_BROCA_ALVO:02d} foi gerado sem erros!")