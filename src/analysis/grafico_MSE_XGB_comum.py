import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("resultados_xgboost_comum.csv")
df["drill"] = df["drill"].str.lower().str.strip()

NORM_POINTS = 100
PROB_ALL = []

for drill in sorted(df["drill"].unique()):
    temp = df[df["drill"] == drill].sort_values("hole").reset_index(drop=True)
    n = len(temp)
    if n < 5:
        continue

    # Linhas do tempo (original vs normalizada em %)
    x_original = np.linspace(0, 100, n)
    x_target = np.linspace(0, 100, NORM_POINTS)

    # Pegamos a probabilidade contínua da classe 1 (score) salva pelo pipeline comum
    coluna_prob = "score"

    prob_interp = np.interp(
        x_target,
        x_original,
        temp[coluna_prob]
    )

    PROB_ALL.append(prob_interp)

PROB_ALL = np.array(PROB_ALL)

# CÁLCULO DA MÉDIA E DESVIO PADRÃO (EM ESPAÇO LINEAR)
prob_mean = np.mean(PROB_ALL, axis=0)
prob_std  = np.std(PROB_ALL, axis=0)

# Truncamos os limites entre 0 e 1 para o desvio padrão não estourar os limites físicos
prob_lower = np.clip(prob_mean - prob_std, 0, 1)
prob_upper = np.clip(prob_mean + prob_std, 0, 1)

tool_life = np.linspace(0, 100, NORM_POINTS)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(7, 4))

# 1. Sombra do Desvio Padrão 
ax.fill_between(
    tool_life,
    prob_lower,
    prob_upper,
    color="#ffcc99",
    alpha=0.4,
    label="±1 SD"
)

# 2. Linha da Média das Probabilidades
ax.plot(
    tool_life,
    prob_mean,
    color="#d35400",
    linewidth=2.5,
    label="Mean Failure Probability"
)

# 3. Linha Vertical dos 80% (Início da Fase Crítica)
ax.axvline(
    80,
    color="black",
    linestyle="--",
    linewidth=1.5
)

# Configurações de Eixos
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Tool Life (%)")
ax.set_ylabel("XGBoost Failure Probability (Score)")
ax.set_title("Average XGBoost Failure Probability (Microfone Comum)")
ax.legend(loc="upper left")

ymin, ymax = ax.get_ylim()
ax.text(
    81,
    ymax * 0.85,
    "Critical Phase",
    fontsize=9,
    weight="bold"
)

plt.grid(True, linestyle=":", alpha=0.3)
plt.tight_layout()

plt.savefig("xgboost_mean_prob_std_comum.pdf", dpi=600, bbox_inches="tight")
plt.savefig("xgboost_mean_prob_std_comum.png", dpi=300, bbox_inches="tight")
plt.show()

print("Gráfico de evolução média do XGBoost (Comum) gerado com sucesso!")