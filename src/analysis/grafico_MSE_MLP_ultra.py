import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arquivo_ultrasonic = "resultados_autoencoder_ultrassonico.csv"
df = pd.read_csv(arquivo_ultrasonic)
df["drill"] = df["drill"].str.lower().str.strip()

NORM_POINTS = 100
MSE_ALL = []

for drill in sorted(df["drill"].unique()):
    temp = df[df["drill"] == drill].sort_values("hole").reset_index(drop=True)
    n = len(temp)
    if n < 5:
        continue

    x_original = np.linspace(0, 100, n)
    x_target = np.linspace(0, 100, NORM_POINTS)

    mse_interp = np.interp(
        x_target,
        x_original,
        temp["hybrid_mse"]
    )
    MSE_ALL.append(mse_interp)

MSE_ALL = np.array(MSE_ALL)

# CÁLCULO ESTATÍSTICO NO ESPAÇO LOGARÍTMICO
log_mse_all = np.log10(MSE_ALL + 1e-8)

log_mean = np.mean(log_mse_all, axis=0)
log_std  = np.std(log_mse_all, axis=0)

mse_mean = 10 ** log_mean
mse_lower = 10 ** (log_mean - log_std)
mse_upper = 10 ** (log_mean + log_std)

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

# 1. Sombra do Desvio Padrão Contínua
ax.fill_between(
    tool_life,
    mse_lower,
    mse_upper,
    color="#2ec4b6",
    alpha=0.2,
    label="±1 SD"
)

# 2. Linha da Média dos Erros 
ax.plot(
    tool_life,
    mse_mean,
    color="#2ec4b6",
    linewidth=2.5,
    label="Mean MSE"
)

# 3. Linha Vertical dos 80% (Início da Fase Crítica)
ax.axvline(
    80,
    color="black",
    linestyle="--",
    linewidth=1.5
)

# Configurações de Escala e Eixos 
ax.set_yscale("log")
ax.set_ylim(10**-2, 10**5)

ax.text(
    81,
    10**4.2,
    "Critical Phase",
    fontsize=9,
    weight="bold"
)

ax.set_xlabel("Tool Life (%)")
ax.set_ylabel("Ultrasonic Reconstruction Error (MSE)")
ax.set_title("Average MLP-AE Reconstruction Error (Microfone Ultrassônico)")
ax.legend(loc="upper left")

plt.grid(True, which="both", linestyle=":", alpha=0.3)
plt.tight_layout()

plt.savefig("mlp_mean_mse_std_ultrasonic.pdf", dpi=600, bbox_inches="tight")
plt.savefig("mlp_mean_mse_std_ultrasonic.png", dpi=300, bbox_inches="tight")
plt.show()

print("Gráfico de rampa média do MLP-AE (Ultrassônico) gerado com sucesso!")