import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

df_xgb  = pd.read_csv("results_xgboost_hybrid.csv")
df_mlp  = pd.read_csv("results_autoencoder_hybrid.csv")
df_lstm = pd.read_csv("results_LSTM_hybrid.csv")

for df in [df_xgb, df_mlp, df_lstm]:
    df["drill"] = df["drill"].str.lower().str.strip()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def generate_categorical_states_matrix(df, pred_column="prediction"):
    NORM_POINTS = 101
    states_matrix = []
    names = []

    drills = sorted(df["drill"].unique())

    for drill in drills:
        temp = df[df["drill"] == drill].sort_values("hole").reset_index(drop=True)
        n = len(temp)
        if n < 5: continue

        hole_states = np.zeros(n)

        for i in range(n):
            pred = int(temp.loc[i, pred_column])
            label = int(temp.loc[i, "label"]) 

            if label == 0 and pred == 0:
                hole_states[i] = 0.0 
            elif label == 0 and pred == 1:
                hole_states[i] = 1.0 
            elif label == 1 and pred == 1:
                hole_states[i] = 2.0 
            elif label == 1 and pred == 0:
                hole_states[i] = 3.0

        x_original = np.linspace(0, 100, n)
        x_target = np.linspace(0, 100, NORM_POINTS)

        interp_state = np.round(np.interp(x_target, x_original, hole_states))

        states_matrix.append(interp_state)
        names.append(drill)

    return np.array(states_matrix), names

matrix_xgb, names = generate_categorical_states_matrix(df_xgb)
matrix_mlp, _     = generate_categorical_states_matrix(df_mlp)
matrix_lstm, _    = generate_categorical_states_matrix(df_lstm)


def plot_categorical_heatmap(matrix, names, title, output_file):
    fig, ax = plt.subplots(figsize=(4.8, 5.2))

    hmi_colors = ['#e6e6e6', '#ffb366', '#d73027', '#ffeb99']
    categorical_cmap = ListedColormap(hmi_colors)

    sns.heatmap(
        matrix,
        cmap=categorical_cmap,
        vmin=0, vmax=3,
        linewidths=0.5,
        linecolor="#f0f0f0",
        cbar=False,
        ax=ax
    )

    ax.axvline(x=80, color="black", linestyle="--", linewidth=1.5)

    drill_ids = [re.search(r"drill_4mm_(\d+)", nome.lower()).group(1).zfill(2) if re.search(r"drill_4mm_(\d+)", nome.lower()) else nome for nome in names]
    ax.set_yticks(np.arange(len(drill_ids)) + 0.5)
    ax.set_yticklabels(drill_ids, rotation=0, fontsize=8)

    ax.set_xticks(np.arange(0, 101, 20)) 
    ax.set_xticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize=9)

    ax.set_xlabel("Tool Life (%)", fontsize=10, weight="bold")
    ax.set_ylabel("Drill Number", fontsize=10, weight="bold")

    patch_normal     = mpatches.Patch(color='#e6e6e6', label='Nominal State\n(No Alarm)')
    patch_false      = mpatches.Patch(color='#ffb366', label='False/Early\nAlarm')
    patch_true       = mpatches.Patch(color='#d73027', label='True Alarm\n(Critical Phase)')
    patch_undetected = mpatches.Patch(color='#ffeb99', label='Undetected\nCritical Wear')

    ax.legend(
        handles=[patch_normal, patch_false, patch_true, patch_undetected],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22), 
        ncol=2,
        borderaxespad=0.,
        fontsize=8.5,
        frameon=True
    )

    plt.tight_layout()
    plt.savefig(output_file + ".pdf", dpi=600, bbox_inches="tight")
    plt.savefig(output_file + ".png", dpi=300, bbox_inches="tight")
    plt.show()

plot_categorical_heatmap(matrix_xgb, names, "XGBoost - Risk Analysis Based on Human Listening", "heatmap_risk_xgboost")
plot_categorical_heatmap(matrix_mlp, names, "MLP-AE - Risk Analysis Based on Human Listening", "heatmap_risk_mlp")
plot_categorical_heatmap(matrix_lstm, names, "LSTM-AE - Risk Analysis Based on Human Listening", "heatmap_risk_lstm")