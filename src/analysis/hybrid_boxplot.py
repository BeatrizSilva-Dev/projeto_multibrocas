import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def generate_ieee_boxplot():
    try:
        script_path = os.path.dirname(os.path.abspath(__file__))

        path_ae = os.path.join(script_path, "results_autoencoder_hybrid.csv")
        path_xgb = os.path.join(script_path, "results_xgboost_hybrid.csv")
        path_lstm = os.path.join(script_path, "results_LSTM_hybrid.csv")

        df_ae_raw = pd.read_csv(path_ae)
        df_xgb_raw = pd.read_csv(path_xgb)
        df_lstm_raw = pd.read_csv(path_lstm)

        def extract_true_f1_per_drill(df, model):
            f1_results = []
            label_col = 'label' if 'label' in df.columns else 'label_real'
            pred_col = 'prediction'

            for drill in sorted(df['drill'].unique()):
                df_drill = df[df['drill'] == drill]
                y_true = df_drill[label_col].values
                y_pred = df_drill[pred_col].values

                score = f1_score(y_true, y_pred, zero_division=0)
                f1_results.append({'drill': drill, 'f1_score': score, 'modelo': model})
            return pd.DataFrame(f1_results)

        df_f1_lstm = extract_true_f1_per_drill(df_lstm_raw, 'LSTM-AE')
        df_f1_ae = extract_true_f1_per_drill(df_ae_raw, 'MLP-AE')
        df_f1_xgb = extract_true_f1_per_drill(df_xgb_raw, 'XGBoost')

        stat, p_value = wilcoxon(df_f1_lstm['f1_score'], df_f1_xgb['f1_score'])


        n = len(df_f1_lstm)
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
        z = (stat - mean_w) / std_w
        r = z / np.sqrt(n)

        print(f"Wilcoxon p-value: {p_value:.6f}")
        print(f"Effect Size (r): {abs(r):.4f}")

        df_plot = pd.concat([df_f1_lstm, df_f1_ae, df_f1_xgb], axis=0).reset_index(drop=True)

        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        })

        fig, ax = plt.subplots(figsize=(4.5, 3.5))

        official_aligned_colors = ['#7209b7', '#2ec4b6', '#e67e22']

        sns.boxplot(x='modelo', y='f1_score', data=df_plot,
                    palette=official_aligned_colors,
                    width=0.45, linewidth=1.2, fliersize=0, ax=ax)

        sns.stripplot(x='modelo', y='f1_score', data=df_plot,
                      color='black', alpha=0.4, jitter=0.15, size=4, ax=ax)

        ax.set_ylabel('F1-score per Drill Unit', fontweight='bold')
        ax.set_xlabel('Detection Architecture', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)

        plt.grid(axis='y', linestyle=':', alpha=0.4)
        sns.despine()
        plt.tight_layout()

        filename = "hybrid_boxplot_final_IEEE.pdf"
        plt.savefig(os.path.join(script_path, filename), bbox_inches='tight', pad_inches=0.01)
        plt.savefig(os.path.join(script_path, "hybrid_boxplot_final_IEEE.png"), dpi=300, bbox_inches='tight', pad_inches=0.01)

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    generate_ieee_boxplot()