import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def gerar_boxplot_ultrasonic_ieee():
    try:
        caminho_script = os.path.dirname(os.path.abspath(__file__))

        path_ae = os.path.join(caminho_script, "resultados_autoencoder_ultrassonico.csv")
        path_xgb = os.path.join(caminho_script, "resultados_xgboost_ultrassonico.csv")
        path_lstm = os.path.join(caminho_script, "resultados_lstm_ultrassonico.csv")

        df_ae_raw = pd.read_csv(path_ae)
        df_xgb_raw = pd.read_csv(path_xgb)
        df_lstm_raw = pd.read_csv(path_lstm)

        def extrair_f1_real_por_broca(df, modelo):
            resultados_f1 = []
            label_col = 'label' if 'label' in df.columns else 'label_real'
            pred_col = 'prediction'

            for drill in sorted(df['drill'].unique()):
                df_drill = df[df['drill'] == drill]
                y_true = df_drill[label_col].values
                y_pred = df_drill[pred_col].values

                score = f1_score(y_true, y_pred, zero_division=0)
                resultados_f1.append({'drill': drill, 'f1_score': score, 'modelo': modelo})
            return pd.DataFrame(resultados_f1)

        df_f1_lstm = extrair_f1_real_por_broca(df_lstm_raw, 'LSTM-AE')
        df_f1_ae = extrair_f1_real_por_broca(df_ae_raw, 'MLP-AE')
        df_f1_xgb = extrair_f1_real_por_broca(df_xgb_raw, 'XGBoost')

        # 4. Teste de Wilcoxon e Effect Size exato com aproximação estatística estável
        res_wilc = wilcoxon(df_f1_lstm['f1_score'], df_f1_xgb['f1_score'], method='approx')
        p_value = res_wilc.pvalue
        z_stat = res_wilc.zstatistic
        n = len(df_f1_lstm)
        r = z_stat / np.sqrt(n) # Fórmula exata do Size Effect para Wilcoxon

        print("ANÁLISE ESTATÍSTICA SENSOR ULTRASSÔNICO (LSTM-AE vs XGBoost)")
        print(f"p-value: {p_value:.6f}")
        print(f"Effect size (r): {abs(r):.4f}")

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

        cores_trancadas = ['#7209b7', '#2ec4b6', '#e67e22']

        sns.boxplot(x='modelo', y='f1_score', data=df_plot,
                    palette=cores_trancadas,
                    width=0.45, linewidth=1.2, fliersize=0, ax=ax)

        # Dispersão dos pontos individuais de cada broca
        sns.stripplot(x='modelo', y='f1_score', data=df_plot,
                      color='black', alpha=0.35, jitter=0.15, size=3.5, ax=ax)

        # Configurações de rótulos e eixos limpos
        ax.set_ylabel('F1-score per Drill Unit', fontweight='bold')
        ax.set_xlabel('Detection Architecture (Ultrasonic Mic)', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)

        ax.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine(ax=ax)

        plt.tight_layout()

        nome_arquivo = "boxplot_ultrasonic_todos_modelos_IEEE.pdf"
        caminho_final = os.path.join(caminho_script, nome_arquivo)
        plt.savefig(caminho_final, bbox_inches='tight', pad_inches=0.01)
        plt.savefig(caminho_final.replace(".pdf", ".png"), dpi=300, bbox_inches='tight', pad_inches=0.01)

        print(f"\nBoxplot oficial ultrassônico gerado: {nome_arquivo}")
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    gerar_boxplot_ultrasonic_ieee()