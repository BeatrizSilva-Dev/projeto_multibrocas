import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

def gerar_boxplot_comum_ieee():
    try:
        caminho_script = os.path.dirname(os.path.abspath(__file__))

        path_ae = os.path.join(caminho_script, "resultados_autoencoder_comum.csv")
        path_xgb = os.path.join(caminho_script, "resultados_xgboost_comum.csv")
        path_lstm = os.path.join(caminho_script, "resultados_lstm_comum.csv")

        # Carregamento dos DataFrames reais
        df_ae_raw = pd.read_csv(path_ae)
        df_xgb_raw = pd.read_csv(path_xgb)
        df_lstm_raw = pd.read_csv(path_lstm)

        # Extrai o F1 usando as predições REAIS prontas dos CSVs
        def extrair_f1_real_por_broca(df, modelo):
            resultados_f1 = []
            # Mapeamento automático dos cabeçalhos específicos exportados
            label_col = 'label' if 'label' in df.columns else 'label_real'
            pred_col = 'prediction'

            for drill in sorted(df['drill'].unique()):
                df_drill = df[df['drill'] == drill]
                y_true = df_drill[label_col].values
                y_pred = df_drill[pred_col].values

                score = f1_score(y_true, y_pred, zero_division=0)
                resultados_f1.append({'drill': drill, 'f1_score': score, 'modelo': modelo})
            return pd.DataFrame(resultados_f1)

        # 3. Processamento contra bugs matemáticos
        df_f1_lstm = extrair_f1_real_por_broca(df_lstm_raw, 'LSTM-AE')
        df_f1_ae = extrair_f1_real_por_broca(df_ae_raw, 'MLP-AE')
        df_f1_xgb = extrair_f1_real_por_broca(df_xgb_raw, 'XGBoost')

        # 4. Teste de Wilcoxon e Effect Size
        stat, p_value = wilcoxon(df_f1_lstm['f1_score'], df_f1_xgb['f1_score'])
        n = len(df_f1_lstm)
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2*n + 1) / 24)
        z = (stat - mean_w) / (std_w + 1e-9)
        r = z / np.sqrt(n)

        print(f"p-value: {p_value:.6f}")
        print(f"Effect size (r): {abs(r):.4f}")

        # 5. Preparação para plotagem
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

        plt.figure(figsize=(4.0, 3.2))
        sns.set_style("white")

        cores_ieee = ['#084594', '#4292c6', '#9ecae1']

        ax = sns.boxplot(x='modelo', y='f1_score', data=df_plot,
                         palette=cores_ieee,
                         width=0.40, linewidth=1.2, fliersize=0)

        sns.stripplot(x='modelo', y='f1_score', data=df_plot,
                      color='black', alpha=0.4, jitter=0.15, size=3.5)

        plt.ylabel('F1-score per Drill Unit', fontweight='bold')
        plt.xlabel('Detection Architecture (Common Mic)', fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine()

        plt.tight_layout()

        nome_arquivo = "boxplot_comum_todos_modelos_IEEE.pdf"
        caminho_final = os.path.join(caminho_script, nome_arquivo)
        plt.savefig(caminho_final, bbox_inches='tight', pad_inches=0.01)
        print(f"\n[SUCESSO] Boxplot oficial audível gerado: {nome_arquivo}")

        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    gerar_boxplot_comum_ieee()