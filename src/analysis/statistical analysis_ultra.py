import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon


script_path = os.path.dirname(os.path.abspath(__file__))

path_ae = os.path.join(script_path, "results_autoencoder_ultrasonic.csv")
path_xgb = os.path.join(script_path, "results_xgboost_ultrasonic.csv")
path_lstm = os.path.join(script_path, "results_lstm_ultrasonic.csv")


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
        f1_results.append({'drill': drill, 'f1_score': score, 'model': model})
        return pd.DataFrame(f1_results)

df_f1_lstm = extract_true_f1_per_drill(df_lstm_raw, 'LSTM-AE')
df_f1_ae = extract_true_f1_per_drill(df_ae_raw, 'MLP-AE')
df_f1_xgb = extract_true_f1_per_drill(df_xgb_raw, 'XGBoost')

res_wilc = wilcoxon(df_f1_lstm['f1_score'], df_f1_xgb['f1_score'], method='approx')
p_value = res_wilc.pvalue
z_stat = res_wilc.zstatistic
n = len(df_f1_lstm)
r = z_stat / np.sqrt(n) 

print(f"Wilcoxon p-value: {p_value:.6f}")
print(f"Effect Size (r): {abs(r):.4f}")