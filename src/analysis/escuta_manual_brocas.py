import pandas as pd

dados_escuta = [
    {"drill": "drill_4mm_04", "hole": 1, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 2, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 3, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "começo do guizo"},
    {"drill": "drill_4mm_04", "hole": 4, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 5, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 6, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 7, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 8, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 9, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 10, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 11, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 12, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 13, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 14, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 15, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 16, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 17, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 18, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 19, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 20, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 21, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 22, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 23, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 24, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 25, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 26, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 27, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 28, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 29, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 30, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 31, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 32, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 33, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 34, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 35, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 36, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 37, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 38, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 39, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
    {"drill": "drill_4mm_04", "hole": 40, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 41, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 42, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_04", "hole": 43, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 44, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 45, "mic_ultrasonic": "guizo muito forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 46, "mic_ultrasonic": "guizo muito forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_04", "hole": 47, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_04", "hole": 48, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_04", "hole": 49, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
    {"drill": "drill_4mm_04", "hole": 50, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_04", "hole": 51, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_04", "hole": 52, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
]


# BROCA 14 
for h in range(1, 23):
    u, r = "guiso", "guiso"
    if h in [1, 2, 3]: u, r = "furo normal", "furo normal"
    elif h == 4: u, r = "começo do guiso", "começo do guiso"
    elif h in [17, 18, 19, 20]: u, r = "parece radio", "parece radio"
    elif h in [13, 21, 22]: u, r = "guiso + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_14", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 15 
for h in range(1, 46):
    u, r = "parece radio", "parece radio"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "começo do guiso", "começo do guiso"
    elif h == 3: u, r = "guiso fraco", "guiso"
    elif h in [4, 7, 9, 14, 31, 37, 39, 43, 45]: u, r = "guiso + travou", "travou"
    elif h in [5, 6, 8, 10, 11, 12, 13, 15, 16, 17]: u, r = "guiso", "guiso"
    elif h in [32, 33, 34, 35]: u, r = "guiso", "parece radio"
    dados_escuta.append({"drill": "drill_4mm_15", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 16 
for h in range(1, 42): 
    u, r = "chiado", "guizo"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h in [3, 4, 5, 6, 8, 10, 11, 12, 13]: u, r = "guizo", "guizo"
    elif h in [7, 14, 17, 29, 33, 38, 39]: u, r = "travou", "travou"
    elif h in [9, 16, 19, 31, 35, 40, 41]: u, r = "guizo","guizo"
    dados_escuta.append({"drill": "drill_4mm_16", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 17 
for h in range(1, 21):
    u, r = "guizo", "guizo"
    if h in [1, 9, 13, 14, 16, 20]: u, r = "travou", "travou"
    elif h == 2: u, r = "furo normal", "furo normal"
    elif h == 3: u, r = "começo do guizo", "começo do guizo"
    elif h in [18, 19]: u, r = "parece radio", "parece rádio"
    dados_escuta.append({"drill": "drill_4mm_17", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 18 
for h in range(1, 83):
    u, r = "chiado", "parece chiado"
    if h in [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]: u, r = "furo normal", "furo normal"
    elif h == 11: u, r = "chiado", "furo normal"
    elif h in [3, 45, 52, 68, 75, 78, 80, 82]: u, r = "travou", "travou"
    elif h == 12: u, r = "guizo", "começo do guizo"
    elif h in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]: u, r = "guizo", "guizo"
    elif h in [29, 30, 33, 35, 36]: u, r = "parece chiado", "guizo"
    elif h in [31, 32, 34]: u, r = "guizo", "guizo"
    dados_escuta.append({"drill": "drill_4mm_18", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 19 
for h in range(1, 55):
    u, r = "chiado", "guizo mais forte"
    if h in [1, 2]: u, r = "normal", "normal"
    elif h in [3, 4]: u, r = "guizo", "guizo"
    elif h in [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]: u, r = "guizo mais forte", "guizo mais forte"
    elif h in [10, 48, 50, 53, 54]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_19", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 20 
for h in range(1, 37):
    u, r = "parece radio", "parece rádio"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "guizo", "começo do guiso"
    elif h in [3, 4, 6, 7, 8, 9, 10]: u, r = "guizo", "guizo"
    elif h in [5, 12, 24, 30, 32, 35, 36]: u, r = "travou", "travou"
    elif h in [11, 13, 14]: u, r = "parece radio", "guizo"
    dados_escuta.append({"drill": "drill_4mm_20", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 21 
for h in range(1, 30):
    u, r = "parece radio", "parece radio"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h in [3, 4]: u, r = "começo do guiso", "começo do guiso"
    elif h in [5, 6, 7, 8, 9]: u, r = "guiso", "parece radio"
    elif h in [13, 22, 24, 26, 29]: u, r = "travou", "travou"
    elif h == 28: u, r = "parece radio", "som diferente e alto"
    dados_escuta.append({"drill": "drill_4mm_21", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 22 
for h in range(1, 26):
    u, r = "guizo", "guizo"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "começo do guizo", "começo do guiso"
    elif h in [19, 20, 21]: u, r = "parece radio", "guizo" if h==19 else "parece radio"
    elif h in [22, 23, 24, 25]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_22", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 23 
for h in range(1, 42):
    u, r = "parece radio", "parece radio"
    if h in [1, 2]: u, r = "normal", "furo normal"
    elif h == 3: u, r = "começo do guiso", "começo do guizo"
    elif h in [4, 5]: u, r = "guiso", "guizo"
    elif h in [26, 28, 33, 34, 39, 41]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_23", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# COMPILAÇÃO E CRIAÇÃO DAS COLUNAS INTELIGENTES 
df_completo = pd.DataFrame(dados_escuta)

total_furos_dict = df_completo.groupby('drill')['hole'].max().to_dict()
df_completo['total_holes_drill'] = df_completo['drill'].map(total_furos_dict)
df_completo['duration_pct'] = ((df_completo['hole'] / df_completo['total_holes_drill']) * 100).round(2)

def classificar_severidade(row):
    u = str(row['mic_ultrasonic']).lower()
    r = str(row['mic_reg']).lower()
    pct = float(row['duration_pct'])

    # Nível 3: Travamento Físico
    if 'travou' in u or 'travou' in r:
        return 3

    # Nível 2: Pré-Falha Severa (Chiado Insuportável OU Guizo Alto/Forte no final da vida)
    if ('radio' in u or 'rádio' in r or 'chiado' in u or 'chuva' in u or 'alto' in r or
            (('guizo' in u or 'guiso' in u or 'forte' in u) and pct >= 50.0)):
        return 2

    # Nível 1: Anomalia Leve (Guizo/Guiso real de início de desgaste na primeira metade da vida)
    if ('guizo' in u or 'guiso' in u) and pct < 50.0:
        return 1

    # Nível 0: Nominal Estável
    return 0

df_completo['human_severity_score'] = df_completo.apply(classificar_severidade, axis=1)

# Salvar planilha mestre unificada
df_completo = df_completo[['drill', 'hole', 'duration_pct', 'mic_ultrasonic', 'mic_reg', 'human_severity_score']]
df_completo.to_csv("escuta_manual_projeto_completo.csv", index=False)
print(f"[SUCESSO TOTAL] Mapeamento concluído. {len(df_completo)} linhas salvas com indexação temporal %!")