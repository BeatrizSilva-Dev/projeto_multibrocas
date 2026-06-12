import pandas as pd

dados_escuta = [
    #BROCA 01
    {"drill": "drill_4mm_01", "hole": 1, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_01", "hole": 2, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_01", "hole": 3, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "começo do guizo"},
    {"drill": "drill_4mm_01", "hole": 4, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 5, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 6, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 7, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 8, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 9, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 10, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 11, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 12, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 13, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 14, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 15, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 16, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 17, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 18, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 19, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 20, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "normal"},
    {"drill": "drill_4mm_01", "hole": 21, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 22, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 23, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 24, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 25, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 26, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 27, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 28, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 29, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 30, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 31, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 32, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 33, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 34, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 35, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 36, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 37, "mic_ultrasonic": "guizo um pouco mais forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 38, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 39, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
    {"drill": "drill_4mm_01", "hole": 40, "mic_ultrasonic": "guizo no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 41, "mic_ultrasonic": "guizo fraco no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 42, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo fraco"},
    {"drill": "drill_4mm_01", "hole": 43, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 44, "mic_ultrasonic": "guizo forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 45, "mic_ultrasonic": "guizo muito forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 46, "mic_ultrasonic": "guizo muito forte no final", "mic_reg": "guizo"},
    {"drill": "drill_4mm_01", "hole": 47, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_01", "hole": 48, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_01", "hole": 49, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
    {"drill": "drill_4mm_01", "hole": 50, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_01", "hole": 51, "mic_ultrasonic": "parece rádio", "mic_reg": "parece rádio"},
    {"drill": "drill_4mm_01", "hole": 52, "mic_ultrasonic": "guizo + travou", "mic_reg": "guizo + travou"},
]

# BROCA 02
for h in range(1, 108):
    u, r = "chiado", "guiso"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h == 3: u, r = "começo do guiso no final do áudio", "furo normal"
    elif h in [4, 5, 6, 7, 9, 10, 11, 12, 19]: u, r = "guiso fraco no final", "guiso fraco" if h != 4 else "começo do guiso"
    elif h in [8] or (13 <= h <= 30): u, r = "guiso mais forte", "guiso fraco" if h in [13,14,15,16,17,18,20,21] else "guiso"
    elif 31 <= h <= 41: u, r = "guiso no final", "guiso"
    elif (42 <= h <= 55) or h in [57, 58, 61]: u, r = "parece chiado de rádio", "guiso"
    elif h in [56, 59, 60] or (62 <= h <= 71): u, r = "chiado mais forte", "guiso" if h == 56 or h in [59, 60] else "guiso forte"
    elif (72 <= h <= 102) and h not in [93, 96]: u, r = "chiado insuportável de ouvir", "guiso forte" if h in [72,73,74,75,76,77,78,79,80,90,92] else "parece rádio"
    elif h in [103, 104, 105, 106]: u, r = "chuva", "guiso" if h == 104 else "parece rádio"
    elif h in [93, 96, 107]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_02", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 03
for h in range(1, 27):
    u, r = "parece radio(insuportável a partir do 17)", "parece radio"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif 3 <= h <= 7: u, r = "começo do guiso", "começo do guiso" if h == 3 else ("guiso fraco" if h in [4, 5] else "guiso")
    elif h in [22, 25, 26]: u, r = "guiso + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_03", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 04
for h in range(1, 34):
    u, r = "parece chuvisco/radio dessintonizada", "parece chuvisco/radio dessintonizada"
    if h in [1, 2]: u, r = "som normal", "furo normal"
    elif h == 3: u, r = "começo do guiso fraco", "começo do guiso"
    elif h in [5, 6]: u, r = "guiso fraco no final", "guiso"
    elif 7 <= h <= 10: u, r = "guiso mais forte", "guiso"
    elif 11 <= h <= 18: u, r = "guiso estranho e forte", "guiso estranho e forte"
    elif 19 <= h <= 20: u, r = "parece chuvisco/radio dessintonizada", "guiso estranho e forte"
    elif h == 31: u, r = "parece chuvisco/radio dessintonizada + travou", "travou"
    elif h in [4, 32, 33]: u, r = "guiso + travou", "guiso + travou" if h == 4 else "travou"
    dados_escuta.append({"drill": "drill_4mm_04", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 05 
for h in range(1, 23):
    u, r = "parece chuva fraca", "guiso forte"
    if h in [1, 2, 3]: u, r = "furo normal", "furo normal"
    elif h in [4, 5]: u, r = "começo guiso", "furo normal" if h == 4 else "começo do guiso"
    elif h in [6, 8, 17]: u, r = "guiso fraco no final", "guiso fraco"
    elif h == 14 or h == 15: u, r = "parece chuva fraca", "guiso fraco"
    elif h == 9: u, r = "guiso fraco no final", "guiso forte"
    elif h in [10, 11]: u, r = "guiso forte no final", "guiso forte"
    elif h in [12, 13, 18, 21]: u, r = "parece chuva fraca", "guiso forte"
    elif h == 20: u, r = "BARULHO ESTRANHO E ALTO", "barulho insuportável e estranho"
    elif h in [7, 16, 19, 22]: u, r = "guiso fraco + travou" if h==7 else ("guiso fraco + TRAVOU" if h in [16,22] else "parece chuva fraca + TRAVOU"), "travou"
    dados_escuta.append({"drill": "drill_4mm_05", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 06
for h in range(1, 53):
    u, r = "parece chuva/radio", "parece rádio"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h in [3, 4]: u, r = "começo do guiso", "começo do guiso" if h==3 else "guiso"
    elif 5 <= h <= 8: u, r = "guiso forte no final", "guiso"
    elif h in [44, 50, 52]: u, r = "parece chuva + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_06", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 07
for h in range(1, 61):
    u, r = "chiado", "guizo"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h in [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]: u, r = "guizo", "guizo"
    elif h in [2, 9, 14, 50, 58, 60]: u, r = "travou", "travou"
    elif h in [51, 59]: u, r = "não travou, não ouvir travamento mas pelo jam é esse furo", "não travou, não ouvir travamento mas pelo jam é esse furo"
    dados_escuta.append({"drill": "drill_4mm_07", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 08 
for h in range(1, 21):
    u, r = "parece chiado de rádio", "parece chiado de rádio"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h == 3: u, r = "começo do guiso", "começo do guiso"
    elif 4 <= h <= 10: u, r = "guiso", "guiso"
    elif h in [16, 19, 20]: u, r = "chiado + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_08", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 09
for h in range(1, 19):
    u, r = "guiso", "guiso"
    if h in [1, 2, 3, 4, 5]: u, r = "furo normal", "furo normal"
    elif h == 6: u, r = "começo do guiso", "começo do guiso"
    elif h == 7: u, r = "guiso", "guiso fraco"
    elif 13 <= h <= 15: u, r = "guiso estranho", "guiso" if h==13 else "guiso estranho"
    elif h in [16, 17, 18]: u, r = "guiso estranho + travou" if h==16 else "guiso + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_09", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 10
for h in range(1, 20):
    u, r = "guiso", "guiso"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "começo do guiso", "começo do guiso"
    elif h in [3, 4, 5]: u, r = "guiso fraco", "guiso"
    elif h in [15, 17, 19]: u, r = "guiso + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_10", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 11 
for h in range(1, 23):
    u, r = "guiso", "guiso"
    if h in [1, 2, 3]: u, r = "furo normal", "furo normal"
    elif h == 4: u, r = "começo do guiso", "começo do guiso"
    elif h in [17, 18, 19, 20]: u, r = "parece radio", "parece radio"
    elif h in [13, 21, 22]: u, r = "guiso + travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_11", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 12
for h in range(1, 46):
    u, r = "parece radio", "parece radio"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "começo do guiso", "começo do guiso"
    elif h == 3: u, r = "guiso fraco", "guiso"
    elif h in [4, 7, 9, 14, 31, 37, 39, 43, 45]: u, r = "guiso + travou", "travou"
    elif h in [5, 6, 8, 10, 11, 12, 13, 15, 16, 17]: u, r = "guiso", "guiso"
    elif h in [32, 33, 34, 35]: u, r = "guiso", "parece radio"
    dados_escuta.append({"drill": "drill_4mm_12", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 13
for h in range(1, 42):
    u, r = "chiado", "guizo"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h in [3, 4, 5, 6, 8, 10, 11, 12, 13]: u, r = "guizo", "guizo"
    elif h in [7, 14, 17, 29, 33, 38, 39]: u, r = "travou", "travou"
    elif h in [9, 16, 19, 31, 35, 40, 41]: u, r = "guizo", "guizo"
    dados_escuta.append({"drill": "drill_4mm_13", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 14
for h in range(1, 21):
    u, r = "guizo", "guizo"
    if h in [1, 9, 13, 14, 16, 20]: u, r = "travou", "travou"
    elif h == 2: u, r = "furo normal", "furo normal"
    elif h == 3: u, r = "começo do guizo", "começo do guizo"
    elif h in [18, 19]: u, r = "parece radio", "parece rádio"
    dados_escuta.append({"drill": "drill_4mm_14", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 15
for h in range(1, 83):
    u, r = "chiado", "parece chiado"
    if h in [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]: u, r = "furo normal", "furo normal"
    elif h == 11: u, r = "chiado", "furo normal"
    elif h in [3, 45, 52, 68, 75, 78, 80, 82]: u, r = "travou", "travou"
    elif h == 12: u, r = "guizo", "começo do guizo"
    elif h in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]: u, r = "guizo", "guizo"
    elif h in [29, 30, 33, 35, 36]: u, r = "parece chiado", "guizo"
    elif h in [31, 32, 34]: u, r = "guizo", "guizo"
    dados_escuta.append({"drill": "drill_4mm_15", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 16
for h in range(1, 55):
    u, r = "chiado", "guizo mais forte"
    if h in [1, 2]: u, r = "normal", "normal"
    elif h in [3, 4]: u, r = "guizo", "guizo"
    elif h in [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]: u, r = "guizo mais forte", "guizo mais forte"
    elif h in [10, 48, 50, 53, 54]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_16", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 17
for h in range(1, 37):
    u, r = "parece radio", "parece rádio"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "guizo", "começo do guiso"
    elif h in [3, 4, 6, 7, 8, 9, 10]: u, r = "guizo", "guizo"
    elif h in [5, 12, 24, 30, 32, 35, 36]: u, r = "travou", "travou"
    elif h in [11, 13, 14]: u, r = "parece radio", "guizo"
    dados_escuta.append({"drill": "drill_4mm_17", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 18 
for h in range(1, 30):
    u, r = "parece radio", "parece radio"
    if h in [1, 2]: u, r = "furo normal", "furo normal"
    elif h in [3, 4]: u, r = "começo do guiso", "começo do guiso"
    elif h in [5, 6, 7, 8, 9]: u, r = "guiso", "parece radio"
    elif h in [13, 22, 24, 26, 29]: u, r = "travou", "travou"
    elif h == 28: u, r = "parece radio", "som diferente e alto"
    dados_escuta.append({"drill": "drill_4mm_18", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 19
for h in range(1, 26):
    u, r = "guizo", "guizo"
    if h == 1: u, r = "furo normal", "furo normal"
    elif h == 2: u, r = "começo do guizo", "começo do guiso"
    elif h in [19, 20, 21]: u, r = "parece radio", "guizo" if h==19 else "parece radio"
    elif h in [22, 23, 24, 25]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_19", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

# BROCA 20
for h in range(1, 42):
    u, r = "parece radio", "parece radio"
    if h in [1, 2]: u, r = "normal", "furo normal"
    elif h == 3: u, r = "começo do guiso", "começo do guizo"
    elif h in [4, 5]: u, r = "guiso", "guizo"
    elif h in [26, 28, 33, 34, 39, 41]: u, r = "travou", "travou"
    dados_escuta.append({"drill": "drill_4mm_20", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

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

df_completo = df_completo[['drill', 'hole', 'duration_pct', 'mic_ultrasonic', 'mic_reg', 'human_severity_score']]
df_completo.to_csv("escuta_manual_projeto_completo.csv", index=False)
print(f"[SUCESSO TOTAL] Mapeamento concluído. {len(df_completo)} linhas salvas com indexação temporal %!")