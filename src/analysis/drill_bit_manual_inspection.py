import os
import pandas as pd

listening_data = [
    {"drill": "drill_4mm_04", "hole": 1, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 2, "mic_ultrasonic": "normal", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 3, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "onset of rattle"},
    {"drill": "drill_4mm_04", "hole": 4, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 5, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 6, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 7, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 8, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 9, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 10, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 11, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 12, "mic_ultrasonic": "rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 13, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 14, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 15, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 16, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 17, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 18, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 19, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 20, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "normal"},
    {"drill": "drill_4mm_04", "hole": 21, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 22, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 23, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 24, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 25, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 26, "mic_ultrasonic": "rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 27, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 28, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 29, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 30, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 31, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 32, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 33, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 34, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 35, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 36, "mic_ultrasonic": "strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 37, "mic_ultrasonic": "slightly stronger rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 38, "mic_ultrasonic": "strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 39, "mic_ultrasonic": "rattle + jammed", "mic_reg": "rattle + jammed"},
    {"drill": "drill_4mm_04", "hole": 40, "mic_ultrasonic": "rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 41, "mic_ultrasonic": "weak rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 42, "mic_ultrasonic": "strong rattle at the end", "mic_reg": "weak rattle"},
    {"drill": "drill_4mm_04", "hole": 43, "mic_ultrasonic": "strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 44, "mic_ultrasonic": "strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 45, "mic_ultrasonic": "very strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 46, "mic_ultrasonic": "very strong rattle at the end", "mic_reg": "rattle"},
    {"drill": "drill_4mm_04", "hole": 47, "mic_ultrasonic": "sounds like radio noise", "mic_reg": "sounds like radio noise"},
    {"drill": "drill_4mm_04", "hole": 48, "mic_ultrasonic": "sounds like radio noise", "mic_reg": "sounds like radio noise"},
    {"drill": "drill_4mm_04", "hole": 49, "mic_ultrasonic": "rattle + jammed", "mic_reg": "rattle + jammed"},
    {"drill": "drill_4mm_04", "hole": 50, "mic_ultrasonic": "sounds like radio noise", "mic_reg": "sounds like radio noise"},
    {"drill": "drill_4mm_04", "hole": 51, "mic_ultrasonic": "sounds like radio noise", "mic_reg": "sounds like radio noise"},
    {"drill": "drill_4mm_04", "hole": 52, "mic_ultrasonic": "rattle + jammed", "mic_reg": "rattle + jammed"},
]

for h in range(1, 108):
    u, r = "hissing", "rattle"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif h == 3: u, r = "onset of rattle at the end of audio", "normal hole"
    elif h in [4, 5, 6, 7, 9, 10, 11, 12, 19]: u, r = "weak rattle at the end", "weak rattle" if h != 4 else "onset of rattle"
    elif h in [8] or (13 <= h <= 30): u, r = "stronger rattle", "weak rattle" if h in [13,14,15,16,17,18,20,21] else "rattle"
    elif 31 <= h <= 41: u, r = "rattle at the end", "rattle"
    elif (42 <= h <= 55) or h in [57, 58, 61]: u, r = "sounds like radio static", "rattle"
    elif h in [56, 59, 60] or (62 <= h <= 71): u, r = "stronger hissing", "rattle" if h in [56, 59, 60] else "strong rattle"
    elif (72 <= h <= 102) and h not in [93, 96]: u, r = "unbearable hissing", "strong rattle" if h in [72,73,74,75,76,77,78,79,80,90,92] else "sounds like radio static"
    elif h in [103, 104, 105, 106]: u, r = "rain noise", "rattle" if h == 104 else "sounds like radio static"
    elif h in [93, 96, 107]: u, r = "jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_05", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 27):
    u, r = "sounds like radio static (unbearable from 17 onward)", "sounds like radio static"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif 3 <= h <= 7: u, r = "onset of rattle", "onset of rattle" if h == 3 else ("weak rattle" if h in [4, 5] else "rattle")
    elif h in [22, 25, 26]: u, r = "rattle + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_06", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 34):
    u, r = "sounds like drizzle / untuned radio static", "sounds like drizzle / untuned radio static"
    if h in [1, 2]: u, r = "normal sound", "normal hole"
    elif h == 3: u, r = "onset of weak rattle", "onset of rattle"
    elif h in [5, 6]: u, r = "weak rattle at the end", "rattle"
    elif 7 <= h <= 10: u, r = "stronger rattle", "rattle"
    elif 11 <= h <= 18: u, r = "strange and strong rattle", "strange and strong rattle"
    elif 19 <= h <= 20: u, r = "sounds like drizzle / untuned radio static", "strange and strong rattle"
    elif h == 31: u, r = "sounds like drizzle / untuned radio static + jammed", "jammed"
    elif h in [4, 32, 33]: u, r = "rattle + jammed", "rattle + jammed" if h == 4 else "jammed"
    listening_data.append({"drill": "drill_4mm_07", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 23):
    u, r = "sounds like light rain", "strong rattle"
    if h in [1, 2, 3]: u, r = "normal hole", "normal hole"
    elif h in [4, 5]: u, r = "onset of rattle", "normal hole" if h == 4 else "onset of rattle"
    elif h in [6, 8, 17]: u, r = "weak rattle at the end", "weak rattle"
    elif h == 14 or h == 15: u, r = "sounds like light rain", "weak rattle"
    elif h == 9: u, r = "weak rattle at the end", "strong rattle"
    elif h in [10, 11]: u, r = "strong rattle at the end", "strong rattle"
    elif h in [12, 13, 18, 21]: u, r = "sounds like light rain", "strong rattle"
    elif h == 20: u, r = "STRANGE AND LOUD NOISE", "unbearable and strange noise"
    elif h in [7, 16, 19, 22]: u, r = "weak rattle + jammed" if h==7 else ("weak rattle + JAMMED" if h in [16,22] else "sounds like light rain + JAMMED"), "jammed"
    listening_data.append({"drill": "drill_4mm_08", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 53):
    u, r = "sounds like rain / radio static", "sounds like radio static"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif h in [3, 4]: u, r = "onset of rattle", "onset of rattle" if h==3 else "rattle"
    elif 5 <= h <= 8: u, r = "strong rattle at the end", "rattle"
    elif h in [44, 50, 52]: u, r = "sounds like rain + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_09", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 61):
    u, r = "hissing", "rattle"
    if h == 1: u, r = "normal hole", "normal hole"
    elif h in [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]: u, r = "rattle", "rattle"
    elif h in [2, 9, 14, 50, 58, 60]: u, r = "jammed", "jammed"
    elif h in [51, 59]: u, r = "not jammed, jamming not audible but confirmed by logs", "not jammed, jamming not audible but confirmed by logs"
    listening_data.append({"drill": "drill_4mm_10", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 21):
    u, r = "sounds like radio static", "sounds like radio static"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif h == 3: u, r = "onset of rattle", "onset of rattle"
    elif 4 <= h <= 10: u, r = "rattle", "rattle"
    elif h in [16, 19, 20]: u, r = "static + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_11", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 19):
    u, r = "rattle", "rattle"
    if h in [1, 2, 3, 4, 5]: u, r = "normal hole", "normal hole"
    elif h == 6: u, r = "onset of rattle", "onset of rattle"
    elif h == 7: u, r = "rattle", "weak rattle"
    elif 13 <= h <= 15: u, r = "strange rattle", "rattle" if h==13 else "strange rattle"
    elif h in [16, 17, 18]: u, r = "strange rattle + jammed" if h==16 else "rattle + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_12", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 20):
    u, r = "rattle", "rattle"
    if h == 1: u, r = "normal hole", "normal hole"
    elif h == 2: u, r = "onset of rattle", "onset of rattle"
    elif h in [3, 4, 5]: u, r = "weak rattle", "rattle"
    elif h in [15, 17, 19]: u, r = "rattle + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_13", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 23):
    u, r = "rattle", "rattle"
    if h in [1, 2, 3]: u, r = "normal hole", "normal hole"
    elif h == 4: u, r = "onset of rattle", "onset of rattle"
    elif h in [17, 18, 19, 20]: u, r = "sounds like radio static", "sounds like radio static"
    elif h in [13, 21, 22]: u, r = "rattle + jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_14", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 46):
    u, r = "sounds like radio static", "sounds like radio static"
    if h == 1: u, r = "normal hole", "normal hole"
    elif h == 2: u, r = "onset of rattle", "onset of rattle"
    elif h == 3: u, r = "weak rattle", "rattle"
    elif h in [4, 7, 9, 14, 31, 37, 39, 43, 45]: u, r = "rattle + jammed", "jammed"
    elif h in [5, 6, 8, 10, 11, 12, 13, 15, 16, 17]: u, r = "rattle", "rattle"
    elif h in [32, 33, 34, 35]: u, r = "rattle", "sounds like radio static"
    listening_data.append({"drill": "drill_4mm_15", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 42):
    u, r = "hissing", "rattle"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif h in [3, 4, 5, 6, 8, 10, 11, 12, 13]: u, r = "rattle", "rattle"
    elif h in [7, 14, 17, 29, 33, 38, 39]: u, r = "jammed", "jammed"
    elif h in [9, 16, 19, 31, 35, 40, 41]: u, r = "rattle", "rattle"
    listening_data.append({"drill": "drill_4mm_16", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 21):
    u, r = "rattle", "rattle"
    if h in [1, 9, 13, 14, 16, 20]: u, r = "jammed", "jammed"
    elif h == 2: u, r = "normal hole", "normal hole"
    elif h == 3: u, r = "onset of rattle", "onset of rattle"
    elif h in [18, 19]: u, r = "sounds like radio static", "sounds like radio static"
    listening_data.append({"drill": "drill_4mm_17", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 83):
    u, r = "hissing", "sounds like hissing"
    if h in [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14]: u, r = "normal hole", "normal hole"
    elif h == 11: u, r = "hissing", "normal hole"
    elif h in [3, 45, 52, 68, 75, 78, 80, 82]: u, r = "jammed", "jammed"
    elif h == 12: u, r = "rattle", "onset of rattle"
    elif h in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]: u, r = "rattle", "rattle"
    elif h in [29, 30, 33, 35, 36]: u, r = "sounds like hissing", "rattle"
    elif h in [31, 32, 34]: u, r = "rattle", "rattle"
    listening_data.append({"drill": "drill_4mm_18", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 55):
    u, r = "hissing", "stronger rattle"
    if h in [1, 2]: u, r = "normal", "normal"
    elif h in [3, 4]: u, r = "rattle", "rattle"
    elif h in [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]: u, r = "stronger rattle", "stronger rattle"
    elif h in [10, 48, 50, 53, 54]: u, r = "jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_19", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 37):
    u, r = "sounds like radio static", "sounds like radio static"
    if h == 1: u, r = "normal hole", "normal hole"
    elif h == 2: u, r = "rattle", "onset of rattle"
    elif h in [3, 4, 6, 7, 8, 9, 10]: u, r = "rattle", "rattle"
    elif h in [5, 12, 24, 30, 32, 35, 36]: u, r = "jammed", "jammed"
    elif h in [11, 13, 14]: u, r = "sounds like radio static", "rattle"
    listening_data.append({"drill": "drill_4mm_20", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 30):
    u, r = "sounds like radio static", "sounds like radio static"
    if h in [1, 2]: u, r = "normal hole", "normal hole"
    elif h in [3, 4]: u, r = "onset of rattle", "onset of rattle"
    elif h in [5, 6, 7, 8, 9]: u, r = "rattle", "sounds like radio static"
    elif h in [13, 22, 24, 26, 29]: u, r = "jammed", "jammed"
    elif h == 28: u, r = "sounds like radio static", "different and loud sound"
    listening_data.append({"drill": "drill_4mm_21", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 26):
    u, r = "rattle", "rattle"
    if h == 1: u, r = "normal hole", "normal hole"
    elif h == 2: u, r = "onset of rattle", "onset of rattle"
    elif h in [19, 20, 21]: u, r = "sounds like radio static", "rattle" if h==19 else "sounds like radio static"
    elif h in [22, 23, 24, 25]: u, r = "jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_22", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

for h in range(1, 42):
    u, r = "sounds like radio static", "sounds like radio static"
    if h in [1, 2]: u, r = "normal", "normal hole"
    elif h == 3: u, r = "onset of rattle", "onset of rattle"
    elif h in [4, 5]: u, r = "rattle", "rattle"
    elif h in [26, 28, 33, 34, 39, 41]: u, r = "jammed", "jammed"
    listening_data.append({"drill": "drill_4mm_23", "hole": h, "mic_ultrasonic": u, "mic_reg": r})

df_complete = pd.DataFrame(listening_data)

article_mapping = {f"drill_4mm_{str(i).zfill(2)}": f"drill_4mm_{str(i-3).zfill(2)}" for i in range(4, 24)}
df_complete['drill'] = df_complete['drill'].map(article_mapping)

total_holes_dict = df_complete.groupby('drill')['hole'].max().to_dict()
df_complete['total_holes_drill'] = df_complete['drill'].map(total_holes_dict)
df_complete['duration_pct'] = ((df_complete['hole'] / df_complete['total_holes_drill']) * 100).round(2)

def classify_severity(row):
    u = str(row['mic_ultrasonic']).lower()
    r = str(row['mic_reg']).lower()
    pct = float(row['duration_pct'])

    if 'jammed' in u or 'jammed' in r:
        return 3

    if ('radio' in u or 'radio' in r or 'hissing' in u or 'rain' in u or 'loud' in r or
            (('rattle' in u or 'rattle' in r or 'strong' in u) and pct >= 50.0)):
        return 2

    if ('rattle' in u or 'rattle' in r) and pct < 50.0:
        return 1

    return 0

df_complete['human_severity_score'] = df_complete.apply(classify_severity, axis=1)

df_complete = df_complete[['drill', 'hole', 'duration_pct', 'mic_ultrasonic', 'mic_reg', 'human_severity_score']]

df_complete.to_csv("manual_listening_complete_project.csv", index=False)

print(f"Masked drills list saved: {sorted(df_complete['drill'].unique())}")
print(f"{len(df_complete)} rows saved with temporal index %!")