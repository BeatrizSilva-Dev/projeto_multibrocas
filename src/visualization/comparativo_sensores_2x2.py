import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROOT_PATH = os.path.join(PROJ_ROOT, "data", "segmented", "drill_4mm_06_batch_00_collet_1_30-01-2025")
CANAL = "Tr4"

def buscar_audio(raiz, num_furo, canal, tipo_pasta):
    padrao = f"hole{str(num_furo).zfill(5)}.wav"
    for root, _, files in os.walk(raiz):
        if tipo_pasta in root.lower():
            for f in files:
                if padrao in f and canal in f:
                    return os.path.join(root, f)
    return None

files_to_plot = [
    buscar_audio(ROOT_PATH, 1, CANAL, "reg_mics"),       # (a) Audible - Normal
    buscar_audio(ROOT_PATH, 24, CANAL, "reg_mics"),      # (b) Audible - Wear
    buscar_audio(ROOT_PATH, 1, CANAL, "ultrasonic"),     # (c) Ultrasonic - Normal
    buscar_audio(ROOT_PATH, 24, CANAL, "ultrasonic")     # (d) Ultrasonic - Wear
]

if None in files_to_plot:
    print("ERROR: One or more audio files were not located. Check folder names.")
else:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.labelweight': 'bold',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))

    labels = ['(a) Audible - Hole 1', '(b) Audible - Hole 24',
              '(c) Ultrasonic - Hole 1', '(d) Ultrasonic - Hole 24']

    img = None
    for i, path in enumerate(files_to_plot):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        y, sr = librosa.load(path, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096)), ref=np.max)

        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear',
                                       ax=ax, cmap='magma', vmin=-80, vmax=0)

        ax.set_title(labels[i], fontsize=10, pad=6)

        if row == 0:  # Linha do Audível
            ax.set_ylim(0, 20000)
            ax.set_ylabel('Frequency (Hz)')
        else:         # Linha do Ultrassom
            ax.set_ylim(0, 48000)
            ax.set_ylabel('Frequency (Hz)')

        if col == 1:
            ax.set_ylabel('')  

        if row == 0:
            ax.set_xlabel('')  
        else:
            ax.set_xlabel('Time (s)')

    plt.subplots_adjust(left=0.11, right=0.88, top=0.92, bottom=0.11, hspace=0.28, wspace=0.22)

    cbar_ax = fig.add_axes([0.90, 0.11, 0.02, 0.81])

    cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
    cbar.set_label('Intensity (dB)', weight='bold')
    cbar.ax.tick_params(labelsize=9)

    output_filename = "comparativo_sensores_2x2.pdf"
    plt.savefig(output_filename, dpi=400, bbox_inches='tight')
    plt.close()

    print(f"Gráfico salvo: {output_filename}")