#!/usr/bin/env python3
"""
convert.py — Processamento de áudios e cópia de jams.txt
- Padroniza WAVs mono ou multicanais
- Mantém espelhamento de pastas em data/standardized
- Cria metadata inicial
- Copia jams.txt de cada drill, mesmo que esteja na raiz antes das subpastas
- Cria subpasta única (UUID) para casos de ultrasonic_mics sem subpastas específicas
"""

import os
import shutil
import uuid
import pandas as pd
import soundfile as sf
import librosa
import numpy as np

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/standardized"
METADATA_CSV = "data/metadata/initial_metadata.csv"

MIC_MAPPING = {
    "Tr1": ("common", "ext"),
    "Tr2": ("common", "ext"),
    "Tr3": ("common", "ext"),
    "Tr4": ("common", "int"),
    "Tr5": ("common", "int"),
    "Tr6": ("common", "int"),
}

ULTRASONIC_MAPPING = {
    "ultrasonic_ext": ("ultrasonic", "ext"),
    "ultrasonic_int": ("ultrasonic", "int"),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_channel(y_channel, sr, out_path):
    sf.write(out_path, y_channel, sr)

def process_multichannel(filepath, drill_id, mic_type, position, out_dir, metadata_list):
    """
    Separa canais multicanais e salva individualmente no mesmo espelhamento de pastas.
    """
    y, sr = sf.read(filepath, always_2d=True)
    num_channels = y.shape[1]

    for ch in range(num_channels):
        y_ch = y[:, ch]
        out_name = f"{drill_id}_{mic_type}_ch{ch+1}_{position}.wav"
        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, out_name)
        process_channel(y_ch, sr, out_path)

        metadata_list.append({
            "drill_id": drill_id,
            "mic_type": mic_type,
            "mic_id": f"ch{ch+1}",
            "position": position,
            "sr": sr,
            "filepath_wav": out_path
        })
        print(f"✅ Canal {ch+1}/{num_channels} salvo: {out_path}")

def process_wav(filepath, drill_id, mic_type, position, mic_id, out_dir, metadata_list):
    y, sr = librosa.load(filepath, sr=None, mono=True)
    out_name = f"{drill_id}_{mic_type}_{mic_id}_{position}.wav"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)

    sf.write(out_path, y, sr)

    metadata_list.append({
        "drill_id": drill_id,
        "mic_type": mic_type,
        "mic_id": mic_id,
        "position": position,
        "sr": sr,
        "filepath_wav": out_path
    })
    print(f"✅ Processado: {out_path}")

def main():
    metadata_list = []

    for drill_folder in os.listdir(RAW_DIR):
        drill_folder = 'drill_4mm_18_batch_01_collet_1_07-02-2025'  # REMOVER ESTA LINHA APÓS TESTES
        drill_path = os.path.join(RAW_DIR, drill_folder)
        if not os.path.isdir(drill_path):
            continue

        # Ajuste do drill_id conforme sua convenção
        drill_id = drill_folder.split("_")[2]

        # --------------------------------------------------
        # 📄 Copiar jams.txt da raiz do drill (se existir)
        # --------------------------------------------------
        jams_file_root = os.path.join(drill_path, "jams.txt")
        if os.path.exists(jams_file_root):
            out_dir_root = os.path.join(OUTPUT_DIR, os.path.relpath(drill_path, RAW_DIR))
            ensure_dir(out_dir_root)
            shutil.copy2(jams_file_root, out_dir_root)
            print(f"📄 jams.txt copiado da raiz do drill: {out_dir_root}")

        # --------------------------------------------------
        # 🔁 Processar subpastas e arquivos
        # --------------------------------------------------
        for root, _, files in os.walk(drill_path):
            rel_path = os.path.relpath(root, RAW_DIR)
            base = os.path.basename(root).lower()

            # --------------------------------------------------
            # 📁 Copiar pasta datalogger inteira (espelhada)
            # --------------------------------------------------
            if base == "datalogger":
                dst_datalogger = os.path.join(OUTPUT_DIR, rel_path)
                if not os.path.exists(dst_datalogger):
                    shutil.copytree(root, dst_datalogger)
                    print(f"📁 datalogger copiado: {dst_datalogger}")
                else:
                    print(f"ℹ️ datalogger já existe, pulando: {dst_datalogger}")
                continue  # não processa WAVs aqui

            # --------------------------------------------------
            # 🔧 ultrasonic_mics sem subpastas → UUID único
            # --------------------------------------------------
            if base == "ultrasonic_mics":
                unique_id = str(uuid.uuid4())[:8]
                rel_path = os.path.join(rel_path, unique_id)

            out_dir_root = os.path.join(OUTPUT_DIR, rel_path)

            for file in files:
                if not file.lower().endswith(".wav") or file.startswith("._"):
                    continue

                filepath = os.path.join(root, file)

                # ------------------------------
                # 🔹 Mic comum
                # ------------------------------
                mic_name = [k for k in MIC_MAPPING.keys() if k in file]
                if mic_name:
                    mic_name = mic_name[0]
                    mic_type, position = MIC_MAPPING[mic_name]
                    mic_id = mic_name

                    process_wav(
                        filepath,
                        drill_id,
                        mic_type,
                        position,
                        mic_id,
                        out_dir_root,
                        metadata_list
                    )
                    continue

                # ------------------------------
                # 🔹 Mic ultrassônico
                # ------------------------------
                if "ultrasonic" in root.lower() or "ultrasonic" in file.lower():

                    if "ext" in file.lower():
                        position = "ext"
                    elif "int" in file.lower():
                        position = "int"
                    else:
                        position = "unknown"

                    mic_type = "ultrasonic"
                    mic_id = os.path.splitext(file)[0]

                    process_multichannel(
                        filepath,
                        drill_id,
                        mic_type,
                        position,
                        out_dir_root,
                        metadata_list
                    )
                    continue

                # ------------------------------
                print(f"⚠️ Mic não mapeado, ignorando arquivo: {file}")

        break  # REMOVER ESTA LINHA APÓS TESTES

    # --------------------------------------------------
    # 📊 Salvar metadata
    # --------------------------------------------------
    ensure_dir(os.path.dirname(METADATA_CSV))
    pd.DataFrame(metadata_list).to_csv(METADATA_CSV, index=False)
    print(f"\n📊 Metadata inicial salva em: {METADATA_CSV}")


if __name__ == "__main__":
    main()