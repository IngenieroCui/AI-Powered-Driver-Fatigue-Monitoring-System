import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

"""
Preprocesado para el dataset global de somnolencia (drowsy).

Objetivo:
- Crear un dataset de rostro completo etiquetado como:
    - "alert"  (conductor en estado normal / sin somnolencia)
    - "drowsy" (conductor somnoliento, ojos cerrados prolongados o bostezando)
- Salida compatible con train_drowsy.py:

    data/drowsy/
        train/
            alert/
            drowsy/
        val/
            alert/
            drowsy/

Suposiciones:
- Ya hemos generado datasets intermedios:
    - data/yawdd/train/{yawn, no_yawn}
    - data/yawdd/val/{yawn, no_yawn}
    - data/eyes_combined/train/{open, closed}
    - data/eyes_combined/val/{open, closed}
- Usaremos principalmente YAWDD como fuente de "drowsy" (frames con yawn)
  y "alert" (no_yawn), mezclado con algunas muestras de ojos abiertos/cerrados
  para equilibrar.

Si en el futuro tienes otra fuente (p.ej. otro dataset de caras), puedes
extender las rutas de origen en las constantes de abajo.
"""

# =============================
# CONFIGURACIÓN DE ORIGEN
# =============================

# Directorios de origen ya preprocesados
YAWDD_TRAIN = Path("data/yawdd/train")
YAWDD_VAL = Path("data/yawdd/val")

EYES_TRAIN = Path("data/eyes_combined/train")
EYES_VAL = Path("data/eyes_combined/val")

# Directorio de salida
DROWSY_ROOT = Path("data/drowsy")

# Proporción train/val ya viene dada por los splits de origen,
# aquí solo mezclamos y copiamos.

# Para no sobrecargar el dataset, podemos hacer un muestreo máximo
MAX_SAMPLES_PER_CLASS_TRAIN = 6000
MAX_SAMPLES_PER_CLASS_VAL = 2000


def ensure_dirs():
    """Crear estructura de carpetas de salida."""
    for split in ["train", "val"]:
        for label in ["alert", "drowsy"]:
            out_dir = DROWSY_ROOT / split / label
            out_dir.mkdir(parents=True, exist_ok=True)


def collect_images_from_folder(folder: Path, label_name: str):
    """Devuelve una lista de rutas de imagen en una carpeta concreta.

    label_name es solo informativo (para logs), no se usa como clase aquí.
    """
    if not folder.is_dir():
        return []

    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in folder.glob("*") if p.suffix.lower() in exts]
    return imgs


def build_drowsy_split(split: str):
    """Construye el split train o val para el dataset drowsy.

    split: "train" o "val".
    """
    print(f"\n=== Construyendo split {split} para data/drowsy ===")

    if split == "train":
        yawdd_root = YAWDD_TRAIN
        eyes_root = EYES_TRAIN
        max_per_class = MAX_SAMPLES_PER_CLASS_TRAIN
    else:
        yawdd_root = YAWDD_VAL
        eyes_root = EYES_VAL
        max_per_class = MAX_SAMPLES_PER_CLASS_VAL

    # ----------------------
    # Clase DROWSY
    # ----------------------
    # Usamos principalmente frames de YAWDD con bostezo (yawn)
    drowsy_sources = []

    yawdd_yawn_dir = yawdd_root / "yawn"
    drowsy_sources.extend(collect_images_from_folder(yawdd_yawn_dir, "yawn"))

    # Opcional: añadir algunos ojos cerrados como drowsy
    eyes_closed_dir = eyes_root / "closed"
    drowsy_sources.extend(collect_images_from_folder(eyes_closed_dir, "closed_eyes"))

    random.shuffle(drowsy_sources)
    drowsy_sources = drowsy_sources[:max_per_class]

    # ----------------------
    # Clase ALERT
    # ----------------------
    alert_sources = []

    yawdd_no_yawn_dir = yawdd_root / "no_yawn"
    alert_sources.extend(collect_images_from_folder(yawdd_no_yawn_dir, "no_yawn"))

    # Añadir ojos abiertos como alert
    eyes_open_dir = eyes_root / "open"
    alert_sources.extend(collect_images_from_folder(eyes_open_dir, "open_eyes"))

    random.shuffle(alert_sources)
    alert_sources = alert_sources[:max_per_class]

    # ----------------------
    # Copiar a estructura final
    # ----------------------
    out_alert_dir = DROWSY_ROOT / split / "alert"
    out_drowsy_dir = DROWSY_ROOT / split / "drowsy"

    print(f"  Samples drowsy (fuentes): {len(drowsy_sources)}")
    print(f"  Samples alert  (fuentes): {len(alert_sources)}")

    # Copiar drowsy
    for src in tqdm(drowsy_sources, desc=f"Copiando {split}/drowsy"):
        out_path = out_drowsy_dir / src.name
        shutil.copy(str(src), str(out_path))

    # Copiar alert
    for src in tqdm(alert_sources, desc=f"Copiando {split}/alert"):
        out_path = out_alert_dir / src.name
        shutil.copy(str(src), str(out_path))

    print(f"  Total drowsy en {split}: {len(list(out_drowsy_dir.glob('*')))}")
    print(f"  Total alert  en {split}: {len(list(out_alert_dir.glob('*')))}")


def main():
    print("=== Preprocesando dataset DROWSY global ===")
    ensure_dirs()

    # Construir train y val
    build_drowsy_split("train")
    build_drowsy_split("val")

    print("\n✔ Dataset drowsy construido en data/drowsy")
    print("   Estructura esperada para train_drowsy.py:")
    print("   data/drowsy/train/alert, data/drowsy/train/drowsy")
    print("   data/drowsy/val/alert,   data/drowsy/val/drowsy")


if __name__ == "__main__":
    main()
