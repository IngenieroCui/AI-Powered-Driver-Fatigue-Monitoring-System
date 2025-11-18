import os
import shutil
from pathlib import Path
import random

"""Preprocesado del dataset FYP.

Se asume que se ejecuta **desde la raíz del proyecto**, tal y como indica
`ejecutar.md`, con la siguiente estructura de carpetas:

data/
├── fyp/
│   └── train/
│       ├── Open_Eyes/
│       └── Closed_Eyes/

Por eso, las rutas se definen relativas a la raíz del proyecto (`data/...`).
"""

SOURCE_DIR = "data/fyp/train"  # Antes era "fyp/train", lo que fallaba al ejecutar desde la raíz
OUT_DIR = "data/fyp_processed"
TRAIN_SPLIT = 0.8

def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["open", "closed"]:
            Path(f"{OUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def collect_images():
    images = []

    for cls_original, cls_new in [
        ("Open_Eyes", "open"),
        ("Closed_Eyes", "closed")
    ]:
        folder = Path(SOURCE_DIR) / cls_original
        for img_path in folder.glob("*"):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                images.append((img_path, cls_new))

    return images

def main():
    ensure_dirs()
    images = collect_images()

    if len(images) == 0:
        print("❌ No se encontraron imágenes en FYP. Revisa la ruta.")
        return

    random.shuffle(images)
    split = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split]
    val_imgs = images[split:]

    # Copiado
    for dataset, items in [("train", train_imgs), ("val", val_imgs)]:
        for src, label in items:
            dst = f"{OUT_DIR}/{dataset}/{label}/{src.stem}.png"
            shutil.copy(src, dst)

    print("✔ FYP procesado correctamente.")
    print(f"Dataset final en: {OUT_DIR}")
    print(f"Total imágenes: {len(images)}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")

if __name__ == "__main__":
    main()
