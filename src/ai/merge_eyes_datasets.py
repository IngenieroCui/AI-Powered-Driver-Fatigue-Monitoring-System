import os
import shutil
from pathlib import Path

CEW_DIR = "data/cew_processed"
FYP_DIR = "data/fyp_processed"
OUT_DIR = "data/eyes_combined"

def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["open", "closed"]:
            Path(f"{OUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

def merge_folder(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        print(f"⚠️ WARN: No existe {src_dir}, saltando...")
        return

    for img_path in src_dir.glob("*"):
        if img_path.is_file() and img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            dst_path = dst_dir / img_path.name
            shutil.copy(img_path, dst_path)

def main():
    ensure_dirs()

    print("📌 Unificando CEW + FYP en eyes_combined...\n")

    datasets = [
        ("CEW", CEW_DIR),
        ("FYP", FYP_DIR)
    ]

    for name, src_root in datasets:
        print(f"➡️ Procesando {name}...")

        for split in ["train", "val"]:
            for cls in ["open", "closed"]:
                source = f"{src_root}/{split}/{cls}"
                dest = f"{OUT_DIR}/{split}/{cls}"

                merge_folder(source, dest)

        print(f"✔️ {name} añadido correctamente.\n")

    total_train_open = len(list(Path(f"{OUT_DIR}/train/open").glob("*")))
    total_train_closed = len(list(Path(f"{OUT_DIR}/train/closed").glob("*")))
    total_val_open = len(list(Path(f"{OUT_DIR}/val/open").glob("*")))
    total_val_closed = len(list(Path(f"{OUT_DIR}/val/closed").glob("*")))

    print("🎉 MERGE COMPLETO!")
    print("=== Estadísticas Finales ===")
    print(f"Train/Open:   {total_train_open}")
    print(f"Train/Closed: {total_train_closed}")
    print(f"Val/Open:     {total_val_open}")
    print(f"Val/Closed:   {total_val_closed}")
    print("\nDataset final en:", OUT_DIR)

if __name__ == "__main__":
    main()
