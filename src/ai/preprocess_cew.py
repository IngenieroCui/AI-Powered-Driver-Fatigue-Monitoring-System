import os
import shutil
import random
from pathlib import Path
import cv2
import mediapipe as mp

# -------- SETTINGS -------- #
SOURCE_DIR = "cew/dataset_B_FacialImages_highResolution"
OUT_DIR = "data/cew_processed"
TRAIN_SPLIT = 0.8
DETECT_EYES = True   # Cambia a False si ya están bien recortadas


mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)


def detect_eye_region(image):
    """Recorta la región de los ojos usando MediaPipe FaceMesh."""
    h, w = image.shape[:2]
    results = mp_face.process(image)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # Landmarks aproximados para ojos
    left_eye_ids = [33, 133]
    right_eye_ids = [362, 263]

    xs = [lm[i].x for i in left_eye_ids + right_eye_ids]
    ys = [lm[i].y for i in left_eye_ids + right_eye_ids]

    x1, x2 = int(min(xs)*w), int(max(xs)*w)
    y1, y2 = int(min(ys)*h) - 10, int(max(ys)*h) + 10

    y1 = max(0, y1)
    y2 = min(h, y2)
    x1 = max(0, x1)
    x2 = min(w, x2)

    return image[y1:y2, x1:x2]


def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["open", "closed"]:
            Path(f"{OUT_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    images = list(Path(SOURCE_DIR).glob("*"))

    random.shuffle(images)
    split_point = int(len(images) * TRAIN_SPLIT)

    train_files = images[:split_point]
    val_files = images[split_point:]

    for dataset, files in [("train", train_files), ("val", val_files)]:
        for path in files:
            filename = path.name.lower()

            # Clasificación
            label = "closed" if "closed_eye" in filename else "open"

            img = cv2.imread(str(path))
            if img is None:
                continue

            if DETECT_EYES:
                crop = detect_eye_region(img)
                if crop is None:
                    continue
                img = crop

            save_path = f"{OUT_DIR}/{dataset}/{label}/{path.stem}.jpg"
            cv2.imwrite(save_path, img)

    print("\n✔ CEW procesado correctamente.")
    print(f"Imágenes en train/ y val/ guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
