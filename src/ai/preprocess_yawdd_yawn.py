import os
import cv2
import random
import shutil
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import csv

# === CONFIG ===
YAWDD_ROOT = "data/yawdd"
OUTPUT_RAW = "data/yawdd_frames/raw"
OUTPUT_DATASET = "data/yawdd"
FRAME_EVERY = 5
TRAIN_SPLIT = 0.8
MAR_THRESHOLD = 0.65  # si dash video no tiene etiqueta explicita

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks for MAR
MOUTH_TOP = [13, 14]
MOUTH_BOTTOM = [17, 0]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

def compute_MAR(lm, w, h):
    top = np.array([lm.landmark[i].x * w, lm.landmark[i].y * h] for i in MOUTH_TOP)
    bottom = np.array([lm.landmark[i].x * w, lm.landmark[i].y * h] for i in MOUTH_BOTTOM)

    left = np.array([lm.landmark[MOUTH_LEFT].x * w, lm.landmark[MOUTH_LEFT].y * h])
    right = np.array([lm.landmark[MOUTH_RIGHT].x * w, lm.landmark[MOUTH_RIGHT].y * h])

    vertical = np.linalg.norm(top.mean(axis=0) - bottom.mean(axis=0))
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0
    return vertical / horizontal


def guess_label(video_name, mar):
    name = video_name.lower()

    if "yawning" in name:
        return "yawn"
    if "normal" in name or "talking" in name:
        return "no_yawn"

    # DASH VIDEOS → Heurística con MAR
    return "yawn" if mar > MAR_THRESHOLD else "no_yawn"


def extract_mouth_region_and_mar(frame_bgr):
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    if not res.multi_face_landmarks:
        return None, None
    
    lm = res.multi_face_landmarks[0]

    # MAR CALCULATION
    mar = compute_MAR(lm, w, h)

    # Crop mouth region
    mouth_points = [61, 291, 0, 17, 78, 308, 13, 14]
    xs = [lm.landmark[i].x * w for i in mouth_points]
    ys = [lm.landmark[i].y * h for i in mouth_points]

    x1, x2 = int(min(xs))-10, int(max(xs))+10
    y1, y2 = int(min(ys))-10, int(max(ys))+10
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, mar


def extract_frames_from_video(video_path, out_list, csv_writer):
    fname = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_EVERY != 0:
            continue

        crop, mar = extract_mouth_region_and_mar(frame)
        if crop is None:
            continue

        label = guess_label(fname, mar)

        out_name = f"{fname}_f{i}_{label}.jpg"
        out_path = os.path.join(OUTPUT_RAW, label, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, crop)

        out_list.append((out_path, label))
        csv_writer.writerow([out_path, label, round(mar, 4)])

    cap.release()


def gather_all():
    all_items = []

    folders = [
        os.path.join(YAWDD_ROOT, "Dash", "Dash", "Female"),
        os.path.join(YAWDD_ROOT, "Dash", "Dash", "Male"),
        os.path.join(YAWDD_ROOT, "Mirror", "Mirror", "Female_mirror"),
        os.path.join(YAWDD_ROOT, "Mirror", "Mirror", "Male_mirror")
    ]

    os.makedirs(OUTPUT_RAW, exist_ok=True)

    with open("data/yawdd_debug.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "mar"])

        for folder in folders:
            if not os.path.isdir(folder):
                continue
            vids = [os.path.join(folder, v) for v in os.listdir(folder)
                    if v.lower().endswith((".avi", ".mp4", ".mov"))]

            for v in tqdm(vids, desc=f"Procesando {folder}"):
                extract_frames_from_video(v, all_items, writer)

    return all_items


def split_dataset(all_items):
    random.shuffle(all_items)

    n_train = int(len(all_items) * TRAIN_SPLIT)
    train, val = all_items[:n_train], all_items[n_train:]

    for split_name, subset in [("train", train), ("val", val)]:
        for src, label in subset:
            dst_dir = os.path.join(OUTPUT_DATASET, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))

    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")


def main():
    print("=== Extrayendo frames YAWDD (IA v2) ===")
    items = gather_all()

    print("=== Dividiendo train/val ===")
    split_dataset(items)

    print("=== COMPLETADO ===")
    print("Dataset listo para train_yawn.py")
    print("Mira el debug en data/yawdd_debug.csv")

if __name__ == "__main__":
    main()
