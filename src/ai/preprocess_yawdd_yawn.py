import os
import cv2
import random
import shutil
import mediapipe as mp
from tqdm import tqdm
import numpy as np
import csv
from multiprocessing import Pool, cpu_count

# === CONFIG ===
YAWDD_ROOT = "data/yawdd"
OUTPUT_RAW = "data/yawdd_frames/raw"
OUTPUT_DATASET = "data/yawdd"
FRAME_EVERY = 10  # antes 5; usamos menos frames para acelerar el preprocesado
TRAIN_SPLIT = 0.8
MAR_THRESHOLD = 0.65  # si dash video no tiene etiqueta explicita

# Para evitar problemas con multiprocessing y estados globales,
# inicializaremos FaceMesh dentro de cada proceso trabajador.
mp_face = None


def get_face_mesh():
    """Inicializa (una vez por proceso) la instancia de FaceMesh.

    Esto permite usar multiproceso sin compartir estados internos de MediaPipe
    entre procesos, lo que podra causar errores o cuellos de botella.
    """
    global mp_face
    if mp_face is None:
        mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return mp_face

# Landmarks for MAR
MOUTH_TOP = [13, 14]
MOUTH_BOTTOM = [17, 0]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

def compute_MAR(lm, w, h):
    """Calcular Mouth Aspect Ratio (MAR) a partir de landmarks MediaPipe.

    Se toma el promedio de los puntos superiores e inferiores de la boca y
    se divide la distancia vertical entre ellos por la distancia horizontal
    entre las comisuras izquierda y derecha.
    """

    # Puntos superiores e inferiores de la boca
    top_pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in MOUTH_TOP])
    bottom_pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in MOUTH_BOTTOM])

    top = top_pts.mean(axis=0)
    bottom = bottom_pts.mean(axis=0)

    # Comisuras izquierda y derecha
    left = np.array([lm.landmark[MOUTH_LEFT].x * w, lm.landmark[MOUTH_LEFT].y * h])
    right = np.array([lm.landmark[MOUTH_RIGHT].x * w, lm.landmark[MOUTH_RIGHT].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def guess_label(video_name, mar):
    """Asignar etiqueta yawn/no_yawn usando nombre del video y MAR.

    Reglas clave:
    - Cualquier video con "talking" en el nombre SIEMPRE es no_yawn, aunque abra la boca.
    - Videos con "normal" tambi
n se consideran no_yawn.
    - Videos con "yawning" se marcan como yawn independientemente del MAR.
    - Solo los videos sin etiqueta explcita (p.ej. DASH) usan la heurstica de MAR.
    """

    name = video_name.lower()

    # Casos con etiqueta explcita en el nombre del archivo
    if "yawning" in name:
        return "yawn"

    # IMPORTANTE: talking SIEMPRE es no_yawn (aunque MAR sea alto)
    if "talking" in name or "normal" in name:
        return "no_yawn"

    # DASH VIDEOS  Heurstica con MAR solo cuando no hay etiqueta en el nombre
    return "yawn" if mar > MAR_THRESHOLD else "no_yawn"


def extract_mouth_region_and_mar(frame_bgr):
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    face_mesh = get_face_mesh()
    res = face_mesh.process(rgb)

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


def process_single_video(video_path):
    """Procesa un único video y devuelve la lista de (path, label, mar).

    Pensado para ser usado con multiprocessing.Pool para aprovechar todos
    los núcleos disponibles y acelerar el preprocesado.
    """
    items = []

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

        items.append((out_path, label, round(mar, 4)))

    cap.release()
    return items


def gather_all(num_workers=None):
    all_items = []

    folders = [
        os.path.join(YAWDD_ROOT, "Dash", "Dash", "Female"),
        os.path.join(YAWDD_ROOT, "Dash", "Dash", "Male"),
        os.path.join(YAWDD_ROOT, "Mirror", "Mirror", "Female_mirror"),
        os.path.join(YAWDD_ROOT, "Mirror", "Mirror", "Male_mirror")
    ]

    os.makedirs(OUTPUT_RAW, exist_ok=True)

    # Construir lista de vídeos
    all_videos = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        vids = [os.path.join(folder, v) for v in os.listdir(folder)
                if v.lower().endswith((".avi", ".mp4", ".mov"))]
        all_videos.extend(vids)

    if not all_videos:
        print("No se encontraron videos YAWDD en las rutas configuradas.")
        return []

    # Elegir número de procesos
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Usando {num_workers} procesos para preprocesar YAWDD...")

    # Procesamiento en paralelo
    with Pool(processes=num_workers) as pool:
        for video_items in tqdm(pool.imap_unordered(process_single_video, all_videos),
                                total=len(all_videos),
                                desc="Procesando videos YAWDD"):
            all_items.extend(video_items)

    # Guardar CSV de debug
    with open("data/yawdd_debug.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "mar"])
        for path, label, mar in all_items:
            writer.writerow([path, label, mar])

    # Devolver sólo (path, label) para el split
    return [(path, label) for path, label, _ in all_items]


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


def main(num_workers=None):
    print("=== Extrayendo frames YAWDD (IA v2) ===")
    items = gather_all(num_workers=num_workers)

    print("=== Dividiendo train/val ===")
    split_dataset(items)

    print("=== COMPLETADO ===")
    print("Dataset listo para train_yawn.py")
    print("Mira el debug en data/yawdd_debug.csv")

if __name__ == "__main__":
    # Llamada por defecto usando todos los cores menos uno para no saturar la máquina
    main()
