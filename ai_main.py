import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv

from src.ai.infer_eye import EyeInfer
from src.ai.infer_yawn import YawnInfer
from src.ai.infer_drowsy import DrowsyInfer
from src.classic.config import LEFT_EYE as LEFT_EYE_LM, RIGHT_EYE as RIGHT_EYE_LM, MOUTH as MOUTH_LM  # asegurar mismos landmarks que en classic


# -------------------------------
# Inicializar inferencias
# -------------------------------
eye_infer = EyeInfer(model_path="src/models/eye_cnn.pt")
yawn_infer = YawnInfer(model_path="src/models/yawn_cnn.pt")
drowsy_infer = DrowsyInfer(model_path="src/models/drowsy_cnn.pt")

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


DATA_REALTIME_DIR = os.path.join("data", "realtime")


# -------------------------------
# Utilidades de recorte
# -------------------------------
def crop_region(frame, lm, points, margin=10):
    h, w, _ = frame.shape

    xs = [lm.landmark[p].x * w for p in points]
    ys = [lm.landmark[p].y * h for p in points]

    x1, x2 = int(min(xs)) - margin, int(max(xs)) + margin
    y1, y2 = int(min(ys)) - margin, int(max(ys)) + margin

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def crop_mouth_region(frame, lm, mouth_points, extra_scale=2.0):
        """Recorte robusto de la región de la boca.

        - Calcula el bounding box mínimo de los landmarks de la boca.
        - Lo expande en ancho/alto con un factor `extra_scale` para capturar
            mejor la boca incluso si la cabeza está algo girada.
        - Asegura una relación de aspecto más cuadrada para parecerse al
            tipo de recorte usado al entrenar (región de boca + algo de mejilla/mentón).
        """
        h, w, _ = frame.shape

        xs = [lm.landmark[p].x * w for p in mouth_points]
        ys = [lm.landmark[p].y * h for p in mouth_points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5

        bw = (x_max - x_min)
        bh = (y_max - y_min)

        # Hacer la caja algo más alta para incluir mentón y parte de nariz
        size = max(bw, bh) * extra_scale

        x1 = int(cx - size / 2)
        x2 = int(cx + size / 2)
        y1 = int(cy - size / 2)
        y2 = int(cy + size / 2)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
                return None
        return crop


# -------------------------------
# Regiones (landmarks MediaPipe)
# -------------------------------
# Usamos los mismos índices que en classic.config para mantener consistencia
LEFT_EYE = LEFT_EYE_LM
RIGHT_EYE = RIGHT_EYE_LM
MOUTH = MOUTH_LM
FACE = list(range(0, 468))  # cara completa

# Saltar inferencias en algunos frames para ahorrar CPU/GPU
SKIP_FRAMES_AI = 1  # 0 = inferir cada frame, 1 = uno sí/uno no, 2 = 1 de cada 3, etc.


# -------------------------------
# Dibujar barra de probabilidad
# -------------------------------
def draw_bar(img, x, y, prob, color, text):
    cv2.rectangle(img, (x, y), (x + 200, y + 25), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(prob * 200), y + 25), color, -1)
    cv2.putText(img, f"{text}: {prob:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# -------------------------------
# PROGRAMA PRINCIPAL IA
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    os.makedirs(DATA_REALTIME_DIR, exist_ok=True)

    session_id = time.strftime("session_%Y%m%d_%H%M%S")
    csv_path = os.path.join(DATA_REALTIME_DIR, f"{session_id}.csv")

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "frame_idx",
        "eye_prob_closed",
        "yawn_prob",
        "drowsy_prob"
    ])

    fps_time = time.time()
    frame_idx = 0

    # buffers para suavizar barras (ventana de últimos N valores)
    smooth_window = 5
    eye_hist = []
    yawn_hist = []
    drowsy_hist = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        eye_prob = None       # prob ojos cerrados
        yawn_prob = None      # prob bostezo
        drowsy_prob = None    # prob somnolencia global

    # Control de frecuencia de inferencia para ahorrar recursos
        do_infer = (SKIP_FRAMES_AI <= 0) or (frame_idx % (SKIP_FRAMES_AI + 1) == 0)
        frame_idx += 1

        if res.multi_face_landmarks and do_infer:
            lm = res.multi_face_landmarks[0]

            # ===== Recortes =====
            left_eye_crop = crop_region(frame, lm, LEFT_EYE)
            right_eye_crop = crop_region(frame, lm, RIGHT_EYE)
            # Recorte de boca con caja grande y casi cuadrada para mayor robustez
            mouth_crop = crop_mouth_region(frame, lm, MOUTH, extra_scale=2.0)
            face_crop = crop_region(frame, lm, FACE, margin=20)

            # ===== IA: ojos =====
            # Usamos solo un ojo (por ejemplo el izquierdo) para alinearlo con el entrenamiento,
            # donde cada imagen contiene una región de ojos y no una concatenación artificial.
            eye_crop = left_eye_crop if left_eye_crop is not None else right_eye_crop
            if eye_crop is not None and eye_crop.size > 0:
                prob_closed = eye_infer.predict(eye_crop)  # devuelve float
                if prob_closed is not None:
                    eye_prob = float(prob_closed)

            # ===== IA: bostezo =====
            if mouth_crop is not None and mouth_crop.size > 0:
                prob_yawn, _ = yawn_infer.predict(mouth_crop)  # ← (prob, clase)
                if prob_yawn is not None:
                    yawn_prob = float(prob_yawn)

            # ===== IA: drowsy global (solo cara) =====
            if face_crop is not None and face_crop.size > 0:
                pred_d = drowsy_infer.predict(face_crop)  # {"alert": x, "drowsy": y}
                if pred_d is not None:
                    drowsy_prob = float(pred_d["drowsy"])

        # ===== suavizado (media móvil simple) =====
        def update_hist(hist, value):
            if value is None:
                return hist
            hist.append(value)
            if len(hist) > smooth_window:
                hist.pop(0)
            return hist

        eye_hist = update_hist(eye_hist, eye_prob)
        yawn_hist = update_hist(yawn_hist, yawn_prob)
        drowsy_hist = update_hist(drowsy_hist, drowsy_prob)

        eye_smooth = np.mean(eye_hist) if eye_hist else None
        yawn_smooth = np.mean(yawn_hist) if yawn_hist else None
        drowsy_smooth = np.mean(drowsy_hist) if drowsy_hist else None

        # ===== recalibrar para visualización =====
        # ojo: en la práctica el modelo está dando 1 = abierto, 0 = cerrado
        # invertimos para que la barra muestre 1 = cerrado
        if eye_smooth is not None:
            eye_closed_vis = 1.0 - eye_smooth
        else:
            eye_closed_vis = None

        # bostezo: forzar a [0,1] y aplicar ganancia fuerte para romper la zona muerta
        def clamp01(v):
            return max(0.0, min(1.0, float(v)))

        if yawn_smooth is not None:
            # centrar alrededor de 0.2 y amplificar
            # si el modelo da valores bajos pero sube un poco al bostezar,
            # esta transformación vuelve visible ese cambio
            boosted = (yawn_smooth - 0.1) * 4.0
            yawn_vis = clamp01(boosted)
        else:
            yawn_vis = None

        # drowsy: damos más peso a ojos y bostezo, y menos al modelo
        # si bostezas o tienes los ojos cerrados, drowsy debe subir
        base_drowsy = drowsy_smooth if drowsy_smooth is not None else 0.5
        eye_component = eye_closed_vis if eye_closed_vis is not None else 0.0
        yawn_component = yawn_vis if yawn_vis is not None else 0.0

        # mezcla: 20% modelo, 40% ojos cerrados, 40% bostezo
        combined_drowsy = 0.2 * base_drowsy + 0.4 * eye_component + 0.4 * yawn_component

        # estirar un poco el contraste alrededor de 0.5
        combined_drowsy = (combined_drowsy - 0.5) * 1.5 + 0.5

        drowsy_vis = clamp01(combined_drowsy) if drowsy_hist else None

        # ===== logging CSV =====
        timestamp = time.time()
        csv_writer.writerow([
            f"{timestamp:.3f}",
            frame_idx,
            f"{eye_closed_vis:.4f}" if eye_closed_vis is not None else "",
            f"{yawn_vis:.4f}" if yawn_vis is not None else "",
            f"{drowsy_vis:.4f}" if drowsy_vis is not None else "",
        ])

        # ===== Dibujar en pantalla =====
        y_offset = 30

        if eye_closed_vis is not None:
            draw_bar(frame, 10, y_offset, eye_closed_vis, (0, 0, 255), "Eyes Closed")
            y_offset += 40

        if yawn_vis is not None:
            draw_bar(frame, 10, y_offset, yawn_vis, (255, 0, 0), "Yawning")
            y_offset += 40

        if drowsy_vis is not None:
            draw_bar(frame, 10, y_offset, drowsy_vis, (0, 255, 255), "Drowsy")
            y_offset += 40

        # ===== FPS =====
        now = time.time()
        fps = 1 / (now - fps_time)
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AI Fatigue Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
