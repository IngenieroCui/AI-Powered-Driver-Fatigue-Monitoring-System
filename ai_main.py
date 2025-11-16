import cv2
import mediapipe as mp
import numpy as np
import time

from src.ai.infer_eye import EyeInfer
from src.ai.infer_yawn import YawnInfer
from src.ai.infer_drowsy import DrowsyInfer


# -------------------------------
# Inicializar inferencias
# -------------------------------
eye_infer = EyeInfer(model_path="models/eye_cnn.pt")
yawn_infer = YawnInfer(model_path="models/yawn_cnn.pt")
drowsy_infer = DrowsyInfer(model_path="models/drowsy_cnn.pt")

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


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


# -------------------------------
# Regiones (landmarks MediaPipe)
# -------------------------------
LEFT_EYE = [33, 160, 158, 153, 144, 145, 163]
RIGHT_EYE = [263, 387, 385, 380, 373, 374, 390]
MOUTH = [61, 291, 0, 17, 78, 308, 13, 14]
FACE = list(range(0, 468))  # cara completa


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

    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        eye_prob = None       # prob ojos cerrados
        yawn_prob = None      # prob bostezo
        drowsy_prob = None    # prob somnolencia global

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]

            # ===== Recortes =====
            left_eye_crop = crop_region(frame, lm, LEFT_EYE)
            right_eye_crop = crop_region(frame, lm, RIGHT_EYE)
            mouth_crop = crop_region(frame, lm, MOUTH)
            face_crop = crop_region(frame, lm, FACE, margin=20)

            # ===== IA: ojos =====
            if left_eye_crop is not None and right_eye_crop is not None:
                combined = np.hstack((left_eye_crop, right_eye_crop))
                prob_closed = eye_infer.predict(combined)  # ← devuelve float
                if prob_closed is not None:
                    eye_prob = float(prob_closed)

            # ===== IA: bostezo =====
            if mouth_crop is not None:
                prob_yawn, _ = yawn_infer.predict(mouth_crop)  # ← (prob, clase)
                if prob_yawn is not None:
                    yawn_prob = float(prob_yawn)

            # ===== IA: drowsy global =====
            if face_crop is not None:
                pred_d = drowsy_infer.predict(face_crop)  # {"alert": x, "drowsy": y}
                if pred_d is not None:
                    drowsy_prob = float(pred_d["drowsy"])

        # ===== Dibujar en pantalla =====
        y_offset = 30

        if eye_prob is not None:
            draw_bar(frame, 10, y_offset, eye_prob, (0, 0, 255), "Eyes Closed")
            y_offset += 40

        if yawn_prob is not None:
            draw_bar(frame, 10, y_offset, yawn_prob, (255, 0, 0), "Yawning")
            y_offset += 40

        if drowsy_prob is not None:
            draw_bar(frame, 10, y_offset, drowsy_prob, (0, 255, 255), "Drowsy")
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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
