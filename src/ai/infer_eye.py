import os
import torch
from torchvision import transforms
from PIL import Image
import cv2
from src.models.mobilenet_fatigue import build_model


class EyeInfer:
    def __init__(self, model_path="models/eye_cnn.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[EyeInfer] No se encontró el modelo '{model_path}'. "
                "Entrena el modelo ejecutando src/ai/train_eye.py o coloca el .pt en la carpeta models."
            )

        self.model = build_model(num_classes=2).to(self.device)
        # Cargar sólo pesos (state_dict) de forma segura
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def predict(self, frame_bgr):
        """Devuelve probabilidad de OJO CERRADO (float 0-1)."""

        # Validar entrada
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        # Asegurar 3 canales: si viene en gris, replicar
        if len(frame_bgr.shape) == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

        # BGR (OpenCV) -> RGB (PIL), igual que en train_eye
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # probs[1] = ojo CERRADO
        return float(probs[1])
