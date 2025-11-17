import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path

from src.models.mobilenet_fatigue import build_model


class YawnInfer:
    def __init__(self, model_path="models/yawn_cnn.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[YawnInfer] No se encontró el modelo '{model_path}'. "
                "Entrena el modelo ejecutando src/ai/train_yawn.py o coloca el .pt en la carpeta models."
            )

        # === modelo ===
        self.model = build_model(num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # === transforms igual que en train_yawn.py ===
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

        print(f"[YawnInfer] Modelo cargado en {self.device}")

    def predict(self, frame_bgr):
        """
        frame_bgr: recorte de boca/cara en formato BGR (OpenCV)
        return: prob_yawn, predicted_class
        """

        # Si el recorte está vacío → no procesar
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None

        # Convertir BGR → RGB → PIL
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Preprocesar
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        prob_yawn = float(probs[0][1])     # clase 1 = yawn
        pred_class = int(probs.argmax(1))  # 0 no_yawn, 1 yawn

        return prob_yawn, pred_class
