import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

from src.models.mobilenet_fatigue import build_model


class DrowsyInfer:
    def __init__(self, model_path="models/drowsy_cnn.pt", image_size=224):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Construir modelo
        self.model = build_model(num_classes=2).to(self.device)

        # Cargar pesos
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        # Clases
        self.classes = ["alert", "drowsy"]  # index 0, 1


    def predict(self, frame_bgr):
        """
        frame_bgr: recorte BGR (OpenCV) de la CARA COMPLETA
        return: prob_drowsy (float 0-1) o None si hay error
        """
        try:
            # Convertir a PIL
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Preprocesar
            img = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Inferencia
            with torch.no_grad():
                logits = self.model(img)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            prob_alert = float(probs[0])
            prob_drowsy = float(probs[1])

            return {
                "alert": prob_alert,
                "drowsy": prob_drowsy
            }

        except Exception as e:
            print("[infer_drowsy] Error:", e)
            return None
