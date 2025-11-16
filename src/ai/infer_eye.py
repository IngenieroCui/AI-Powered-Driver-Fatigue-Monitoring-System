import torch
from torchvision import transforms
from PIL import Image
import cv2
from src.models.mobilenet_fatigue import build_model

class EyeInfer:
    def __init__(self, model_path="models/eye_cnn.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def predict(self, frame_bgr):
        img = Image.fromarray(frame_bgr[:, :, ::-1])
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # probs[1] = ojo CERRADO
        return float(probs[1])
