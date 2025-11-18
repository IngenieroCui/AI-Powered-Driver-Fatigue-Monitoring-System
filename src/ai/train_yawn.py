import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Asegurar que la raíz del proyecto esté en sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === IMPORTAR TU MODELO ===
from src.models.mobilenet_fatigue import build_model


# ============================================================
#            DATASET PARA YAWN vs NO_YAWN
# ============================================================

class YawnDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.samples = []

        for label in ["yawn", "no_yawn"]:
            folder = self.root / label
            for img in folder.glob("*"):
                self.samples.append((str(img), 1 if label == "yawn" else 0))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# ============================================================
#                      TRAIN FUNCTION
# ============================================================

def train_yawn_model():

    TRAIN_DIR = "data/yawdd/train"
    VAL_DIR = "data/yawdd/val"
    SAVE_PATH = "src/models/yawn_cnn.pt"

    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("\n=== Cargando dataset YAWDD Yawn ===")

    train_ds = YawnDataset(TRAIN_DIR)
    val_ds = YawnDataset(VAL_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {device}")

    # Modelo
    model = build_model(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10

    # Configurar AMP sólo si hay CUDA y la API está disponible
    use_amp = (device == "cuda") and hasattr(torch.cuda, "amp")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # ------------- TRAIN -------------
        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_ds)
        train_loss = total_loss / len(train_loader)

        # ------------- VALIDATION -------------
        model.eval()
        val_correct = 0
        val_loss_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}/{EPOCHS}"):
                imgs = imgs.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_ds)
        val_loss = val_loss_total / len(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n=== Modelo guardado en {SAVE_PATH} ===")


# ============================================================
#                       MAIN
# ============================================================

if __name__ == "__main__":
    train_yawn_model()
