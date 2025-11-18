import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Asegurar que la raíz del proyecto esté en sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.mobilenet_fatigue import build_model

# ==========================
# CONFIG
# ==========================
DATA_ROOT = "data/eyes_combined"  # <--- ANTES: data/cew
MODEL_OUTPUT = "src/models/eye_cnn.pt"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 12
IMAGE_SIZE = 224


def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("Clases:", train_ds.classes)  # debería ser ["open", "closed"]

    return train_loader, val_loader


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_eye] Entrenando en: {device}")

    # Crear carpeta de modelos si no existe
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)

    train_loader, val_loader = get_loaders()

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    # Configurar AMP sólo si hay CUDA y la API está disponible
    use_amp = (device == "cuda") and hasattr(torch.cuda, "amp")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for imgs, labels in tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Validación
        model.eval()
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[Val]   Epoch {epoch}/{EPOCHS}"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += imgs.size(0)

        val_acc = val_correct / val_samples if val_samples > 0 else 0.0

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"✓ Nuevo mejor modelo guardado en {MODEL_OUTPUT}")

    print("[train_eye] Entrenamiento completado. Mejor Val Acc:", best_val_acc)


if __name__ == "__main__":
    train()
