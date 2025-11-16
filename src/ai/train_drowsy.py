import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.mobilenet_fatigue import build_model


# ================================
# CONFIG
# ================================
DATA_ROOT = "data/drowsy"
MODEL_OUTPUT = "models/drowsy_cnn.pt"

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

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_ds.classes  # ["alert", "drowsy"]

    print("Clases encontradas:", class_names)

    return train_loader, val_loader, class_names


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_drowsy] Entrenando en: {device}")

    train_loader, val_loader, classes = get_loaders()

    model = build_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

        train_acc = total_correct / total_samples
        train_loss = total_loss / total_samples

        # === VALIDACIÓN ===
        model.eval()
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += imgs.size(0)

        val_acc = val_correct / val_samples

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        # Guardar si es el mejor
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"✓ Nuevo mejor modelo guardado: {MODEL_OUTPUT}")

    print("Entrenamiento completado.")
    print(f"Mejor accuracy validación: {best_val_acc:.3f}")


if __name__ == "__main__":
    train()
