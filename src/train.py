import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from config import *


def build_dataloaders():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225],   # ImageNet std
    )

    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes: int):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # optionally unfreeze last residual block (stronger than training only fc)
    if UNFREEZE_LAYER4:
        for p in model.layer4.parameters():
            p.requires_grad = True

    # replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total else 0.0
    return total_loss, acc


def train_one_epoch(model, loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total if total else 0.0
    return total_loss, acc


def main():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()
    num_classes = len(train_dataset.classes)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Classes:", num_classes)
    print("Device:", DEVICE)
    print("UNFREEZE_LAYER4:", UNFREEZE_LAYER4)
    print("Training started...")

    model = build_model(num_classes)

    # params to train: fc + (optional) layer4
    params_to_train = list(model.fc.parameters())
    if UNFREEZE_LAYER4:
        params_to_train += list(model.layer4.parameters())

    optimizer = optim.Adam(
        params_to_train,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    log_path = Path("outputs") / "reports" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # init csv log
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(CHECKPOINT_DIR) / "best_resnet50_lfw.pth")

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")
        print(f"Best Val acc so far: {best_val_acc:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{best_val_acc:.6f}"])

    # save final too
    torch.save(model.state_dict(), Path(CHECKPOINT_DIR) / "final_resnet50_lfw.pth")
    print("Saved:")
    print(" - best:", Path(CHECKPOINT_DIR) / "best_resnet50_lfw.pth")
    print(" - final:", Path(CHECKPOINT_DIR) / "final_resnet50_lfw.pth")
    print("Log:", log_path)


if __name__ == "__main__":
    main()