import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import *


def main():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    num_classes = len(test_dataset.classes)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR) / "best_resnet50_lfw.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    acc = accuracy_score(y_true, y_pred)
    print("Test accuracy:", acc)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    out_dir = Path("outputs") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "test_metrics.json").write_text(json.dumps({
        "test_accuracy": acc,
        "classification_report": report
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # confusion matrix (сохраним как .npy, а картинку сделаем позже при необходимости)
    cm = confusion_matrix(y_true, y_pred)
    import numpy as np
    np.save(out_dir / "confusion_matrix.npy", cm)

    print("Saved reports to:", out_dir)


if __name__ == "__main__":
    main()