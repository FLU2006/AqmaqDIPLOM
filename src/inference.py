import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image

from config import *


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print("File not found:", img_path)
        sys.exit(1)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # classes must match training order
    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(Path(CHECKPOINT_DIR) / "best_resnet50_lfw.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        topk = torch.topk(probs, k=min(5, num_classes))

    print("Top predictions:")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"{class_names[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()