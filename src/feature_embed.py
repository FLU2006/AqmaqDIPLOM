from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

from config import *


def build_feature_extractor(num_classes: int):
    # Загружаем архитектуру так же, как при обучении (чтобы state_dict совпал)
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    ckpt = Path(CHECKPOINT_DIR) / "best_resnet50_lfw.pth"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Убираем финальный классификатор, оставляем backbone до avgpool
    backbone = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
    return backbone


def get_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


@torch.no_grad()
def image_to_embedding(backbone, pil_img) -> np.ndarray:
    x = get_transform()(pil_img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    feat = backbone(x)  # (1,2048,1,1)
    feat = feat.view(feat.size(0), -1)  # (1,2048)
    emb = feat[0].detach().cpu().numpy().astype(np.float32)

    # L2-normalize для cosine similarity
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


def get_class_names() -> list[str]:
    # порядок классов должен совпадать с тренировкой
    ds = datasets.ImageFolder(TRAIN_DIR)
    return ds.classes


def get_num_classes() -> int:
    return len(get_class_names())