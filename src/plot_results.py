from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def main():
    log_path = Path("outputs") / "reports" / "train_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss by epoch")
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy by epoch")
    plt.savefig(PLOTS_DIR / "accuracy_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()