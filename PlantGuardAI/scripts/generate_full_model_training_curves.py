"""
Generate Training Curves Plot for the Full-Dataset MobileNetV2 Model.
Reads genuine training history from models/trained_plantvillage_full_model_history.json.
Saves plot to assets/full_model_training_curves.png.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "models" / "trained_plantvillage_full_model_history.json"
OUTPUT_PATH = BASE_DIR / "assets" / "full_model_training_curves.png"

with open(HISTORY_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data["history"]
epochs = list(range(1, len(history["loss"]) + 1))

plt.figure(figsize=(12, 5), dpi=150)

# 1. Loss Subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, history["loss"], "o-", color="#38bdf8", label="Training Loss", linewidth=2.2, markersize=6)
plt.plot(epochs, history["val_loss"], "s-", color="#10b981", label="Validation Loss", linewidth=2.2, markersize=6)
plt.title("Full-Dataset MobileNetV2: Cross-Entropy Loss", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="#082136", edgecolor="#0f495e", labelcolor="#ffffff")
plt.gca().set_facecolor("#082136")

# 2. Accuracy Subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, [a * 100 for a in history["accuracy"]], "o-", color="#38bdf8", label="Training Accuracy", linewidth=2.2, markersize=6)
plt.plot(epochs, [a * 100 for a in history["val_accuracy"]], "s-", color="#10b981", label="Validation Accuracy", linewidth=2.2, markersize=6)
plt.title("Full-Dataset MobileNetV2: Categorical Accuracy", fontsize=12, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True, facecolor="#082136", edgecolor="#0f495e", labelcolor="#ffffff")
plt.gca().set_facecolor("#082136")

# Style figure background
plt.gcf().patch.set_facecolor("#041320")
for ax in plt.gcf().axes:
    ax.tick_params(colors="#8daec0")
    ax.xaxis.label.set_color("#ffffff")
    ax.yaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#0f495e")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, facecolor=plt.gcf().get_facecolor(), edgecolor="none", bbox_inches="tight")
plt.close()

print(f"Generated full model training curves plot at: {OUTPUT_PATH}")

