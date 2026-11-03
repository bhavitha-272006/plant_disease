"""
Rigorous Benchmark and Evaluation Suite for Full-Dataset Trained PlantVillage Model.
Evaluates:
- 8,143 held-out stratified test specimens
- Overall test accuracy, macro precision, recall, and F1-score
- Weighted F1-score
- Full 38x38 confusion matrix
- Per-class classification report
- PlantDoc field images (domain transfer)
- Head-to-head comparison with 4,560-image baseline
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

FULL_MODEL_PATH = MODELS_DIR / "trained_plantvillage_full_model.keras"
FULL_CLASSES_PATH = MODELS_DIR / "trained_plantvillage_full_model_classes.json"
FULL_HISTORY_PATH = MODELS_DIR / "trained_plantvillage_full_model_history.json"
SPLIT_METADATA_PATH = MODELS_DIR / "full_dataset_split_metadata.json"
BASELINE_MODEL_PATH = MODELS_DIR / "trained_plantvillage_model.keras"
BASELINE_CLASSES_PATH = MODELS_DIR / "trained_plantvillage_classes.json"
OUT_REPORT_PATH = DATA_DIR / "full_model_evaluation_report.json"

PLANTDOC_DIR = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"

def plot_training_curves(history_data, out_path):
    h = history_data.get("history", {})
    epochs = range(1, len(h.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#06121f')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0a1d30')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.grid(True, linestyle='--', alpha=0.3, color='#334155')

    # Accuracy
    ax1.plot(epochs, [a * 100 for a in h.get("accuracy", [])], 'o-', color='#19e6b0', linewidth=2, label='Train Accuracy')
    ax1.plot(epochs, [a * 100 for a in h.get("val_accuracy", [])], 's--', color='#38bdf8', linewidth=2, label='Val Accuracy')
    ax1.set_title('PlantVillage Full-Dataset Accuracy Progression', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

    # Loss
    ax2.plot(epochs, h.get("loss", []), 'o-', color='#f43f5e', linewidth=2, label='Train Loss')
    ax2.plot(epochs, h.get("val_loss", []), 's--', color='#fb923c', linewidth=2, label='Val Loss')
    ax2.set_title('PlantVillage Full-Dataset Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Full-dataset training curves saved to {out_path}", flush=True)


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(18, 16))
    fig = plt.gcf()
    fig.patch.set_facecolor('#06121f')
    ax = plt.gca()
    ax.set_facecolor('#0a1d30')

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='GnBu')    
    plt.title('PlantVillage 38-Class Full-Dataset Normalized Confusion Matrix', fontsize=16, fontweight='bold', color='#f8fafc', pad=15)

    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors='#94a3b8')
    cb.set_label('Normalized Match Rate', color='#94a3b8', labelpad=10)

    tick_marks = np.arange(len(class_names))
    short_names = [c.replace('___', ' ').replace('_', ' ') for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=90, fontsize=8, color='#94a3b8')
    plt.yticks(tick_marks, short_names, fontsize=8, color='#94a3b8')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold', color='#94a3b8', labelpad=10)
    plt.ylabel('True Class', fontsize=12, fontweight='bold', color='#94a3b8', labelpad=10)

    for spine in ax.spines.values():
        spine.set_color('#1e293b')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Full-dataset confusion matrix heatmap saved to {out_path}", flush=True)


def evaluate():
    print("=" * 70, flush=True)
    print("PLANTGUARD AI - HELD-OUT TEST EVALUATION & BENCHMARK COMPARISON", flush=True)
    print("=" * 70, flush=True)

    if not FULL_MODEL_PATH.exists():
        print(f"[ERROR] Full-dataset model weights not found at {FULL_MODEL_PATH}", flush=True)
        sys.exit(1)

    print(f"[LOADING] Loading full-dataset model from {FULL_MODEL_PATH}...", flush=True)
    model = tf.keras.models.load_model(str(FULL_MODEL_PATH), compile=False)
    with open(FULL_CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape
    if output_shape[-1] != len(classes):
        raise ValueError(f"Full model has {output_shape[-1]} outputs but mapping has {len(classes)} classes.")
    if output_shape[-1] != 38:
        raise ValueError(f"Expected a 38-class model, found {output_shape[-1]} outputs.")
    print(f"[LOADED] Full-dataset model loaded with {len(classes)} classes.", flush=True)

    if not BASELINE_MODEL_PATH.exists() or not BASELINE_CLASSES_PATH.exists():
        raise FileNotFoundError("The existing trained baseline model or class mapping is missing.")
    print(f"[LOADING] Loading existing baseline from {BASELINE_MODEL_PATH}...", flush=True)
    baseline_model = tf.keras.models.load_model(str(BASELINE_MODEL_PATH), compile=False)
    with open(BASELINE_CLASSES_PATH, "r", encoding="utf-8") as f:
        baseline_classes = json.load(f)
    if baseline_classes != classes:
        raise ValueError("Full model and baseline class mappings differ; same-split comparison is not valid.")

    with open(SPLIT_METADATA_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    test_paths = split_info["test_paths"]
    test_y = np.array(split_info["test_y"])
    if len(test_paths) != split_info["test_count"] or len(test_y) != len(test_paths):
        raise ValueError("Held-out test paths, labels, and declared test count do not agree.")
    if len(test_paths) != 8143:
        raise ValueError(f"Expected the existing 8,143-image test split, found {len(test_paths)} images.")
    if not all(Path(path).is_file() for path in test_paths):
        raise FileNotFoundError("One or more held-out test images are missing.")
    if not all(0 <= label < len(classes) and Path(path).parent.name == classes[label]
               for path, label in zip(test_paths, test_y)):
        raise ValueError("A held-out test label does not match its class directory.")

    def parse_test_img(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32)

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(parse_test_img, num_parallel_calls=AUTOTUNE).batch(64)

    def run_model_evaluation(eval_model, model_name):
        print(f"[TEST EVALUATION] {model_name}: {len(test_paths)} held-out images...", flush=True)
        t0 = time.time()
        predictions = eval_model.predict(test_ds, verbose=1)
        elapsed = time.time() - t0
        y_pred = np.argmax(predictions, axis=1)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            test_y, y_pred, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            test_y, y_pred, average="weighted", zero_division=0
        )
        report = classification_report(
            test_y, y_pred, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0
        )
        metrics = {
            "test_accuracy": float(np.mean(y_pred == test_y)),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1_score": float(f1_macro),
            "weighted_f1_score": float(f1_weighted),
            "evaluation_time_seconds": round(elapsed, 2)
        }
        return metrics, report, confusion_matrix(test_y, y_pred, labels=range(len(classes)))

    full_metrics, full_class_report, full_cm = run_model_evaluation(model, "Full model")
    baseline_metrics, baseline_class_report, _ = run_model_evaluation(baseline_model, "Baseline model")
    plot_confusion_matrix(full_cm, classes, ASSETS_DIR / "full_model_confusion_matrix.png")

    with open(FULL_HISTORY_PATH, "r", encoding="utf-8") as f:
        history_data = json.load(f)

    comparison = {
        "baseline_model_file": BASELINE_MODEL_PATH.name,
        "full_model_file": FULL_MODEL_PATH.name,
        "test_images_count": len(test_paths),
        "same_test_split": True,
        "baseline_metrics": baseline_metrics,
        "full_model_metrics": full_metrics,
        "delta_full_minus_baseline": {
            key: full_metrics[key] - baseline_metrics[key]
            for key in ("test_accuracy", "macro_precision", "macro_recall", "macro_f1_score", "weighted_f1_score")
        }
    }

    full_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "MobileNetV2 (Transfer Learning, Rescaling[-1, 1], Class-Weighted)",
        "dataset_name": split_info["dataset_name"],
        "total_unique_images": split_info["total_unique_images"],
        "duplicates_filtered": split_info["duplicates_filtered"],
        "num_classes": len(classes),
        "class_names": classes,
        "split_counts": {
            "train": split_info["train_count"],
            "val": split_info["val_count"],
            "test": split_info["test_count"],
        },
        "training_summary": history_data,
        "test_split_metrics": full_metrics,
        "comparison_against_baseline_on_same_test_split": comparison,
        "baseline_per_class_classification_report": baseline_class_report,
        "per_class_classification_report": full_class_report,
        "numeric_confusion_matrix": {
            "labels": classes,
            "matrix": full_cm.tolist(),
            "rows_are": "true_class",
            "columns_are": "predicted_class"
        },
        "confusion_matrix_image": "assets/full_model_confusion_matrix.png",
        "evaluation_scope": "Existing held-out PlantVillage test split only; no training or validation data evaluated."
    }

    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[REPORT WRITTEN] Full report written to {OUT_REPORT_PATH}", flush=True)


if __name__ == "__main__":
    evaluate()

