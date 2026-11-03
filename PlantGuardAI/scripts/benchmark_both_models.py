import os
import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ['KERAS_HOME'] = str(BASE_DIR / '.keras')

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

BASELINE_MODEL_PATH = MODELS_DIR / "trained_plantvillage_model.keras"
FULL_MODEL_PATH = MODELS_DIR / "trained_plantvillage_full_model.keras"
CLASSES_PATH = MODELS_DIR / "trained_plantvillage_classes.json"
SPLIT_META_PATH = MODELS_DIR / "full_dataset_split_metadata.json"

def plot_confusion_matrix_clean(cm, class_names, out_path, title):
    plt.figure(figsize=(18, 16))
    fig = plt.gcf()
    fig.patch.set_facecolor('#06121f')
    ax = plt.gca()
    ax.set_facecolor('#0a1d30')

    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16, fontweight='bold', color='#f8fafc', pad=15)
    
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
    print(f"[PLOT] Confusion matrix saved to {out_path}", flush=True)

def main():
    print("=" * 70, flush=True)
    print("PLANTGUARD AI - COMPARATIVE BENCHMARK ON 8,143 HELD-OUT TEST IMAGES", flush=True)
    print("=" * 70, flush=True)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    with open(SPLIT_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_paths = meta["test_paths"]
    test_y = np.array(meta["test_y"])
    print(f"[DATASET] Evaluating {len(test_paths)} test specimens across {len(classes)} classes.", flush=True)

    def parse_test_img(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32)

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(parse_test_img, num_parallel_calls=AUTOTUNE).batch(64)

    # 1. EVALUATE BASELINE MODEL
    print(f"\n--- 1. Evaluating Baseline Model ({BASELINE_MODEL_PATH.name}) ---", flush=True)
    model_baseline = tf.keras.models.load_model(str(BASELINE_MODEL_PATH), compile=False)
    t0 = time.time()
    preds_baseline_raw = model_baseline.predict(test_ds, verbose=1)
    t_base = time.time() - t0
    y_pred_baseline = np.argmax(preds_baseline_raw, axis=1)

    acc_base = accuracy_score(test_y, y_pred_baseline)
    p_base_macro, r_base_macro, f1_base_macro, _ = precision_recall_fscore_support(test_y, y_pred_baseline, average='macro', zero_division=0)
    p_base_w, r_base_w, f1_base_w, _ = precision_recall_fscore_support(test_y, y_pred_baseline, average='weighted', zero_division=0)
    cm_base = confusion_matrix(test_y, y_pred_baseline, labels=range(len(classes)))

    print(f"Baseline Results on 8,143 Test Images:")
    print(f"  Accuracy:     {acc_base*100:.2f}%")
    print(f"  Macro Prec:   {p_base_macro*100:.2f}% | Recall: {r_base_macro*100:.2f}% | Macro F1: {f1_base_macro*100:.2f}%")
    print(f"  Weighted F1:  {f1_base_w*100:.2f}%")
    print(f"  Speed:        {len(test_paths)/t_base:.1f} img/s")

    # 2. EVALUATE FULL DATASET MODEL
    print(f"\n--- 2. Evaluating Full-Dataset Model ({FULL_MODEL_PATH.name}) ---", flush=True)
    model_full = tf.keras.models.load_model(str(FULL_MODEL_PATH), compile=False)
    t0 = time.time()
    preds_full_raw = model_full.predict(test_ds, verbose=1)
    t_full = time.time() - t0
    y_pred_full = np.argmax(preds_full_raw, axis=1)

    acc_full = accuracy_score(test_y, y_pred_full)
    p_full_macro, r_full_macro, f1_full_macro, _ = precision_recall_fscore_support(test_y, y_pred_full, average='macro', zero_division=0)
    p_full_w, r_full_w, f1_full_w, _ = precision_recall_fscore_support(test_y, y_pred_full, average='weighted', zero_division=0)
    cm_full = confusion_matrix(test_y, y_pred_full, labels=range(len(classes)))

    print(f"\nFull-Dataset Model Results on 8,143 Test Images:")
    print(f"  Accuracy:     {acc_full*100:.2f}%")
    print(f"  Macro Prec:   {p_full_macro*100:.2f}% | Recall: {r_full_macro*100:.2f}% | Macro F1: {f1_full_macro*100:.2f}%")
    print(f"  Weighted F1:  {f1_full_w*100:.2f}%")
    print(f"  Speed:        {len(test_paths)/t_full:.1f} img/s")

    # Save Plots
    plot_confusion_matrix_clean(cm_base, classes, ASSETS_DIR / "confusion_matrix_baseline.png", "Baseline Model (4.5k) Confusion Matrix on 8,143 Test Images")
    plot_confusion_matrix_clean(cm_full, classes, ASSETS_DIR / "confusion_matrix_full.png", "Full-Dataset Model (54k) Confusion Matrix on 8,143 Test Images")

    # Save Full Comparative JSON Report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_images_count": len(test_paths),
        "num_classes": len(classes),
        "baseline_model": {
            "model_file": str(BASELINE_MODEL_PATH.name),
            "training_images": 4560,
            "test_accuracy": float(acc_base),
            "macro_precision": float(p_base_macro),
            "macro_recall": float(r_base_macro),
            "macro_f1_score": float(f1_base_macro),
            "weighted_f1_score": float(f1_base_w),
            "evaluation_time_seconds": round(t_base, 2)
        },
        "full_dataset_model": {
            "model_file": str(FULL_MODEL_PATH.name),
            "training_images": 54284,
            "test_accuracy": float(acc_full),
            "macro_precision": float(p_full_macro),
            "macro_recall": float(r_full_macro),
            "macro_f1_score": float(f1_full_macro),
            "weighted_f1_score": float(f1_full_w),
            "evaluation_time_seconds": round(t_full, 2)
        },
        "accuracy_improvement": float(acc_full - acc_base),
        "f1_improvement": float(f1_full_w - f1_base_w),
        "per_class_full": classification_report(test_y, y_pred_full, target_names=classes, output_dict=True, zero_division=0)
    }

    report_path = DATA_DIR / "model_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[REPORT] Saved comparative report to {report_path}", flush=True)

if __name__ == "__main__":
    main()

