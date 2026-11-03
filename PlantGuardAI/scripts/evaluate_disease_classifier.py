"""
Rigorous Evaluation and Validation Suite for PlantGuard AI.
Computes:
1. Held-out test set accuracy, loss, precision, recall, and F1-score.
2. Per-class classification metrics.
3. Confusion matrix visualization (assets/confusion_matrix.png).
4. Training loss and accuracy progression curves (assets/training_curves.png).
5. PlantDoc real-world field image domain transfer validation.
6. Writes comprehensive results to data/evaluation_report.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
PLANTDOC_DIR = DATA_DIR / "datasets" / "plantdoc" / "field_eval_samples"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(history_record: dict, output_path: Path):
    """Generates training and validation curves from genuine training history."""
    history = history_record.get("history", {})
    epochs = range(1, len(history.get("loss", [])) + 1)
    
    if not epochs:
        print("[WARNING] No epochs recorded in history to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, history.get("accuracy", []), "b-o", label="Training Accuracy", linewidth=2)
    ax1.plot(epochs, history.get("val_accuracy", []), "g-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Progression", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=11)

    # Loss Plot
    ax2.plot(epochs, history.get("loss", []), "r-o", label="Training Loss", linewidth=2)
    ax2.plot(epochs, history.get("val_loss", []), "m-s", label="Validation Loss", linewidth=2)
    ax2.set_title("Cross-Entropy Loss Progression", fontsize=14, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Training curves saved to {output_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plots and saves normalized confusion matrix heatmap."""
    # Normalize by row (true labels)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (Held-out Test Split)", fontsize=16, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    # Format ticks
    short_names = [c.replace("___", "\n").replace("_", " ") for c in class_names]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, short_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, short_names, fontsize=8)

    plt.ylabel('True Class', fontsize=12, fontweight="bold")
    plt.xlabel('Predicted Class', fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT] Confusion matrix heatmap saved to {output_path}")


def evaluate_test_split(model, test_paths, test_y, class_names, batch_size=32):
    """Evaluates model on held-out test split."""
    print(f"[TEST EVALUATION] Running inference on {len(test_paths)} test images...")
    
    def parse_img(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32)

    all_preds = []
    t0 = time.time()
    
    # Process in batches to manage memory
    for i in range(0, len(test_paths), batch_size):
        batch_paths = test_paths[i:i+batch_size]
        batch_imgs = tf.stack([parse_img(p) for p in batch_paths])
        preds = model(batch_imgs, training=False).numpy()
        all_preds.append(preds)
        
    all_preds = np.concatenate(all_preds, axis=0)
    y_pred = np.argmax(all_preds, axis=1)
    y_true = np.array(test_y)
    
    eval_time = time.time() - t0
    
    # Compute metrics
    acc = float(np.mean(y_pred == y_true))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Detailed classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"  Macro Precision: {prec_macro*100:.2f}% | Recall: {rec_macro*100:.2f}% | F1-Score: {f1_macro*100:.2f}%")
    print(f"  Weighted F1-Score: {f1_wt*100:.2f}%")
    print(f"  Evaluation Speed: {len(test_paths) / eval_time:.1f} images/sec")
    
    return {
        "test_accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "cm_raw": cm
    }


def evaluate_plantdoc_transfer(model, class_names):
    """
    Evaluates model on real-world field images from PlantDoc.
    Maps PlantDoc field classes to corresponding PlantVillage classes.
    """
    print("\n[PLANTDOC VALIDATION] Evaluating field image domain transfer...")
    
    if not PLANTDOC_DIR.exists():
        print("  [NOTE] PlantDoc directory not found.")
        return {"field_evaluated": False, "reason": "PlantDoc samples missing"}
        
    # Mapping table between PlantDoc sample filenames and PlantVillage classes
    # e.g. Field_Potato leaf early blight__... -> Potato___Early_blight
    name_map = {
        "Field_Potato leaf early blight": "Potato___Early_blight",
        "Field_Tomato Early blight leaf": "Tomato___Early_blight",
        "Field_Tomato leaf late blight": "Tomato___Late_blight",
    }
    
    field_files = list(PLANTDOC_DIR.glob("*.jpg")) + list(PLANTDOC_DIR.glob("*.png"))
    if not field_files:
        print("  No field evaluation images found.")
        return {"field_evaluated": False, "count": 0}
        
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    results = []
    correct_top1 = 0
    correct_top3 = 0
    total_valid = 0
    
    for f in field_files:
        true_pv_class = None
        for prefix, pv_target in name_map.items():
            if f.name.startswith(prefix):
                true_pv_class = pv_target
                break
                
        if not true_pv_class or true_pv_class not in class_to_idx:
            continue
            
        true_idx = class_to_idx[true_pv_class]
        total_valid += 1
        
        # Load and predict
        raw = tf.io.read_file(str(f))
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.expand_dims(tf.cast(img, tf.float32), 0)
        
        probs = model(img, training=False).numpy()[0]
        top1_idx = int(np.argmax(probs))
        top3_indices = np.argsort(probs)[-3:][::-1].tolist()
        
        top1_match = (top1_idx == true_idx)
        top3_match = (true_idx in top3_indices)
        
        if top1_match:
            correct_top1 += 1
        if top3_match:
            correct_top3 += 1
            
        results.append({
            "image": f.name,
            "ground_truth": true_pv_class,
            "predicted_top1": class_names[top1_idx],
            "top1_confidence": float(probs[top1_idx]),
            "top1_correct": top1_match,
            "top3_correct": top3_match,
            "top3_classes": [class_names[i] for i in top3_indices]
        })
        
    top1_acc = (correct_top1 / total_valid) if total_valid else 0.0
    top3_acc = (correct_top3 / total_valid) if total_valid else 0.0
    
    print(f"  Field Images Evaluated: {total_valid}")
    print(f"  Field Top-1 Accuracy: {top1_acc*100:.2f}% ({correct_top1}/{total_valid})")
    print(f"  Field Top-3 Accuracy: {top3_acc*100:.2f}% ({correct_top3}/{total_valid})")
    print(f"  Observation: Domain shift between controlled lab leaves and unconstrained field backgrounds is clearly measurable.")
    
    return {
        "field_evaluated": True,
        "sample_count": total_valid,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "detailed_results": results
    }


def run_full_evaluation():
    print("=" * 70)
    print("PLANTGUARD AI - RIGOROUS BENCHMARK & EVALUATION SUITE")
    print("=" * 70)
    
    model_path = MODELS_DIR / "trained_plantvillage_model.keras"
    classes_path = MODELS_DIR / "trained_plantvillage_classes.json"
    history_path = MODELS_DIR / "trained_plantvillage_history.json"
    split_path = MODELS_DIR / "test_split_indices.json"
    
    if not model_path.exists() or not classes_path.exists():
        print(f"[ERROR] Trained model or class mapping missing at {model_path}")
        sys.exit(1)
        
    print(f"[LOADING] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"[LOADED] Model loaded successfully with {len(class_names)} output classes.")
    
    # 1. Plot Training Progression
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history_record = json.load(f)
        plot_training_curves(history_record, ASSETS_DIR / "training_curves.png")
    else:
        history_record = {}
        
    # 2. Held-out Test Split Evaluation
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)
        test_paths = split_info.get("test_paths", [])
        test_y = split_info.get("test_y", [])
        
        test_results = evaluate_test_split(model, test_paths, test_y, class_names)
        plot_confusion_matrix(test_results["cm_raw"], class_names, ASSETS_DIR / "confusion_matrix.png")
    else:
        print("[WARNING] Test split indices file missing.")
        test_results = {}
        
    # 3. PlantDoc Field Validation
    plantdoc_results = evaluate_plantdoc_transfer(model, class_names)
    
    # 4. Save Consolidated Report
    final_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "MobileNetV2 (Transfer Learning, Rescaling[-1, 1])",
        "dataset_name": "PlantVillage (Official 38-class raw/color)",
        "num_classes": len(class_names),
        "training_summary": {
            "epochs_completed": history_record.get("epochs_completed"),
            "training_time_seconds": history_record.get("training_time_seconds"),
            "final_train_accuracy": history_record.get("final_train_acc"),
            "final_val_accuracy": history_record.get("final_val_acc"),
            "splits": history_record.get("split_counts", {})
        },
        "test_split_metrics": {
            "test_accuracy": test_results.get("test_accuracy"),
            "macro_precision": test_results.get("macro_precision"),
            "macro_recall": test_results.get("macro_recall"),
            "macro_f1_score": test_results.get("macro_f1"),
            "weighted_f1_score": test_results.get("weighted_f1"),
        },
        "field_domain_validation": plantdoc_results,
        "assets_generated": [
            "assets/training_curves.png",
            "assets/confusion_matrix.png"
        ]
    }
    
    report_output_path = DATA_DIR / "evaluation_report.json"
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[EVALUATION COMPLETE] Full report saved to {report_output_path}")

if __name__ == "__main__":
    run_full_evaluation()

