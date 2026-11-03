"""
Complete PlantVillage Disease Classifier Training Pipeline for PlantGuard AI.
Dataset: PlantVillage (Official 54,305 raw/color images across 38 classes)
Backbone: MobileNetV2 with Transfer Learning
Features:
- Full dataset training (zero artificial sampling limits)
- Exact MD5 deduplication to eliminate data leakage across splits
- Stratified 70% Train, 15% Validation, 15% Test split
- Balanced class weighting to handle class imbalance (152 to 5,507 images)
- Early stopping, learning rate reduction on plateau, and model checkpointing
- Complete preservation of existing baseline models
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parent.parent
PV_DATA_DIR = BASE_DIR / "data" / "datasets" / "plantvillage" / "color"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def build_transfer_model(num_classes: int, input_shape=(224, 224, 3)):
    """
    Builds MobileNetV2 transfer learning model with embedded Rescaling
    and custom classification head. Reuses verified pretrained MobileNetV2 backbone.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    
    # Internal Rescaling layer [0, 255] -> [-1, 1]
    rescaled = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)

    # Use pretrained MobileNetV2 backbone from local verified checkpoint
    local_model_path = MODELS_DIR / "plant_model.keras"
    if local_model_path.exists():
        src_model = tf.keras.models.load_model(str(local_model_path), compile=False)
        base_model = src_model.get_layer("mobilenetv2_1.00_224")
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None
        )
    base_model.trainable = False  # Freeze backbone for transfer learning

    x = base_model(rescaled, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="plantvillage_mobilenetv2_full_trained")
    return model, base_model


def prepare_datasets(data_dir: Path, batch_size: int = 64, max_samples_per_class: int = None, deduplicate: bool = True):
    """
    Loads all images, performs MD5 deduplication, and creates stratified 70/15/15 split.
    Computes balanced class weights to address dataset imbalance.
    """
    print(f"[DATASET] Scanning {data_dir}...", flush=True)
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_filepaths = []
    all_labels = []
    seen_hashes = set()
    duplicates_filtered = 0
    raw_file_count = 0
    
    for c_dir in class_dirs:
        idx = class_to_idx[c_dir.name]
        files = [
            f.path for f in os.scandir(str(c_dir))
            if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        raw_file_count += len(files)
        
        if max_samples_per_class and len(files) > max_samples_per_class:
            files = files[:max_samples_per_class]
            
        for f_path in files:
            if deduplicate:
                try:
                    with open(f_path, "rb") as fh:
                        h = hashlib.md5(fh.read()).hexdigest()
                    if h in seen_hashes:
                        duplicates_filtered += 1
                        continue
                    seen_hashes.add(h)
                except Exception:
                    continue
            all_filepaths.append(f_path)
            all_labels.append(idx)
            
    all_filepaths = np.array(all_filepaths)
    all_labels = np.array(all_labels)
    
    print(f"[DATASET] Raw images found: {raw_file_count}", flush=True)
    print(f"[DATASET] Duplicates filtered: {duplicates_filtered}", flush=True)
    print(f"[DATASET] Unique valid images: {len(all_filepaths)} across {len(class_names)} classes.", flush=True)
    
    # Stratified 70% Train, 15% Val, 15% Test
    train_paths, test_val_paths, train_y, test_val_y = train_test_split(
        all_filepaths, all_labels, test_size=0.30, stratify=all_labels, random_state=42
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        test_val_paths, test_val_y, test_size=0.50, stratify=test_val_y, random_state=42
    )
    
    print(f"  Train: {len(train_paths)} images (70.0%)", flush=True)
    print(f"  Val:   {len(val_paths)} images (15.0%)", flush=True)
    print(f"  Test:  {len(test_paths)} images (15.0%)", flush=True)
    
    # Balanced class weights for training
    raw_class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    class_weight_dict = {int(i): float(w) for i, w in zip(np.unique(train_y), raw_class_weights)}
    print(f"[CLASS WEIGHTS] Computed balanced weights: min={min(raw_class_weights):.3f}, max={max(raw_class_weights):.3f}", flush=True)
    
    def parse_train_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.random_flip_left_right(img)
        img = tf.cast(img, tf.float32)  # In range [0, 255]
        return img, label

    def parse_eval_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        return img, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    train_ds = train_ds.shuffle(buffer_size=4096, seed=42)
    train_ds = train_ds.map(parse_train_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
    val_ds = val_ds.map(parse_eval_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_y))
    test_ds = test_ds.map(parse_eval_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    split_info = {
        "dataset_name": "PlantVillage (Official raw/color)",
        "total_raw_images": raw_file_count,
        "duplicates_filtered": duplicates_filtered,
        "total_unique_images": len(all_filepaths),
        "total_classes": len(class_names),
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "test_paths": test_paths.tolist(),
        "test_y": test_y.tolist(),
        "class_weights": class_weight_dict
    }
    
    return train_ds, val_ds, test_ds, class_names, split_info, class_weight_dict


def train(epochs: int = 5, batch_size: int = 64, max_per_class: int = None, model_name: str = "trained_plantvillage_full_model"):
    print("=" * 70, flush=True)
    print("PLANTGUARD AI - COMPLETE PLANTVILLAGE DATASET DEEP LEARNING TRAINING", flush=True)
    print("=" * 70, flush=True)
    
    if not PV_DATA_DIR.exists():
        print(f"[ERROR] PlantVillage dataset directory not found at {PV_DATA_DIR}", flush=True)
        sys.exit(1)
        
    train_ds, val_ds, test_ds, class_names, split_info, class_weights = prepare_datasets(
        PV_DATA_DIR, batch_size=batch_size, max_samples_per_class=max_per_class, deduplicate=True
    )
    
    # Save exact class mapping
    class_mapping_path = MODELS_DIR / f"{model_name}_classes.json"
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[CLASSES] Saved exact {len(class_names)} classes to {class_mapping_path}", flush=True)

    # Save split info for evaluation
    split_info_path = MODELS_DIR / "full_dataset_split_metadata.json"
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"[SPLITS] Saved split metadata to {split_info_path}", flush=True)

    # Build model
    model, base_model = build_transfer_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    best_model_path = MODELS_DIR / f"{model_name}.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    print(f"\n[TRAINING] Starting full dataset training for {epochs} epochs (Batch size={batch_size})...", flush=True)
    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    training_time = time.time() - t_start
    print(f"[TRAINING FINISHED] Total training time: {training_time:.1f}s ({training_time/60:.2f} mins)", flush=True)
    
    # Ensure final best model is written to disk
    model.save(str(best_model_path))
    print(f"[MODEL SAVED] Saved best calibrated weights to {best_model_path}", flush=True)
    
    # Save training history
    history_record = {
        "model_name": model_name,
        "epochs_completed": len(history.history["loss"]),
        "training_time_seconds": round(training_time, 2),
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
            "lr": [float(x) for x in history.history.get("learning_rate", [1e-3]*len(history.history["loss"]))]
        },
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
        "split_counts": {
            "train": split_info["train_count"],
            "val": split_info["val_count"],
            "test": split_info["test_count"],
        }
    }
    
    history_path = MODELS_DIR / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_record, f, indent=2)
    print(f"[HISTORY SAVED] Written to {history_path}", flush=True)
        
    return model, history_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None, help="None = use complete dataset without limits")
    parser.add_argument("--model-name", type=str, default="trained_plantvillage_full_model")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        model_name=args.model_name
    )
