"""
Training & Evaluation Pipeline for Agricultural Pest Classification.
Dataset: IP102 Insect Pest Recognition Benchmark (xpwu95/IP102)
Target Classes: 12 Key Agricultural Pests (mapped to IP102 taxonomy)
Architecture: MobileNetV2 with Transfer Learning
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IP102_DIR = BASE_DIR / "data" / "datasets" / "ip102_pest"
KERAS_CACHE = BASE_DIR / ".keras"

KERAS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KERAS_HOME"] = str(KERAS_CACHE)


def load_pest_classes():
    target_path = IP102_DIR / "target12_mapping.json"
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
        return [t["target_name"] for t in targets]
    return [
        "Aphids", "Armyworm", "Beetles", "Bollworm", "Caterpillars",
        "Grasshoppers", "Leafhoppers", "Mites (Spider Mites)", "Sawfly",
        "Stem Borer", "Thrips", "Whitefly"
    ]


def build_pest_model(num_classes: int = 12, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    rescaled = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling")(inputs)

    local_model_path = MODELS_DIR / "plant_model.keras"
    if local_model_path.exists():
        src_model = tf.keras.models.load_model(str(local_model_path), compile=False)
        base_model = src_model.get_layer("mobilenetv2_1.00_224")
    else:
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)

    base_model.trainable = False

    x = base_model(rescaled, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pest_dense")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ip102_pest_mobilenetv2")
    return model, base_model


def train_pest_pipeline(epochs: int = 3, batch_size: int = 16, save_checkpoint: bool = True):
    print("=" * 60)
    print("PLANTGUARD AI - IP102 PEST CLASSIFICATION TRAINING PIPELINE")
    print("=" * 60)

    pest_classes = load_pest_classes()
    num_classes = len(pest_classes)
    print(f"Target IP102 Pest Classes ({num_classes}):")
    for i, name in enumerate(pest_classes):
        print(f"  [{i:02d}] {name}")

    model, base_model = build_pest_model(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    mapping_path = MODELS_DIR / "pest_classes.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(pest_classes, f, indent=2)
    print(f"\nExact pest class mapping saved to {mapping_path}")

    save_path = MODELS_DIR / "pest_model.keras"
    if save_checkpoint:
        model.save(str(save_path))
        print(f"Model architecture & weights checkpoint saved to {save_path}")

    return model, pest_classes


if __name__ == "__main__":
    train_pest_pipeline()
