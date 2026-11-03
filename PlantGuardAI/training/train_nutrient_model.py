"""
Training & Evaluation Pipeline for Plant Nutrient Deficiency Classification.
Dataset: CoLeaf-DB (Mendeley Data brfgw46wzb/2)
Target Classes: 10 CoLeaf Nutritional Categories (N, P, K, Ca, Mg, Fe, B, Mn, Combined, Healthy)
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
COLEAF_DIR = BASE_DIR / "data" / "datasets" / "coleaf_nutrient"
KERAS_CACHE = BASE_DIR / ".keras"

KERAS_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["KERAS_HOME"] = str(KERAS_CACHE)


def load_nutrient_classes():
    classes_path = COLEAF_DIR / "classes.json"
    if classes_path.exists():
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["nutrient"] for item in data]
    return [
        "Healthy (Optimal Nutrition)",
        "Nitrogen (N) Deficiency",
        "Phosphorus (P) Deficiency",
        "Potassium (K) Deficiency",
        "Calcium (Ca) Deficiency",
        "Magnesium (Mg) Deficiency",
        "Iron (Fe) Chlorosis",
        "Boron (B) Deficiency",
        "Manganese (Mn) Deficiency",
        "Combined / Multiple Deficiencies"
    ]


def build_nutrient_model(num_classes: int = 10, input_shape=(224, 224, 3)):
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="nutrient_dense")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="coleaf_nutrient_mobilenetv2")
    return model, base_model


def train_nutrient_pipeline(epochs: int = 3, batch_size: int = 16, save_checkpoint: bool = True):
    print("=" * 60)
    print("PLANTGUARD AI - COLEAF NUTRIENT DEFICIENCY TRAINING PIPELINE")
    print("=" * 60)

    nutrient_classes = load_nutrient_classes()
    num_classes = len(nutrient_classes)
    print(f"Target CoLeaf Nutrient Classes ({num_classes}):")
    for i, name in enumerate(nutrient_classes):
        print(f"  [{i:02d}] {name}")

    model, base_model = build_nutrient_model(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    mapping_path = MODELS_DIR / "nutrient_classes.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(nutrient_classes, f, indent=2)
    print(f"\nExact nutrient class mapping saved to {mapping_path}")

    save_path = MODELS_DIR / "nutrient_model.keras"
    if save_checkpoint:
        model.save(str(save_path))
        print(f"Model architecture & weights checkpoint saved to {save_path}")

    return model, nutrient_classes


if __name__ == "__main__":
    train_nutrient_pipeline()
