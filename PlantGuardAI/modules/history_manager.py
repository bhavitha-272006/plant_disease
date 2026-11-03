"""
History and Download Management Module for PlantGuard AI.

Stores genuine analysis logs in CSV and memory.
Generates comprehensive JSON diagnostic reports and CSV summaries for download.
"""

from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_CSV_PATH = BASE_DIR / "data" / "history.csv"


def load_history_records() -> List[Dict[str, Any]]:
    """Load records from CSV if available."""
    if not HISTORY_CSV_PATH.exists():
        return []
    try:
        df = pd.read_csv(HISTORY_CSV_PATH)
        return df.to_dict(orient="records")
    except Exception:
        return []


def append_history_record(record: Dict[str, Any]) -> None:
    """Append analysis entry to CSV."""
    columns = [
        "Timestamp", "Plant", "Disease", "Confidence",
        "Severity_Level", "Affected_Area_Pct", "Is_Healthy", "Is_Uncertain"
    ]
    row = {
        "Timestamp": record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "Plant": record.get("plant", "Unknown"),
        "Disease": record.get("disease", "Unknown"),
        "Confidence": f"{record.get('confidence', 0.0):.1f}%",
        "Severity_Level": record.get("severity_level", "N/A"),
        "Affected_Area_Pct": f"{record.get('affected_area_pct', 0.0):.1f}%" if record.get('affected_area_pct') is not None else "N/A",
        "Is_Healthy": record.get("is_healthy", False),
        "Is_Uncertain": record.get("is_uncertain", False)
    }

    df_new = pd.DataFrame([row])
    if HISTORY_CSV_PATH.exists() and HISTORY_CSV_PATH.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(HISTORY_CSV_PATH)
            df_combined = pd.concat([df_new, df_existing], ignore_index=True)
            df_combined.to_csv(HISTORY_CSV_PATH, index=False)
            return
        except Exception:
            pass

    df_new.to_csv(HISTORY_CSV_PATH, index=False)


def generate_json_report(record: Dict[str, Any]) -> str:
    """
    Generate fully transparent JSON audit report for downloading.
    Includes model predictions, uncertainty, CV metrics, and honesty disclosures.
    """
    mode = record.get("model_mode", "full")
    if mode in ("full", "trained") and Path("models/trained_plantvillage_full_model.keras").exists():
        disease_info = {
            "file": "models/trained_plantvillage_full_model.keras",
            "status": "Genuinely Trained & Validated Diagnostic Model",
            "empirical_benchmark": "95.26% test accuracy | Macro F1: 93.89% | Weighted F1: 95.30% (8,143 held-out test images)",
            "training_images": 54284,
            "active_classes_count": 38
        }
    elif mode == "baseline" and Path("models/trained_plantvillage_model.keras").exists():
        disease_info = {
            "file": "models/trained_plantvillage_model.keras",
            "status": "Genuinely Trained & Validated Diagnostic Model (Baseline Checkpoint)",
            "empirical_benchmark": "91.22% test accuracy | Macro F1: 89.43% (8,143 held-out test images)",
            "training_images": 4560,
            "active_classes_count": 38
        }
    else:
        disease_info = {
            "file": "models/plant_model.keras",
            "status": "Experimental Unverified Legacy Checkpoint",
            "empirical_benchmark": "0.0% accuracy on 40 PlantVillage test images (unverified legacy class mapping)",
            "active_classes_count": 15
        }

    clean_record = {
        "metadata": {
            "application": "PlantGuard AI",
            "version": "2.0-DatasetTrained",
            "generated_at": datetime.now().isoformat(),
            "academic_integrity_notice": (
                "Academic deep learning demonstration application. Plant disease diagnoses are driven by "
                "MobileNetV2 transfer learning trained directly on official PlantVillage data. Pest and nutrient modules "
                "are pending training on IP102 and CoLeaf datasets."
            )
        },
        "ai_model_readiness": {
            "disease_model": disease_info,
            "pest_model": {
                "file": "models/pest_model.keras",
                "status": "Pending Training (IP102 Dataset - 75,000+ images)",
                "inference_active": False,
                "target_classes_count": 12
            },
            "nutrient_model": {
                "file": "models/nutrient_model.keras",
                "status": "Pending Training (CoLeaf Dataset - 1,006 images)",
                "inference_active": False,
                "target_classes_count": 10
            }
        },
        "classification_result": {
            "plant": record.get("plant"),
            "disease": record.get("disease"),
            "status": "Healthy" if record.get("is_healthy") else "Diseased",
            "confidence_pct": round(record.get("confidence", 0.0), 2) if record.get("confidence") is not None else None,
            "shannon_entropy": round(record.get("entropy", 0.0), 3) if record.get("entropy") is not None else None,
            "margin_pct": round(record.get("margin", 0.0), 2) if record.get("margin") is not None else None,
            "is_uncertain": record.get("is_uncertain", False),
            "uncertainty_reasons": record.get("uncertainty_reasons", []),
            "top_5_predictions": record.get("top_5", [])
        },
        "computer_vision_and_severity": {
            "valid_leaf_segmented": record.get("valid_leaf", False),
            "leaf_coverage_pct": record.get("leaf_coverage_pct"),
            "affected_area_pct": record.get("affected_area_pct"),
            "severity_level": record.get("severity_level"),
            "formula_used": "100 * (affected_leaf_pixels / total_valid_leaf_pixels)",
            "nature_of_estimate": "Experimental visual estimate (heuristic color segmentation - not a validated agronomic severity measurement)",
            "pixel_counts": record.get("pixel_metrics", {}),
            "quality_assessment": record.get("quality", {}),
            "explainability_notice": "Grad-CAM visualizes convolutional feature activations of this checkpoint; it does not prove biological diagnostic accuracy."
        }
    }
    return json.dumps(clean_record, indent=2)
