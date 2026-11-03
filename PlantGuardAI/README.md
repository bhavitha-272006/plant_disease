# PlantGuard AI - Multi-Plant Disease Detection & Plant Health Monitoring System

An explainable Computer Vision and Deep Learning system for agricultural plant disease diagnosis, pest recognition, nutrient deficiency analysis, and quantitative severity estimation.

---

## 🌟 Executive Summary & Review-Readiness Audit

**PlantGuard AI** was engineered for college review demonstration with strict scientific and academic integrity standards:
* **No Fabricated Benchmarks:** Legacy fabricated numbers (93.2% accuracy) were purged.
* **Empirical Validation:** An authentic 40-specimen PlantVillage benchmark revealed **0.0% accuracy** on the legacy checkpoint (`plant_model.keras`), definitively proving that the legacy class mapping in `class_names.json` is unverified against the model weights.
* **Untrained Models Disabled:** Untrained architecture checkpoints for Pest Detection (IP102) and Nutrient Analysis (CoLeaf) are presented as **planned capabilities**; classification predictions are explicitly disabled to prevent presenting random weights as authentic diagnoses.
* **Explainable AI (Grad-CAM):** Native Grad-CAM visualizes convolutional feature activations from the MobileNetV2 `out_relu` layer (1280 channels), clearly disclaiming that activation heatmaps do not prove biological diagnostic validity.

---

## 🛡️ AI Model Readiness Matrix

| Subsystem | Architecture / Model File | Readiness Status | Empirical Benchmark | Notes & Scientific Disclosures |
|:---|:---|:---:|:---:|:---|
| **Disease Classifier (Active)** | `models/plant_model.keras` (MobileNetV2) | **Experimental Legacy Checkpoint** | **0.0%** (40 PlantVillage test images) | Active in live detection; class mapping unverified against original training split. |
| **Disease Classifier (Target)** | `models/plantvillage_model.keras` (MobileNetV2) | **Architecture Checkpoint Ready** | *Pending Full Training* | Target 38-class architecture; training pipeline established in `training/train_disease_classifier.py`. |
| **Pest Detection (IP102)** | `models/pest_model.keras` (MobileNetV2) | **Planned Capability (Training Pending)** | *Pending Full Training* | 12 target agricultural pests mapped; untrained predictions disabled. Training pipeline: `training/train_pest_model.py`. |
| **Nutrient Deficiency (CoLeaf)** | `models/nutrient_model.keras` (MobileNetV2) | **Planned Capability (Training Pending)** | *Pending Full Training* | 10 nutritional categories mapped; untrained predictions disabled. Training pipeline: `training/train_nutrient_model.py`. |
| **Grad-CAM Explainability** | Layer: `mobilenetv2_1.00_224 / out_relu` | **Operational Tool** | N/A (Interpretability) | Visualizes convolutional layer focus; does not prove biological diagnostic correctness. |
| **Foliage & Severity CV** | Adaptive LAB/HSV Segmentation | **Operational Module** | N/A (Heuristic CV) | Heuristic visual estimate: $100 \times \frac{\text{Symptom Pixels}}{\text{Leaf Pixels}}$; not a laboratory-validated measurement. |

---

## 🗄️ Sourced Public Datasets

| Dataset | Source Repository / DOI | Classes Sourced | Full Dataset Size | Local Status |
|:---|:---|:---:|:---:|:---|
| **PlantVillage** | [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) | 38 classes (14 crops) | 54,303 lab images | Manifest + 20 curated test specimens in `data/datasets/plantvillage/` |
| **PlantDoc** | [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | 29 classes (Singh et al., 2020) | 2,598 field images | Manifest + 6 curated field specimens in `data/datasets/plantdoc/` |
| **IP102 Pest** | [xpwu95/IP102](https://github.com/xpwu95/IP102) | 102 classes (Wu et al., CVPR 2019) | 75,222 images | Official `classes.txt` + `target12_mapping.json` (0 images downloaded) |
| **CoLeaf-DB** | [Mendeley Data brfgw46wzb/2](https://data.mendeley.com/datasets/brfgw46wzb/2) | 10 classes (*Data in Brief*, 2023) | 1,006 leaf images | Manifest + 10 nutritional categories (0 images downloaded) |

---

## 🚀 Execution Instructions

### 1. Run Verification Test Suite
```powershell
cd "c:\Users\chinn\OneDrive\Desktop\plant_cv\PlantGuardAI"
.\venv\Scripts\Activate.ps1
python test_pipeline.py
```

### 2. Launch Streamlit Application
```powershell
streamlit run app.py
```
