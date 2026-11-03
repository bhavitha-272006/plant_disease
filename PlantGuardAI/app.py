"""
PlantGuard AI - Multi-Plant Disease Detection & Plant Health Monitoring System
Deep Learning Computer Vision Application for Agricultural Plant Pathology
Professional Two-Stage User Workflow & Explainable AI Architecture
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path

import textwrap
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.core_ai import (
    load_model,
    predict_plant_disease,
    evaluate_leaf_suitability,
    compute_gradcam,
    get_model_info,
    get_available_models,
    set_active_model_mode
)
from modules.vision import (
    analyze_severity_and_vision,
    assess_image_quality
)
from modules.pest_detector import (
    analyze_pest,
    get_pest_status
)
from modules.nutrient_analyzer import (
    analyze_nutrients,
    get_nutrient_status
)
from modules.progression import (
    compare_scans,
    compute_image_difference
)
from modules.history_manager import (
    append_history_record,
    generate_json_report,
    load_history_records
)
from modules.architecture_catalog import (
    ACTIVE_CLASSES_15,
    load_dataset_classes,
    load_dataset_manifest,
    get_architecture_summary
)
from scripts.evaluate_test_split import evaluate_plantvillage_test_split

def render_html(snippet: str):
    """Safely renders HTML in Streamlit natively without markdown code block parsing."""
    cleaned = "\n".join(line.strip() for line in snippet.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)


# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================

st.set_page_config(
    page_title="PlantGuard AI | Multi-Plant Health System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Agronomic Deep Learning Dark Navy & Teal Theme
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #03111e 0%, #061e31 50%, #07253d 100%);
    color: #e2f1f5;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}
section[data-testid="stSidebar"] {
    background: #041424 !important;
    border-right: 1px solid #0d384d;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.pg-hero {
    background: linear-gradient(135deg, #07253d 0%, #0b3954 100%);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
}
.pg-card {
    background: #082136;
    border: 1px solid #0f495e;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
}
.pg-card-glow {
    background: #08243b;
    border: 1px solid #19e6b0;
    box-shadow: 0 0 20px rgba(25, 230, 176, 0.22);
}
.pg-card-header {
    font-size: 16.5px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.pg-card-sub {
    font-size: 13px;
    color: #8daec0;
    line-height: 1.5;
}
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid transparent;
}
.pill-green {
    background: rgba(25, 230, 176, 0.15);
    color: #19e6b0;
    border-color: #19e6b0;
}
.pill-red {
    background: rgba(255, 92, 108, 0.15);
    color: #ff5c6c;
    border-color: #ff5c6c;
}
.pill-amber {
    background: rgba(255, 194, 71, 0.15);
    color: #ffc247;
    border-color: #ffc247;
}
.pill-cyan {
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border-color: #38bdf8;
}
.pill-gray {
    background: rgba(140, 160, 175, 0.15);
    color: #94a3b8;
    border-color: #64748b;
}
.metric-box {
    background: #061e31;
    border: 1px solid #0d4154;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #8daec0;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: #19e6b0;
}
.warning-banner {
    background: rgba(70, 40, 10, 0.35);
    border-left: 4px solid #ffc247;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 13px;
    color: #ffe6b0;
    margin-bottom: 14px;
}
.info-banner {
    background: rgba(14, 58, 82, 0.4);
    border-left: 4px solid #38bdf8;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 13px;
    color: #c8e3ed;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #061c2c;
    padding: 6px;
    border-radius: 10px;
    border: 1px solid #0d384d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: #8daec0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #0f495e !important;
    color: #ffffff !important;
    font-weight: 600;
}
div.stButton > button:first-child {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 15px;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "input"  # "input" or "dashboard"

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "image_source_desc" not in st.session_state:
    st.session_state.image_source_desc = ""

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

if "history_scans" not in st.session_state:
    st.session_state.history_scans = []


# ==========================================
# SIDEBAR NAVIGATION & MODEL SELECTOR
# ==========================================

avail_models = get_available_models()
available_modes = [k for k in ["full", "baseline", "legacy"] if k in avail_models]

if "selected_model_mode" not in st.session_state or st.session_state.selected_model_mode not in available_modes:
    st.session_state.selected_model_mode = "full" if "full" in available_modes else available_modes[0]

set_active_model_mode(st.session_state.selected_model_mode)

with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
        <span style="font-size:32px;">🌿</span>
        <div>
            <h2 style="margin:0; font-size:22px; color:#ffffff;">PlantGuard AI</h2>
            <span style="font-size:12px; color:#19e6b0; font-weight:600;">Deep Learning Health Suite</span>
        </div>
    </div>
    """)

    nav_options = [
        "📷 Live Camera",
        "⬆️ Upload Image",
        "📊 Analysis Dashboard",
        "📈 Disease Progression",
        "🧪 Model Evaluation",
        "📋 Supported Classes",
        "🕒 Recent History",
        "ℹ️ About Project"
    ]

    # Map current workflow state to default nav
    current_nav_index = 1
    if st.session_state.workflow_stage == "dashboard" and st.session_state.current_analysis is not None:
        current_nav_index = 2

    nav_selection = st.radio(
        "Navigation Menu",
        nav_options,
        index=current_nav_index,
        label_visibility="collapsed"
    )

    # Sync navigation with workflow stage
    if nav_selection in ["📷 Live Camera", "⬆️ Upload Image"]:
        if st.session_state.workflow_stage != "input":
            st.session_state.workflow_stage = "input"
    elif nav_selection == "📊 Analysis Dashboard":
        if st.session_state.current_analysis is not None:
            st.session_state.workflow_stage = "dashboard"

    st.markdown("<hr style='border-color:#0d384d; margin:14px 0;'>", unsafe_allow_html=True)

    # Compact AI Engine Status Indicator
    st.markdown("**🧠 Active AI Model Engine**")
    
    MODE_INFO = {
        "full": {
            "label": "🌿 Full-Dataset MobileNetV2 (54k Images, 38 Classes)",
            "file": "trained_plantvillage_full_model.keras",
            "classes": 38,
            "badge": '<span class="pill pill-green">ACTIVE & VALIDATED</span> 95.26% Test Acc (54k images)'
        },
        "baseline": {
            "label": "📊 Baseline MobileNetV2 (4.5k Images, 38 Classes)",
            "file": "trained_plantvillage_model.keras",
            "classes": 38,
            "badge": '<span class="pill pill-cyan">COMPARATIVE BASELINE</span> 91.22% Test Acc (4.5k subset)'
        },
        "legacy": {
            "label": "⚠️ Legacy Checkpoint (15 Classes - Experimental)",
            "file": "plant_model.keras",
            "classes": 15,
            "badge": '<span class="pill pill-amber">EXPERIMENTAL BASELINE</span> 15 Classes (Unverified)'
        }
    }
    
    selector_labels = [MODE_INFO[k]["label"] for k in available_modes]
    current_mode_idx = available_modes.index(st.session_state.selected_model_mode) if st.session_state.selected_model_mode in available_modes else 0
    
    chosen_label = st.selectbox(
        "Classifier Checkpoint",
        selector_labels,
        index=current_mode_idx,
        label_visibility="collapsed"
    )
    
    new_mode = available_modes[selector_labels.index(chosen_label)]
    if new_mode != st.session_state.selected_model_mode:
        st.session_state.selected_model_mode = new_mode
        set_active_model_mode(new_mode)
        st.rerun()
    else:
        set_active_model_mode(new_mode)
        
    cfg = MODE_INFO[new_mode]
    render_html(f"""
    <div style="font-size:11.5px; color:#8daec0; margin-top:4px; line-height:1.45;">
        {cfg['badge']}<br>
        <span style="font-size:10.5px; color:#628a9e;">Loaded: <code>{cfg['file']}</code> ({cfg['classes']} classes)</span>
    </div>
    """)

    st.markdown("<hr style='border-color:#0d384d; margin:14px 0;'>", unsafe_allow_html=True)
    st.caption("College Review Demonstration • Deep Learning Computer Vision")


# ==========================================
# STAGE 1: WELCOME & IMAGE INPUT PAGE
# ==========================================

if nav_selection in ["📷 Live Camera", "⬆️ Upload Image"] and st.session_state.workflow_stage == "input":

    render_html("""
    <div class="pg-hero">
        <div style="display:flex; align-items:center; gap:16px;">
            <span style="font-size:48px;">🌿</span>
            <div>
                <h1 style="margin:0; font-size:30px; color:#ffffff;">PlantGuard AI</h1>
                <div style="color:#19e6b0; font-size:16px; font-weight:600; margin-top:4px;">
                    Intelligent Plant Health Analysis & Deep Learning Disease Detection
                </div>
            </div>
        </div>
        <p style="color:#c8e3ed; font-size:14px; margin-top:14px; line-height:1.6; max-width:900px;">
            PlantGuard AI uses deep convolutional transfer learning (MobileNetV2) and explainable computer vision (Grad-CAM)
            to diagnose crop pathogens, estimate symptom severity, and monitor longitudinal recovery across multiple plant species.
        </p>
    </div>
    """)

    st.markdown("### 📥 Stage 1: Select Plant Leaf Specimen")
    st.markdown("Choose an input method below to provide a single plant leaf photograph:")

    render_html("""
    <div style="background:rgba(56, 189, 248, 0.08); border-left:4px solid #38bdf8; border-radius:6px; padding:10px 14px; margin-bottom:16px; font-size:13px; color:#c8e3ed;">
        <b>🌿 Specimen Scope Notice:</b> PlantGuard AI is trained to identify diseases from leaf images. Fruit, stem, flower, root, and whole-plant images are outside its validated scope.
    </div>
    """)

    col_cam, col_upload = st.columns(2, gap="large")

    temp_image = None
    temp_desc = ""

    with col_cam:
        render_html("""
        <div class="pg-card">
            <div class="pg-card-header">📷 Live Camera Capture</div>
            <div class="pg-card-sub" style="margin-bottom:12px;">
                Point your device camera directly at a single crop leaf against a neutral background.
            </div>
        </div>
        """)

        cam_file = st.camera_input("Open Camera", key="webcam_input")
        if cam_file is not None:
            temp_image = ImageOps.exif_transpose(Image.open(cam_file)).convert("RGB")
            temp_desc = "Live Camera Capture"
            st.session_state.selected_image = temp_image
            st.session_state.image_source_desc = temp_desc

    with col_upload:
        render_html("""
        <div class="pg-card">
            <div class="pg-card-header">⬆️ Upload Leaf Photograph</div>
            <div class="pg-card-sub" style="margin-bottom:12px;">
                Upload a high-resolution JPG, JPEG, or PNG photograph of a symptomatic or healthy leaf.
            </div>
        </div>
        """)

        up_file = st.file_uploader(
            "Upload Image File",
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader_input",
            help="Supports high-resolution crop leaf imagery."
        )
        if up_file is not None:
            temp_image = ImageOps.exif_transpose(Image.open(up_file)).convert("RGB")
            temp_desc = f"Uploaded: {up_file.name}"
            st.session_state.selected_image = temp_image
            st.session_state.image_source_desc = temp_desc

    # Curated Demonstration Specimen Gallery (for easy testing during college review)
    with st.expander("🖼️ Or Select a Verified Specimen from Sourced Datasets (PlantVillage & PlantDoc)", expanded=False):
        st.caption("Conveniently select an authentic specimen to test the diagnostic pipeline without a live leaf:")
        pv_samples_dir = BASE_DIR / "data" / "datasets" / "plantvillage" / "test_samples"
        pd_samples_dir = BASE_DIR / "data" / "datasets" / "plantdoc" / "field_eval_samples"
        synth_dir = BASE_DIR / "data" / "sample_leaves"

        gallery_items = {}
        if synth_dir.exists():
            if (synth_dir / "sample_early_blight_leaf.jpg").exists():
                gallery_items["[Curated Demo] Early Blight Tomato Specimen"] = synth_dir / "sample_early_blight_leaf.jpg"
            if (synth_dir / "sample_healthy_leaf.jpg").exists():
                gallery_items["[Curated Demo] Healthy Green Specimen"] = synth_dir / "sample_healthy_leaf.jpg"

        if pv_samples_dir.exists():
            for p in sorted(pv_samples_dir.glob("*.JPG"))[:6]:
                lbl = p.name.split("__")[0].replace("___", " - ").replace("_", " ")
                gallery_items[f"[PlantVillage Lab Specimen] {lbl}"] = p

        if pd_samples_dir.exists():
            for p in sorted(pd_samples_dir.glob("*.jpg"))[:4]:
                clean_n = p.name.replace("Field_", "").split("__")[0]
                gallery_items[f"[PlantDoc Field Specimen] {clean_n}"] = p

        if gallery_items:
            sel_choice = st.selectbox("Choose Verification Specimen", list(gallery_items.keys()))
            if st.button("Load Selected Gallery Specimen"):
                sel_path = gallery_items[sel_choice]
                st.session_state.selected_image = ImageOps.exif_transpose(Image.open(sel_path)).convert("RGB")
                st.session_state.image_source_desc = f"Gallery: {sel_choice}"
                st.rerun()

    # Image Preview & Prominent Analyze Button
    active_img = st.session_state.selected_image
    if active_img is not None:
        st.markdown("<hr style='border-color:#0d384d; margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Image Preview & Action")

        prev_col1, prev_col2 = st.columns([1, 2], gap="large")
        with prev_col1:
            st.image(active_img, caption=st.session_state.image_source_desc, use_container_width=True)
        with prev_col2:
            render_html(f"""
            <div class="pg-card">
                <div class="pg-card-header">Specimen Selected</div>
                <div class="pg-card-sub">
                    <b>Source:</b> {st.session_state.image_source_desc}<br>
                    <b>Dimensions:</b> {active_img.width} × {active_img.height} pixels (RGB)<br>
                    <b>Target Preprocessing:</b> Bilinear resize to 224 × 224 × 3 float32 [0.0, 255.0]<br>
                    <b>Active Checkpoint:</b> <code>{get_model_info()['model_file']}</code> ({get_model_info()['num_classes']} classes)<br>
                    <b>Architecture:</b> <code>{get_model_info()['architecture']}</code>
                </div>
            </div>
            """)

            # Specimen Type Verification Selector (Required Step)
            specimen_type = st.radio(
                "📋 Specimen Type Verification (Required):",
                [
                    "Single leaf",
                    "Whole plant / canopy",
                    "Fruit",
                    "Stem / flower / root",
                    "Not sure"
                ],
                index=0,
                help="The 38-class PlantVillage model was trained exclusively on single leaf specimens. Select the specimen category depicted in your image."
            )
            st.session_state.selected_specimen_type = specimen_type

            # Automated Suitability Evaluation Gate
            suit_res = evaluate_leaf_suitability(active_img, user_specimen_type=specimen_type)

            if suit_res["is_permitted"]:
                st.success("✅ Valid specimen: Single leaf confirmed. Supported by the 38-class MobileNetV2 classifier.")
                btn_run, btn_clear = st.columns([2, 1])
                with btn_run:
                    if st.button("🔬 Analyze Leaf Specimen", use_container_width=True):
                        with st.spinner("Executing Deep Learning Diagnostic Pipeline..."):
                            # Execute Core AI Inference with specimen validation
                            pred = predict_plant_disease(active_img, mode=st.session_state.selected_model_mode, specimen_type=specimen_type)
                            if pred.get("is_blocked"):
                                analysis_res = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "source": st.session_state.image_source_desc,
                                    "specimen_type": specimen_type,
                                    "suitability_status": pred.get("suitability_status", "DOMAIN_SHIFT_OOD"),
                                    "suitability_reason": pred.get("suitability_reason", "Specimen outside validated scope."),
                                    "is_blocked": True,
                                    "original_image": active_img,
                                    "plant": pred.get("plant", "Diagnosis Withheld"),
                                    "disease": pred.get("disease", "Specimen Outside Supported Distribution"),
                                    "raw_class": "N/A",
                                    "is_healthy": False,
                                    "confidence": 0.0,
                                    "entropy": pred.get("entropy", 0.0),
                                    "margin": 0.0,
                                    "is_uncertain": True,
                                    "uncertainty_reasons": pred.get("uncertainty_reasons", []),
                                    "top_5": [],
                                    "model_mode": pred.get("model_mode", "full"),
                                    "model_readiness": pred.get("model_readiness"),
                                    "gradcam_image": None,
                                    "gradcam_raw": None,
                                    "prototype_similarity": pred.get("prototype_similarity", 0.0),
                                    "vision": {"severity_level": "N/A", "affected_area_pct": 0.0, "quality": {}, "valid": False},
                                    "pest": {"status": "SKIPPED"},
                                    "nutrient": {"status": "SKIPPED"}
                                }
                                st.session_state.current_analysis = analysis_res
                                st.session_state.workflow_stage = "dashboard"
                                st.rerun()
                            else:
                                gradcam_overlay, gradcam_raw = compute_gradcam(active_img, pred["class_index"], mode=pred.get("model_mode"))
                                vision = analyze_severity_and_vision(active_img)
                                pest = analyze_pest(active_img)
                                nutrient = analyze_nutrients(active_img)

                                analysis_res = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "source": st.session_state.image_source_desc,
                                    "specimen_type": specimen_type,
                                    "suitability_status": pred.get("suitability_status", "VALID_LEAF_SPECIMEN"),
                                    "suitability_reason": pred.get("suitability_reason", "Verified Single Leaf"),
                                    "is_blocked": False,
                                    "original_image": active_img,
                                    "plant": pred["plant"],
                                    "disease": pred["disease"],
                                    "raw_class": pred["raw_class"],
                                    "is_healthy": pred["is_healthy"],
                                    "confidence": pred["confidence"],
                                    "entropy": pred["entropy"],
                                    "margin": pred["margin"],
                                    "is_uncertain": pred["is_uncertain"],
                                    "uncertainty_reasons": pred["uncertainty_reasons"],
                                    "top_5": pred["top_5"],
                                    "model_mode": pred.get("model_mode", "legacy"),
                                    "model_readiness": pred.get("model_readiness"),
                                    "gradcam_image": gradcam_overlay,
                                    "gradcam_raw": gradcam_raw,
                                    "prototype_similarity": pred.get("prototype_similarity", 1.0),
                                    "vision": vision,
                                    "pest": pest,
                                    "nutrient": nutrient
                                }

                                st.session_state.current_analysis = analysis_res
                                st.session_state.workflow_stage = "dashboard"

                                # Record in history only when genuine diagnosis was permitted
                                rec = {
                                    "timestamp": analysis_res["timestamp"],
                                    "plant": analysis_res["plant"],
                                    "disease": analysis_res["disease"],
                                    "confidence": analysis_res["confidence"],
                                    "severity_level": vision.get("severity_level", "N/A"),
                                    "affected_area_pct": vision.get("affected_area_pct"),
                                    "is_healthy": analysis_res["is_healthy"],
                                    "is_uncertain": analysis_res["is_uncertain"],
                                    "entropy": analysis_res["entropy"],
                                    "margin": analysis_res["margin"],
                                    "original_image": active_img,
                                    "annotated_image": vision.get("annotated"),
                                    "top_5": analysis_res["top_5"],
                                    "pixel_metrics": vision.get("metrics"),
                                    "quality": vision.get("quality")
                                }
                                st.session_state.history_scans.insert(0, rec)
                                append_history_record(rec)
                                st.rerun()

                with btn_clear:
                    if st.button("❌ Clear Image", use_container_width=True):
                        st.session_state.selected_image = None
                        st.session_state.image_source_desc = ""
                        st.session_state.current_analysis = None
                        st.rerun()
            else:
                # Diagnosis Withheld / Blocked
                render_html(f"""
                <div style="background:rgba(255, 92, 108, 0.12); border:1px solid #ff5c6c; border-radius:10px; padding:14px 16px; margin:12px 0;">
                    <div style="font-size:15px; font-weight:700; color:#ff5c6c; display:flex; align-items:center; gap:8px;">
                        <span>⛔ Diagnostic Execution Blocked: {suit_res['status']}</span>
                    </div>
                    <div style="font-size:13px; color:#e2e8f0; margin-top:8px; line-height:1.5;">
                        {suit_res['reason']}<br>
                        <b>PlantGuard AI is trained exclusively on individual leaf specimens (PlantVillage benchmark).</b>
                    </div>
                    <div style="font-size:12px; color:#ffc247; margin-top:8px; border-top:1px dashed rgba(255, 92, 108, 0.3); padding-top:6px;">
                        ℹ️ Disease classification, severity estimation, and history recording are strictly withheld for non-leaf inputs to prevent invalid or misleading predictions. Please provide a clear photograph of a single crop leaf.
                    </div>
                </div>
                """)
                if st.button("❌ Clear Image", use_container_width=True):
                    st.session_state.selected_image = None
                    st.session_state.image_source_desc = ""
                    st.session_state.current_analysis = None
                    st.rerun()
                    st.session_state.selected_image = None
                    st.session_state.image_source_desc = ""
                    st.session_state.current_analysis = None
                    st.rerun()


# ==========================================
# STAGE 2: PROFESSIONAL ANALYSIS DASHBOARD
# ==========================================

elif nav_selection == "📊 Analysis Dashboard" or st.session_state.workflow_stage == "dashboard":

    cur = st.session_state.current_analysis

    if cur is None and st.session_state.selected_image is not None:
        st.session_state.workflow_stage = "input"
        st.rerun()

    if cur is None:
        st.info("ℹ️ No leaf specimen currently analyzed. Please return to Image Input to capture or upload an image.")
        if st.button("⬅️ Return to Image Input"):
            st.session_state.workflow_stage = "input"
            st.rerun()
    else:
        vis = cur["vision"]
        active_img = cur["original_image"]
        _is_uncertain = cur.get("is_uncertain", False)
        _cur_mode = cur.get("model_mode", "legacy")

        # Top Action Bar
        act_col1, act_col2, act_col3 = st.columns([2.2, 1, 1])
        with act_col1:
            render_html(f"""
            <div style="display:flex; align-items:center; gap:10px;">
                <h2 style="margin:0; font-size:24px; color:#ffffff;">🌿 Specimen Analysis Dashboard</h2>
                <span style="font-size:12px; color:#8daec0;">({cur['timestamp']})</span>
            </div>
            """)
        with act_col2:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                st.session_state.workflow_stage = "input"
                st.session_state.selected_image = None
                st.session_state.current_analysis = None
                st.rerun()
        with act_col3:
            if st.button("⬅️ Back to Input", use_container_width=True):
                st.session_state.workflow_stage = "input"
                st.rerun()

        st.markdown("<hr style='border-color:#0d384d; margin:10px 0 16px 0;'>", unsafe_allow_html=True)

        # ====================================================
        # MAIN DASHBOARD: BALANCED TWO-COLUMN LAYOUT
        # Left: Specimen Analysis | Right: Diagnosis Results
        # ====================================================
        col_specimen, col_diagnosis = st.columns([1.05, 1.0], gap="large")

        # ----------------------------------------------------
        # LEFT COLUMN: SPECIMEN ANALYSIS
        # ----------------------------------------------------
        with col_specimen:
            q = vis.get("quality", {})
            blur_status = "⚠️ Blurry Image" if q.get("is_blurry", False) else "✅ Adequate Sharpness"
            lap_val = f"{q.get('laplacian_variance', 0.0):.1f}"
            foliage_pct = f"{vis.get('leaf_coverage_pct', 'N/A')}"
            bright_val = f"{q.get('mean_brightness', 0.0):.0f}"
            contrast_val = f"{q.get('std_contrast', 0.0):.1f}"

            render_html(f"""
            <div class="pg-card">
                <div class="pg-card-header">📷 Specimen Analysis & Imaging</div>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; font-size:12px; color:#8daec0;">
                    <span class="pill pill-cyan">Source: {cur.get('source', 'Uploaded Leaf')}</span>
                    <span class="pill pill-gray">{active_img.width} × {active_img.height} px</span>
                    <span class="pill {'pill-green' if not q.get('is_blurry') else 'pill-amber'}">{blur_status} (Laplacian: {lap_val})</span>
                    <span class="pill pill-cyan">Foliage Coverage: {foliage_pct}%</span>
                </div>
            </div>
            """)

            if cur.get("gradcam_raw") is not None and cur.get("gradcam_image") is not None:
                tab_orig, tab_cam, tab_overlay = st.tabs([
                    "🌿 Original Leaf",
                    "🔥 Grad-CAM Heatmap",
                    "🎯 Superimposed Focus"
                ])
                with tab_orig:
                    st.image(active_img, caption="Original Input Specimen", use_container_width=True)
                with tab_cam:
                    st.image(cur["gradcam_raw"], caption="Convolutional Activation Heatmap (out_relu)", use_container_width=True)
                with tab_overlay:
                    st.image(cur["gradcam_image"], caption="Grad-CAM Activation Superimposed on Specimen", use_container_width=True)
            else:
                st.image(active_img, caption="Original Input Specimen", use_container_width=True)
                st.info("ℹ️ Grad-CAM neural attention heatmap is disabled because diagnosis was withheld (Domain Shift / OOD safeguard).")

            render_html(f"""
            <div style="background:#061e31; border:1px solid #0d4154; border-radius:8px; padding:10px 14px; margin-top:10px; font-size:12px; color:#8daec0;">
                <b>Optical Quality Screening:</b> Sharpness Variance: <code>{lap_val}</code> (threshold: 60.0) •
                Illumination: <code>{bright_val}/255</code> • Contrast (std): <code>{contrast_val}</code>
            </div>
            """)

        # ----------------------------------------------------
        # RIGHT COLUMN: DIAGNOSIS RESULTS
        # ----------------------------------------------------
        with col_diagnosis:
            _is_blocked = cur.get("is_blocked", False)
            if _is_blocked:
                proto_sim = cur.get("prototype_similarity", 0.0)
                reasons = cur.get("uncertainty_reasons", [])
                reasons_li = "".join(f"<li style='margin-bottom:3px;'>{r}</li>" for r in reasons)
                render_html(f"""
                <div class="pg-card pg-card-glow" style="border-left:4px solid #ff5c6c;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div class="pg-card-header" style="margin:0;">🔬 Diagnosis Results</div>
                        <span class="pill pill-red">🛑 DIAGNOSIS WITHHELD</span>
                    </div>
                    <div style="font-size:14px; color:#8daec0; margin-bottom:2px;">
                        Host Crop: <strong style="color:#ffffff; font-size:16px;">Diagnosis Withheld (OOD)</strong>
                    </div>
                    <div style="font-size:20px; font-weight:700; color:#ff5c6c; margin:4px 0 8px 0;">
                        {cur['disease']}
                    </div>
                    <div style="background:rgba(255, 92, 108, 0.12); border:1px solid #ff5c6c; border-radius:8px; padding:10px 12px; margin:10px 0;">
                        <div style="font-weight:700; color:#ff5c6c; font-size:12.5px;">⚠️ Out-of-Distribution / Domain Shift Protection</div>
                        <div style="font-size:11.5px; color:#e2e8f0; margin-top:4px; line-height:1.4;">
                            The diagnostic engine withheld prediction to protect against overconfident misdiagnosis:
                            <ul style="margin:4px 0 4px 16px; padding:0; color:#ffc247;">
                                {reasons_li}
                            </ul>
                        </div>
                    </div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin:10px 0;">
                        <div class="metric-box">
                            <div class="metric-label">Softmax Confidence</div>
                            <div class="metric-value" style="color:#8daec0;">0.0% (Suppressed)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Prototype Cosine Sim</div>
                            <div class="metric-value" style="color:#ff5c6c;">{proto_sim:.3f} (< 0.20)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Shannon Entropy</div>
                            <div class="metric-value" style="color:#38bdf8;">{cur['entropy']:.2f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Diagnostic Status</div>
                            <div class="metric-value" style="color:#ffc247;">Withheld (Safe)</div>
                        </div>
                    </div>
                    <div style="font-size:11.5px; color:#c8e3ed; background:rgba(56, 189, 248, 0.08); border:1px solid #0d4154; border-radius:6px; padding:8px 10px; margin-top:8px; line-height:1.4;">
                        <b>Root Cause & Dataset Limitation:</b> The 38-class MobileNetV2 was trained exclusively on single excised leaves against uniform laboratory backgrounds (PlantVillage). Natural field conditions, background foliage, soil, or fruit organs exhibit strong domain shift, forcing uncalibrated softmax overconfidence.
                        <br><br>
                        <b>Engineering Solution:</b> Reliable in-field diagnostic performance requires training/fine-tuning on authentic in-the-wild agricultural datasets such as <b>PlantDoc</b>.
                    </div>
                </div>
                """)
            else:
                if _is_uncertain:
                    status_pill = '<span class="pill pill-amber">⚠️ UNCERTAIN CANDIDATE</span>'
                else:
                    status_pill = (
                        '<span class="pill pill-green">PREDICTED HEALTHY</span>'
                        if cur["is_healthy"]
                        else '<span class="pill pill-red">PREDICTED DISEASE</span>'
                    )

                if _cur_mode == "full":
                    model_name_disp = "MobileNetV2 (54k Images, 38 Classes)"
                    checkpoint_disp = "trained_plantvillage_full_model.keras"
                elif _cur_mode == "baseline":
                    model_name_disp = "Baseline MobileNetV2 (4.5k Images, 38 Classes)"
                    checkpoint_disp = "trained_plantvillage_model.keras"
                else:
                    model_name_disp = "Legacy Checkpoint (15 Classes - Experimental)"
                    checkpoint_disp = "plant_model.keras"

                cond_color = "#19e6b0" if cur["is_healthy"] else "#ff5c6c"
                disease_text = cur["disease"]
                if _is_uncertain:
                    disease_text = f"{disease_text} (Uncertain Candidate)"

                # Build uncertainty notice if triggered
                uncertainty_html = ""
                if _is_uncertain:
                    reasons = cur.get("uncertainty_reasons", [])
                    reasons_li = "".join(f"<li style='margin-bottom:2px;'>{r}</li>" for r in reasons)
                    uncertainty_html = f"""
                    <div style="background:rgba(255, 194, 71, 0.12); border:1px solid #ffc247; border-radius:8px; padding:10px 12px; margin:10px 0;">
                        <div style="font-weight:700; color:#ffc247; font-size:12.5px;">⚠️ Diagnostic Uncertainty Notice</div>
                        <div style="font-size:11.5px; color:#e2e8f0; margin-top:4px; line-height:1.4;">
                            This prediction does not meet statistical diagnostic confidence criteria:
                            <ul style="margin:4px 0 4px 16px; padding:0; color:#ffc247;">
                                {reasons_li}
                            </ul>
                            <div style="font-size:11px; color:#94a3b8; border-top:1px dashed rgba(255,194,71,0.3); padding-top:4px; margin-top:4px;">
                                <b>Model Uncertainty vs. Unsupported Species:</b> Statistical ambiguity between known classes vs. unlisted plant species. Uploading an unsupported plant forces a nearest-match prediction.
                            </div>
                        </div>
                    </div>
                    """

                proto_sim_val = cur.get('prototype_similarity')
                proto_metric_html = f"<div class='metric-box'><div class='metric-label'>Prototype Cosine Sim</div><div class='metric-value' style='color:#10b981;'>{proto_sim_val:.2f}</div></div>" if proto_sim_val is not None else ""

                render_html(f"""
                <div class="pg-card pg-card-glow">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div class="pg-card-header" style="margin:0;">🔬 Diagnosis Results</div>
                        {status_pill}
                    </div>
                    <div style="font-size:14px; color:#8daec0; margin-bottom:2px;">
                        Host Crop: <strong style="color:#ffffff; font-size:16px;">{cur['plant']}</strong>
                    </div>
                    <div style="font-size:22px; font-weight:700; color:{cond_color}; margin:4px 0 8px 0;">
                        {disease_text}
                    </div>
                    <div style="font-size:11px; color:#ffc247; background:rgba(255,194,71,0.08); border-left:3px solid #ffc247; padding:6px 10px; border-radius:4px; margin-bottom:10px;">
                        ⚠️ <b>Shared Architecture Notice:</b> Host crop and disease category are derived <i>jointly from the same single 38-class MobileNetV2 classification head</i> (Crop___Disease), not independent models.
                    </div>
                    {uncertainty_html}
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin:10px 0;">
                        <div class="metric-box">
                            <div class="metric-label">Softmax Confidence</div>
                            <div class="metric-value">{cur['confidence']:.1f}%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Shannon Entropy</div>
                            <div class="metric-value" style="color:#38bdf8;">{cur['entropy']:.2f}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Top-2 Margin</div>
                            <div class="metric-value" style="color:#10b981;">{cur.get('margin', 0.0):.1f}%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Class Index</div>
                            <div class="metric-value" style="color:#ffc247;">#{cur['raw_class']}</div>
                        </div>
                    </div>
                    <div style="font-size:12px; color:#8daec0; margin-top:8px;">
                        <b>Active Model:</b> {model_name_disp}<br>
                        <b>Checkpoint File:</b> <code>{checkpoint_disp}</code>
                    </div>
                    <div style="font-size:11px; color:#628a9e; margin-top:8px; border-top:1px solid #0d384d; padding-top:6px;">
                        ℹ️ <b>Indicative Prediction:</b> Automated deep learning screening output based on visual features. This is not a confirmed agricultural diagnosis.
                    </div>
                </div>
                """)

                with st.expander("📊 View Top-5 Class Probability Distribution", expanded=False):
                    top_5_data = cur["top_5"]
                    df_top5 = pd.DataFrame([
                        {"Class": item["display_name"], "Confidence (%)": round(item["percentage"], 1)}
                        for item in top_5_data
                    ])
                    st.dataframe(df_top5, hide_index=True, use_container_width=True)

        # ====================================================
        # BELOW THE MAIN DASHBOARD: SUPPORTING ANALYSIS
        # ====================================================
        st.markdown("<hr style='border-color:#0d384d; margin:24px 0 20px 0;'>", unsafe_allow_html=True)

        if _is_blocked:
            st.info("ℹ️ Secondary diagnostic modules (Grad-CAM neural attention heatmaps, heuristic damage contours, severity scoring, and agronomic treatment protocols) are withheld for this specimen to prevent unverified clinical guidance on out-of-distribution inputs.")
            st.markdown("<hr style='border-color:#0d384d; margin:20px 0 14px 0;'>", unsafe_allow_html=True)
            st.caption("ℹ️ Longitudinal Progression: Specimen diagnosis was withheld due to domain shift or non-leaf features. Not recorded in persistent diagnostic history.")
        else:


            # ----------------------------------------------------
            # SECTION A: VISUAL EXPLANATION (GRAD-CAM)
            # ----------------------------------------------------
            st.markdown("### 🧠 Visual Explanation (Explainable AI Grad-CAM)")
            gc_col1, gc_col2, gc_col3 = st.columns(3)
            with gc_col1:
                st.image(cur["original_image"], caption="1. Original Input Leaf", use_container_width=True)
            with gc_col2:
                st.image(cur["gradcam_raw"], caption="2. MobileNetV2 Activation Focus (out_relu)", use_container_width=True)
            with gc_col3:
                st.image(cur["gradcam_image"], caption="3. Activation Superimposed on Specimen", use_container_width=True)

            peak_act = f"{np.max(cur['gradcam_raw']):.2f}"
            render_html(f"""
            <div class="pg-card" style="margin-top:10px;">
                <div class="pg-card-header">Grad-CAM Interpretability Scope & Limitations</div>
                <div class="pg-card-sub">
                    Grad-CAM calculates the gradient of the predicted class score with respect to feature maps of the
                    final convolutional layer (<code>mobilenetv2 / out_relu</code>, 1280 feature channels).
                    Peak normalized activation focus: <code>{peak_act}</code> on target class <code>#{cur.get('raw_class', 'N/A')}</code>.<br>
                    <span style="color:#ffc247;">
                        ⚠️ <b>Scientific Scope:</b> Grad-CAM visualizes where convolutional filters responded most strongly to drive the model's decision.
                        It illustrates neural attention and is <b>not verified proof of biological disease-lesion boundary localization</b>.
                    </span>
                </div>
            </div>
            """)

            # ----------------------------------------------------
            # SECTIONS B & C: HEURISTIC DAMAGE ANALYSIS & SEVERITY
            # ----------------------------------------------------
            st.markdown("### 📐 Surface Damage Analysis & Severity Estimation")
            dam_col1, dam_col2 = st.columns([1, 1], gap="medium")
            with dam_col1:
                if vis.get("valid"):
                    st.image(vis["annotated"], caption="Annotated Symptom Contours & Bounding Boxes", use_container_width=True)
                    render_html(f"""
                    <div class="pg-card">
                        <div class="pg-card-header">Heuristic Computer Vision Damage Analysis</div>
                        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin:8px 0;">
                            <div class="metric-box">
                                <div class="metric-label">Estimated Affected Area</div>
                                <div class="metric-value" style="color:#ff5c6c;">{vis.get('affected_area_pct', 0.0):.1f}%</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Symptom Contours</div>
                                <div class="metric-value" style="color:#ffc247;">{vis.get('symptom_contour_count', 0)}</div>
                            </div>
                        </div>
                        <div class="pg-card-sub">
                            Calculated as: <code>100 × (Symptom Pixels / Segmented Leaf Pixels)</code> via adaptive HSV/LAB color thresholding.
                        </div>
                    </div>
                    """)
            with dam_col2:
                if vis.get("valid"):
                    st.image(vis["leaf_mask"], caption="Segmented Foliage Mask (Otsu & LAB)", use_container_width=True)
                    sev_level = vis.get("severity_level", "N/A")
                    sev_color = vis.get("severity_color", "gray")
                    render_html(f"""
                    <div class="pg-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <div class="pg-card-header" style="margin:0;">🩺 Experimental Severity Estimate</div>
                            <span class="pill pill-{sev_color}">{sev_level.upper()} (ESTIMATE)</span>
                        </div>
                        <div class="pg-card-sub">
                            Heuristic visual estimate indicates <b>{sev_level.lower()}</b> symptomatic surface discoloration.<br>
                            <span style="color:#ffc247;">
                                ⚠️ <b>Methodological Disclaimer:</b> This is an experimental visual estimate derived from color segmentation,
                                <b>not a laboratory-validated agronomic measurement</b>. Do not use for chemical treatment dosage decisions.
                            </span>
                        </div>
                    </div>
                    """)

            # ----------------------------------------------------
            # SECTION D: ADDITIONAL AI MODULES (COMPACT)
            # ----------------------------------------------------
            st.markdown("### 🧩 Additional Diagnostic Modules (Status)")
            sub_col1, sub_col2 = st.columns(2, gap="medium")
            with sub_col1:
                render_html("""
                <div class="pg-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <div class="pg-card-header" style="margin:0;">🐛 Pest Detection Subsystem</div>
                        <span class="pill pill-gray">PENDING TRAINING</span>
                    </div>
                    <div class="pg-card-sub">
                        <b>Planned Target:</b> 12 Key Crop Pests (IP102 Benchmark)<br>
                        <b>Pipeline Status:</b> Training pipeline implemented (<code>training/train_pest_model.py</code>). Inference disabled pending GPU training.<br>
                        <span style="font-size:11px; color:#ffc247;">
                            ⚠️ Untrained predictions are disabled to preserve academic integrity. No random predictions or fake scores.
                        </span>
                    </div>
                </div>
                """)
            with sub_col2:
                render_html("""
                <div class="pg-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <div class="pg-card-header" style="margin:0;">🧪 Nutrient Deficiency Subsystem</div>
                        <span class="pill pill-gray">PENDING TRAINING</span>
                    </div>
                    <div class="pg-card-sub">
                        <b>Planned Target:</b> 10 Nutritional Deficiencies (CoLeaf Benchmark)<br>
                        <b>Pipeline Status:</b> Architecture implemented (<code>training/train_nutrient_model.py</code>). Inference disabled pending training.<br>
                        <span style="font-size:11px; color:#8daec0;">
                            ℹ️ Nutrient analysis is an independent domain; no unsupported predictions are derived from the disease model.
                        </span>
                    </div>
                </div>
                """)

            # ----------------------------------------------------
            # SECTION E: DIAGNOSIS SUMMARY & BIOSECURITY ADVISORY
            # ----------------------------------------------------
            st.markdown("### 📋 Diagnosis Summary & Field Biosecurity Advisory")
            _conf8 = cur.get("confidence", 0.0)
            _unc8  = cur.get("is_uncertain", False)
            _hlth8 = cur.get("is_healthy", False)
            _ent8  = cur.get("entropy", 0.0)
            _mrg8  = cur.get("margin", 0.0)

            certainty_status = "⚠️ Uncertain candidate — see notice above" if _unc8 else ("✅ High confidence prediction" if _conf8 >= 70 else "🟡 Moderate confidence prediction")
            sev_disp8 = vis.get("severity_level", "N/A") if vis.get("valid") else "N/A"
            affected8 = f"{vis.get('affected_area_pct', 0.0):.1f}%" if vis.get("valid") else "N/A"

            render_html(f"""
            <div class="pg-card" style="margin-bottom:16px;">
                <div class="pg-card-header">🔎 Consolidated Diagnostic Summary</div>
                <table style="width:100%; border-collapse:collapse; font-size:13px; color:#e2e8f0;">
                    <tr style="border-bottom:1px solid #0d384d;">
                        <td style="padding:6px 10px; color:#8daec0; width:35%;"><b>Category</b></td>
                        <td style="padding:6px 10px;"><b>Finding</b></td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d; background:rgba(25,230,176,0.04);">
                        <td style="padding:6px 10px; color:#8daec0;">Active Model</td>
                        <td style="padding:6px 10px;">{model_name_disp} (<code>{checkpoint_disp}</code>)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d;">
                        <td style="padding:6px 10px; color:#8daec0;">Predicted Crop</td>
                        <td style="padding:6px 10px; font-weight:700; color:#ffffff;">{cur.get("plant", "N/A")}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d; background:rgba(25,230,176,0.04);">
                        <td style="padding:6px 10px; color:#8daec0;">Predicted Condition</td>
                        <td style="padding:6px 10px; font-weight:700; color:{"#19e6b0" if _hlth8 else "#ff5c6c"};">{"Healthy Foliage" if _hlth8 else cur.get("disease", "N/A")}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d;">
                        <td style="padding:6px 10px; color:#8daec0;">Diagnostic Certainty</td>
                        <td style="padding:6px 10px;">{certainty_status} ({_conf8:.1f}% confidence, entropy {_ent8:.2f}, margin {_mrg8:.1f}%)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d; background:rgba(25,230,176,0.04);">
                        <td style="padding:6px 10px; color:#8daec0;">Heuristic Severity Estimate</td>
                        <td style="padding:6px 10px;">{sev_disp8} ({affected8} estimated affected surface area)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #0d384d;">
                        <td style="padding:6px 10px; color:#8daec0;">Subsystems (Pest / Nutrient)</td>
                        <td style="padding:6px 10px; color:#94a3b8;">Pending training — inference disabled</td>
                    </tr>
                    <tr>
                        <td style="padding:6px 10px; color:#8daec0;">Architecture Disclosure</td>
                        <td style="padding:6px 10px; font-size:11.5px; color:#8daec0;">Host crop and disease category share a single 38-class MobileNetV2 classification head.</td>
                    </tr>
                </table>
                <div style="margin-top:12px; padding:10px 12px; background:rgba(255,194,71,0.08); border-left:3px solid #ffc247; border-radius:6px; font-size:12px; color:#94a3b8; line-height:1.5;">
                    <b style="color:#ffc247;">⚠️ Biosecurity Advisory:</b> This automated prediction is an AI-assisted screening tool only. It is not a substitute for laboratory pathogen confirmation or expert agronomic assessment. Do not apply chemical treatments or make crop management decisions based solely on this output. Consult a certified agricultural extension officer or plant pathologist for confirmed diagnosis.
                </div>
            </div>
            """)

            # ----------------------------------------------------
            # SECTION F: ANALYSIS HISTORY & AUDIT REPORT
            # ----------------------------------------------------
            st.markdown("### 📂 Specimen Analysis History & Audit Report")
            history_scans = st.session_state.get("history_scans", [])
            if history_scans:
                hist_rows = []
                for rec in history_scans[:20]:
                    hist_rows.append({
                        "Timestamp": rec.get("timestamp", "—"),
                        "Crop": rec.get("plant", "—"),
                        "Condition": rec.get("disease", "—"),
                        "Confidence (%)": f"{rec.get('confidence', 0.0):.1f}",
                        "Severity": rec.get("severity_level", "—"),
                        "Uncertain": "Yes" if rec.get("is_uncertain") else "No",
                    })
                df_hist = pd.DataFrame(hist_rows)
                st.dataframe(df_hist, hide_index=True, use_container_width=True)

                json_report_str9 = generate_json_report(cur)
                st.download_button(
                    label="📄 Download JSON Audit Report for Current Analysis",
                    data=json_report_str9,
                    file_name=f"PlantGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=False,
                    key="hist_download_btn"
                )
            else:
                render_html("""
                <div class="pg-card">
                    <div class="pg-card-sub">No analysis records in this session. Run at least one analysis to begin building history.</div>
                </div>
                """)

            st.markdown("<hr style='border-color:#0d384d; margin:20px 0 14px 0;'>", unsafe_allow_html=True)
            st.caption("ℹ️ Longitudinal Progression: This specimen is recorded in your session history. Use the **Disease Progression** tab to compare scans over time.")

# VIEW 2: DISEASE PROGRESSION TRACKING
# ==========================================

elif nav_selection == "📈 Disease Progression":
    render_html("""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0; font-size:28px; color:#ffffff;">📈 Longitudinal Disease Progression Tracking</h1>
        <div style="color:#8daec0; font-size:14px; margin-top:4px;">
            Compare actual historical scans of a plant over time to track symptom expansion, stability, or recovery.
        </div>
    </div>
    """)

    history = st.session_state.history_scans

    if len(history) < 2:
        st.info("ℹ️ At least two real scans are required to compute progression metrics. You can upload two comparison images below:")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            file_past = st.file_uploader("Upload Earlier Scan (Scan A)", type=["jpg", "png", "jpeg"], key="past_upload")
        with col_u2:
            file_now = st.file_uploader("Upload Later Scan (Scan B)", type=["jpg", "png", "jpeg"], key="now_upload")

        if file_past and file_now:
            img_past = ImageOps.exif_transpose(Image.open(file_past)).convert("RGB")
            img_now = ImageOps.exif_transpose(Image.open(file_now)).convert("RGB")

            vis_a = analyze_severity_and_vision(img_past)
            vis_b = analyze_severity_and_vision(img_now)
            pred_a = predict_plant_disease(img_past, mode=st.session_state.selected_model_mode)
            pred_b = predict_plant_disease(img_now, mode=st.session_state.selected_model_mode)

            scan_a = {
                "timestamp": "Scan A (Earlier Upload)",
                "plant": pred_a["plant"],
                "disease": pred_a["disease"],
                "confidence": pred_a["confidence"],
                "affected_area_pct": vis_a.get("affected_area_pct", 0.0),
                "original_image": img_past
            }
            scan_b = {
                "timestamp": "Scan B (Later Upload)",
                "plant": pred_b["plant"],
                "disease": pred_b["disease"],
                "confidence": pred_b["confidence"],
                "affected_area_pct": vis_b.get("affected_area_pct", 0.0),
                "original_image": img_now
            }
            prog_res = compare_scans(scan_a, scan_b)

            st.markdown("### 📊 Progression Analysis Results")
            p1, p2, p3 = st.columns(3)
            p1.image(img_past, caption=f"Scan A ({scan_a['disease']})", use_container_width=True)
            p2.image(img_now, caption=f"Scan B ({scan_b['disease']})", use_container_width=True)
            if prog_res["visual_difference_map"] is not None:
                p3.image(prog_res["visual_difference_map"], caption=f"Pixel Difference Map (Shift: {prog_res['mean_pixel_shift_pct']}%)", use_container_width=True)

            trend_pill = f"pill-{prog_res['trend_color']}"
            render_html(f"""
            <div class="pg-card">
                <div class="pg-card-header">Progression Trajectory: <span class="pill {trend_pill}">{prog_res['trend'].upper()}</span></div>
                <div class="pg-card-sub">
                    <b>Earlier Scan Affected Area:</b> {prog_res['scan_a_severity']}%<br>
                    <b>Later Scan Affected Area:</b> {prog_res['scan_b_severity']}%<br>
                    <b>Delta Severity:</b> {prog_res['delta_severity']:+0.1f}%<br>
                    <b>Delta Confidence:</b> {prog_res['delta_confidence']:+0.1f}%<br>
                    <b>Clinical Interpretation:</b> {prog_res['interpretation']}<br>
                    <span style="font-size:11px; color:#ffc247;">
                    ⚠️ Visual progression estimate based on heuristic color masks.
                    </span>
                </div>
            </div>
            """)

    else:
        options = [f"#{idx+1}: {item['plant']} - {item['disease']} ({item['timestamp']})" for idx, item in enumerate(history)]
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            idx_a = st.selectbox("Select Earlier Baseline Scan (Scan A)", range(len(history)), format_func=lambda i: options[i], index=min(1, len(history)-1))
        with sel_col2:
            idx_b = st.selectbox("Select Recent Follow-Up Scan (Scan B)", range(len(history)), format_func=lambda i: options[i], index=0)

        scan_a = history[idx_a]
        scan_b = history[idx_b]
        prog_res = compare_scans(scan_a, scan_b)

        st.markdown("### 📊 Progression Metrics")
        p1, p2, p3 = st.columns(3)
        if "original_image" in scan_a:
            p1.image(scan_a["original_image"], caption=f"Scan A: {scan_a['timestamp']}", use_container_width=True)
        if "original_image" in scan_b:
            p2.image(scan_b["original_image"], caption=f"Scan B: {scan_b['timestamp']}", use_container_width=True)
        if prog_res["visual_difference_map"] is not None:
            p3.image(prog_res["visual_difference_map"], caption=f"Difference Map (Shift: {prog_res['mean_pixel_shift_pct']}%)", use_container_width=True)

        trend_pill = f"pill-{prog_res['trend_color']}"
        render_html(f"""
        <div class="pg-card">
            <div class="pg-card-header">Progression Trajectory: <span class="pill {trend_pill}">{prog_res['trend'].upper()}</span></div>
            <div class="pg-card-sub">
                <b>Scan A Affected Area:</b> {prog_res['scan_a_severity']}%<br>
                <b>Scan B Affected Area:</b> {prog_res['scan_b_severity']}%<br>
                <b>Delta Severity:</b> {prog_res['delta_severity']:+0.1f}%<br>
                <b>Delta Model Confidence:</b> {prog_res['delta_confidence']:+0.1f}%<br>
                <b>Clinical Interpretation:</b> {prog_res['interpretation']}
            </div>
        </div>
        """)


# ==========================================
# VIEW 3: MODEL EVALUATION & BENCHMARK
# ==========================================

elif nav_selection == "🧪 Model Evaluation":
    render_html("""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0; font-size:28px; color:#ffffff;">🧪 Model Evaluation & Empirical Benchmark Audit</h1>
        <div style="color:#8daec0; font-size:14px; margin-top:4px;">
            Rigorous evaluation metrics, training curves, confusion matrix, and comparative audit for the primary diagnostic model.
        </div>
    </div>
    """)

    eval_report_file = BASE_DIR / "data" / "full_model_evaluation_report.json"
    curves_file = BASE_DIR / "assets" / "full_model_training_curves.png"
    cm_file = BASE_DIR / "assets" / "full_model_confusion_matrix.png"

    if eval_report_file.exists():
        with open(eval_report_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        tsm = eval_data.get("test_split_metrics", {})
        tsum = eval_data.get("training_summary", {})
        splits = eval_data.get("split_counts", {})
        checkpoint_name = tsum.get("model_name", "trained_plantvillage_full_model") + ".keras"
        test_images_count = splits.get("test", 8143)
        eval_timestamp = eval_data.get("evaluation_timestamp", "N/A")
        dataset_name = eval_data.get("dataset_name", "PlantVillage (Official raw/color)")
        total_unique = eval_data.get("total_unique_images", 54284)

        render_html(f"""
        <div class="pg-card pg-card-glow">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="pg-card-header" style="margin:0;">🌿 Primary Disease Classifier Evaluation (<code>{checkpoint_name}</code>)</div>
                <span class="pill pill-green">TEST SPLIT EVALUATED</span>
            </div>
            <div style="font-size:13px; color:#8daec0; margin:6px 0 10px 0;">
                <b>Evaluation Dataset:</b> {dataset_name} ({total_unique:,} total unique specimens across 38 classes)
            </div>
            <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; margin:14px 0;">
                <div class="metric-box">
                    <div class="metric-label">Held-Out Test Accuracy</div>
                    <div class="metric-value">{tsm.get('test_accuracy', 0.0)*100:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Macro Precision</div>
                    <div class="metric-value" style="color:#38bdf8;">{tsm.get('macro_precision', 0.0)*100:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Macro Recall</div>
                    <div class="metric-value" style="color:#10b981;">{tsm.get('macro_recall', 0.0)*100:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Macro F1-Score</div>
                    <div class="metric-value" style="color:#ffc247;">{tsm.get('macro_f1_score', 0.0)*100:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Weighted F1-Score</div>
                    <div class="metric-value" style="color:#19e6b0;">{tsm.get('weighted_f1_score', 0.0)*100:.2f}%</div>
                </div>
            </div>
            <div class="pg-card-sub">
                <b>Checkpoint Filename:</b> <code>{checkpoint_name}</code> (38 classes, MobileNetV2 transfer learning)<br>
                <b>Held-Out Test Images Evaluated:</b> {test_images_count:,} genuine specimens (15% stratified test split)<br>
                <b>Stratified Dataset Splits:</b> Train: {splits.get('train', 0):,} images (70%) • Val: {splits.get('val', 0):,} images (15%) • Test: {splits.get('test', 0):,} images (15%)<br>
                <b>Training Duration:</b> {tsum.get('epochs_completed', 3)} epochs in {tsum.get('training_time_seconds', 0):.1f}s | Final Train Acc: {tsum.get('final_train_acc', 0.0)*100:.2f}% | Final Val Acc: {tsum.get('final_val_acc', 0.0)*100:.2f}%<br>
                <b>Evaluation Timestamp:</b> {eval_timestamp}
            </div>
        </div>
        """)

        ev_c1, ev_c2 = st.columns(2, gap="large")
        with ev_c1:
            if curves_file.exists():
                st.image(str(curves_file), caption=f"Full-Dataset MobileNetV2: Training & Validation Loss & Accuracy Progression ({checkpoint_name})", use_container_width=True)
            else:
                st.info(f"Full model training curves asset not found at assets/{curves_file.name}.")
        with ev_c2:
            if cm_file.exists():
                st.image(str(cm_file), caption=f"Full-Dataset MobileNetV2: Normalized Confusion Matrix ({test_images_count:,} Held-out Test Images)", use_container_width=True)
            else:
                st.info(f"Full model confusion matrix asset not found at assets/{cm_file.name}.")

        # Side-by-side Comparative Benchmark against Baseline on Same 8,143 Test Split
        comp_baseline = eval_data.get("comparison_against_baseline_on_same_test_split", {})
        if comp_baseline:
            with st.expander("📊 Comparative Audit: Full-Dataset vs. Baseline Checkpoint (Same 8,143 Test Images)", expanded=True):
                base_m = comp_baseline.get("baseline_metrics", {})
                diff_acc = (tsm.get('test_accuracy', 0.9526) - base_m.get('test_accuracy', 0.9122)) * 100
                diff_prec = (tsm.get('macro_precision', 0.9388) - base_m.get('macro_precision', 0.8838)) * 100
                diff_rec = (tsm.get('macro_recall', 0.9437) - base_m.get('macro_recall', 0.9107)) * 100
                diff_f1 = (tsm.get('macro_f1_score', 0.9389) - base_m.get('macro_f1_score', 0.8943)) * 100
                render_html(f"""
                <div class="pg-card">
                    <div class="pg-card-header">Controlled Empirical Comparison on Identical 8,143 Held-Out Images</div>
                    <div style="font-size:12.5px; color:#8daec0; margin-bottom:12px;">
                        Both checkpoints were evaluated on the exact same held-out test split (15% stratified) from PlantVillage.
                    </div>
                    <table style="width:100%; border-collapse:collapse; font-size:13px; color:#ffffff;">
                        <tr style="border-bottom:1px solid #0f495e; text-align:left; color:#8daec0;">
                            <th style="padding:8px;">Evaluation Metric</th>
                            <th style="padding:8px;">Baseline (<code>trained_plantvillage_model.keras</code>)</th>
                            <th style="padding:8px; color:#19e6b0;">Full-Dataset (<code>trained_plantvillage_full_model.keras</code>)</th>
                            <th style="padding:8px; color:#38bdf8;">Improvement</th>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(15,73,94,0.4);">
                            <td style="padding:8px;"><b>Training Images Used</b></td>
                            <td style="padding:8px;">4,560 images (120/class subset)</td>
                            <td style="padding:8px; color:#19e6b0;"><b>54,284 images</b> (Complete dataset)</td>
                            <td style="padding:8px; color:#38bdf8;"><b>+49,724 images (+1,090%)</b></td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(15,73,94,0.4);">
                            <td style="padding:8px;"><b>Test Accuracy</b></td>
                            <td style="padding:8px;">{base_m.get('test_accuracy', 0.9122)*100:.2f}%</td>
                            <td style="padding:8px; color:#19e6b0;"><b>{tsm.get('test_accuracy', 0.9526)*100:.2f}%</b></td>
                            <td style="padding:8px; color:#38bdf8;"><b>+{diff_acc:.2f}%</b></td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(15,73,94,0.4);">
                            <td style="padding:8px;"><b>Macro Precision</b></td>
                            <td style="padding:8px;">{base_m.get('macro_precision', 0.8838)*100:.2f}%</td>
                            <td style="padding:8px; color:#19e6b0;"><b>{tsm.get('macro_precision', 0.9388)*100:.2f}%</b></td>
                            <td style="padding:8px; color:#38bdf8;"><b>+{diff_prec:.2f}%</b></td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(15,73,94,0.4);">
                            <td style="padding:8px;"><b>Macro Recall</b></td>
                            <td style="padding:8px;">{base_m.get('macro_recall', 0.9107)*100:.2f}%</td>
                            <td style="padding:8px; color:#19e6b0;"><b>{tsm.get('macro_recall', 0.9437)*100:.2f}%</b></td>
                            <td style="padding:8px; color:#38bdf8;"><b>+{diff_rec:.2f}%</b></td>
                        </tr>
                        <tr>
                            <td style="padding:8px;"><b>Macro F1-Score</b></td>
                            <td style="padding:8px;">{base_m.get('macro_f1_score', 0.8943)*100:.2f}%</td>
                            <td style="padding:8px; color:#19e6b0;"><b>{tsm.get('macro_f1_score', 0.9389)*100:.2f}%</b></td>
                            <td style="padding:8px; color:#38bdf8;"><b>+{diff_f1:.2f}%</b></td>
                        </tr>
                    </table>
                </div>
                """)
    else:
        st.error(f"Evaluation report not found at {eval_report_file}. Please run the evaluation pipeline first.")

    # Historical Baseline Disclosure
    st.markdown("### 📜 Academic Integrity Baseline: Legacy Checkpoint Audit")
    render_html("""
    <div class="pg-card">
        <div class="pg-card-header">Empirical Benchmark of Legacy Checkpoint (plant_model.keras)</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; margin:12px 0;">
            <div class="metric-box">
                <div class="metric-label">Authentic Specimens Tested</div>
                <div class="metric-value" style="color:#38bdf8;">40</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Correct Predictions</div>
                <div class="metric-value" style="color:#ff5c6c;">0</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Empirical Test Accuracy</div>
                <div class="metric-value" style="color:#ff5c6c;">0.0%</div>
            </div>
        </div>
        <div class="pg-card-sub">
            <b>Historical Context:</b> The original codebase displayed fabricated metrics (93.2% accuracy). Our rigorous empirical
            benchmark revealed 0.0% accuracy due to unverified legacy class label permutations, establishing the imperative for
            complete dataset acquisition and end-to-end retrained models.
        </div>
    </div>
    """)


# ==========================================
# VIEW 4: SUPPORTED CLASSES & DATASETS
# ==========================================

elif nav_selection == "📋 Supported Classes":
    render_html("""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0; font-size:28px; color:#ffffff;">📋 Sourced Datasets, Taxonomy & Model Classes</h1>
        <div style="color:#8daec0; font-size:14px; margin-top:4px;">
            Provenance, manifests, and class mappings for all 4 official data sources.
        </div>
    </div>
    """)

    tab_pv, tab_pd, tab_ip, tab_cl = st.tabs([
        "🌿 PlantVillage (38 Classes)",
        "🌾 PlantDoc Field (29 Classes)",
        "🐛 IP102 Pest (102 Classes)",
        "🧪 CoLeaf Nutrient (10 Classes)"
    ])

    with tab_pv:
        pv_inv_file = BASE_DIR / "data" / "datasets" / "plantvillage" / "inventory_report.json"
        if pv_inv_file.exists():
            with open(pv_inv_file, "r", encoding="utf-8") as f:
                inv_data = json.load(f)
            render_html(f"""
            <div class="pg-card">
                <div class="pg-card-header">PlantVillage Dataset Inventory</div>
                <div class="pg-card-sub">
                    <b>Total Verified Images:</b> {inv_data.get('total_valid_images', 0):,} RGB color images<br>
                    <b>Total Disease & Crop Classes:</b> {inv_data.get('total_classes', 38)} classes<br>
                    <b>Corrupted Files Removed:</b> {inv_data.get('total_corrupted_removed', 0)}<br>
                    <b>Cataloging Timestamp:</b> {inv_data.get('timestamp', 'N/A')}
                </div>
            </div>
            """)

            counts = inv_data.get("class_counts", {})
            df_pv = pd.DataFrame([
                {"Class Name": k.replace("___", " - ").replace("_", " "), "Raw Label": k, "Images": v}
                for k, v in counts.items()
            ])
            st.dataframe(df_pv, hide_index=True, use_container_width=True)
        else:
            m = load_dataset_manifest("plantvillage")
            render_html(f"""
            <div class="pg-card">
                <div class="pg-card-header">{m.get('dataset_name', 'PlantVillage')}</div>
                <div class="pg-card-sub">
                    <b>Official Source:</b> <a href="{m.get('official_source')}" target="_blank">{m.get('official_source')}</a><br>
                    <b>Total Classes:</b> {m.get('total_classes')} across {m.get('unique_crops_count')} crops<br>
                    <b>Total Images in Full Dataset:</b> {m.get('total_images_in_full_dataset'):,} lab images
                </div>
            </div>
            """)
            classes_pv = load_dataset_classes("plantvillage")
            if classes_pv:
                st.dataframe(pd.DataFrame(classes_pv), hide_index=True, use_container_width=True)

    with tab_pd:
        m = load_dataset_manifest("plantdoc")
        render_html(f"""
        <div class="pg-card">
            <div class="pg-card-header">{m.get('dataset_name', 'PlantDoc')}</div>
            <div class="pg-card-sub">
                <b>Official Source:</b> <a href="{m.get('official_source')}" target="_blank">{m.get('official_source')}</a><br>
                <b>Publication:</b> {m.get('publication')}<br>
                <b>Field Setting:</b> {m.get('field_setting')}<br>
                <b>Total Classes:</b> {m.get('total_classes')} classes
            </div>
        </div>
        """)
        classes_pd = load_dataset_classes("plantdoc")
        if classes_pd:
            st.dataframe(pd.DataFrame(classes_pd), hide_index=True, use_container_width=True)

    with tab_ip:
        m = load_dataset_manifest("ip102_pest")
        render_html(f"""
        <div class="pg-card">
            <div class="pg-card-header">{m.get('dataset_name', 'IP102 Pest Dataset')}</div>
            <div class="pg-card-sub">
                <b>Official Source:</b> <a href="{m.get('official_source')}" target="_blank">{m.get('official_source')}</a><br>
                <b>Total Pest Classes:</b> {m.get('total_classes')} classes (75,222 images in full dataset)<br>
                <b>Agricultural Target Mapping:</b> 12 key crop pests mapped (<code>target12_mapping.json</code>)
            </div>
        </div>
        """)
        target_p = BASE_DIR / "data" / "datasets" / "ip102_pest" / "target12_mapping.json"
        if target_p.exists():
            with open(target_p, "r", encoding="utf-8") as f:
                df_t12 = pd.DataFrame(json.load(f))
            st.markdown("#### Planned 12 Agricultural Pest Subset Mapped to IP102")
            st.dataframe(df_t12, hide_index=True, use_container_width=True)

    with tab_cl:
        m = load_dataset_manifest("coleaf_nutrient")
        render_html(f"""
        <div class="pg-card">
            <div class="pg-card-header">{m.get('dataset_name', 'CoLeaf-DB')}</div>
            <div class="pg-card-sub">
                <b>Official Source:</b> <a href="{m.get('official_source')}" target="_blank">{m.get('official_source')}</a><br>
                <b>Plant Species:</b> {m.get('plant_species')}<br>
                <b>Total Classes:</b> {m.get('total_classes')} (1,006 images in full dataset)
            </div>
        </div>
        """)
        classes_cl = load_dataset_classes("coleaf_nutrient")
        if classes_cl:
            st.dataframe(pd.DataFrame(classes_cl), hide_index=True, use_container_width=True)


# ==========================================
# VIEW 5: RECENT HISTORY
# ==========================================

elif nav_selection == "🕒 Recent History":
    render_html("""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0; font-size:28px; color:#ffffff;">🕒 Genuine Analysis History</h1>
        <div style="color:#8daec0; font-size:14px; margin-top:4px;">
            Auditable log of diagnostic scans performed during the current session and persisted to CSV.
        </div>
    </div>
    """)

    csv_records = load_history_records()
    if csv_records:
        df_hist = pd.DataFrame(csv_records)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History as CSV",
            csv_data,
            "PlantGuard_History.csv",
            "text/csv",
            use_container_width=False
        )
    else:
        st.info("No analysis history recorded yet. Run leaf analyses in the Upload Image or Live Camera views to populate records.")


# ==========================================
# VIEW 6: ABOUT PROJECT & PROVENANCE
# ==========================================

elif nav_selection == "ℹ️ About Project":
    render_html("""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0; font-size:28px; color:#ffffff;">ℹ️ Technical Specifications & Academic Provenance</h1>
        <div style="color:#8daec0; font-size:14px; margin-top:4px;">
            System architecture audit, bug fixes, Grad-CAM interpretability, and academic integrity disclosures.
        </div>
    </div>
    """)

    render_html("""
    <div class="pg-card">
        <div class="pg-card-header">1. Double-Normalization Bug Resolution</div>
        <div class="pg-card-sub">
            The MobileNetV2 architecture contains an embedded <code>Rescaling(scale=1/127.5, offset=-1)</code> layer.
            This mathematically scales inputs from <code>[0.0, 255.0]</code> to <code>[-1.0, 1.0]</code>.<br>
            <b>Bug Resolved:</b> The previous implementation divided pixels by 255 before feeding the network,
            resulting in double-normalization into <code>[-1.0, -0.992]</code> and obliterating feature representation.
            Our pipeline preserves raw <code>[0.0, 255.0] float32</code> RGB arrays.
        </div>
    </div>

    <div class="pg-card">
        <div class="pg-card-header">2. Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM)</div>
        <div class="pg-card-sub">
            Grad-CAM computes the gradient of the predicted class score with respect to feature maps of the final convolutional layer
            (<code>mobilenetv2 / out_relu</code>).<br>
            <b>Scientific Scope:</b> It visualizes where the neural network's convolutional filters are responding, but does
            <b>not</b> prove correct disease localization or ground-truth biological pathology.
        </div>
    </div>

    <div class="pg-card">
        <div class="pg-card-header">3. Computer Vision & Severity Estimation</div>
        <div class="pg-card-sub">
            • <b>Foliage Segmentation:</b> Leaf area is isolated from background using adaptive color thresholds in LAB and HSV spaces.<br>
            • <b>Quality Gate:</b> Laplacian variance detects out-of-focus imagery, while mean brightness identifies extreme illumination.<br>
            • <b>Severity Formula:</b><br>
            <code>Severity (%) = 100 × (Symptom Pixels in Leaf Mask / Total Segmented Leaf Pixels)</code><br>
            • <b>Transparency:</b> HSV symptom thresholding is explicitly presented as an experimental visual estimate, not a laboratory-validated measurement.
        </div>
    </div>
    """)


# ==========================================
# FOOTER
# ==========================================

st.markdown("<hr style='border-color:#0d384d; margin:28px 0 12px 0;'>", unsafe_allow_html=True)
st.caption(
    "🌿 **PlantGuard AI** | Senior Computer Vision & Deep Learning Engineering System | "
    "Datasets: PlantVillage • PlantDoc • IP102 • CoLeaf"
)
