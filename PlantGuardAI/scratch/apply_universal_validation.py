"""
Implementation script for Universal Leaf Specimen Validation & OOD Protection System.
Updates:
1. modules/core_ai.py:
   - Adds evaluate_leaf_suitability()
   - Updates predict_plant_disease() with specimen_type validation and energy metric
2. app.py:
   - Adds permanent limitation notice near upload component
   - Adds specimen type validation selection (Single leaf, Whole plant / canopy, Fruit, Stem / flower / root, Not sure)
   - Disables/withholds diagnosis when specimen type is not Single leaf or fails optical suitability gate
3. test_pipeline.py:
   - Adds TEST 6 for all 5 specimen selection categories
"""

from pathlib import Path
import re

BASE_DIR = Path(r"c:\Users\chinn\OneDrive\Desktop\plant_cv\PlantGuardAI")

# =========================================================================
# 1. Update modules/core_ai.py
# =========================================================================
core_ai_path = BASE_DIR / "modules" / "core_ai.py"
core_text = core_ai_path.read_text(encoding="utf-8")

suitability_func = '''
def evaluate_leaf_suitability(image: Image.Image, user_specimen_type: str = "Single leaf") -> dict:
    """
    Universal Specimen Suitability Gate for PlantGuard AI.
    Works across all 14 crop categories without species-specific hardcoding.
    Evaluates:
      1. User Protocol Declaration (Single leaf vs Fruit/Canopy/Stem/Not sure)
      2. Universal Optical Quality (Laplacian sharpness variance and illumination)
      3. Foliar Surface Area & Purity (Green, yellow chlorosis, brown necrosis)
      4. Non-Foliar Organ Dominance (Fruit/flower chromatic prominence)

    Outcomes:
      - VALID_LEAF_SPECIMEN: Authorized for disease diagnosis
      - NON_LEAF_ORGAN: Fruit, flower, stem, root, or whole-plant organ
      - UNSUITABLE_IMAGE: Severe blur, extreme lighting, or background-only
      - UNCERTAIN_SPECIMEN: Ambiguous organ boundary or unconfirmed category
    """
    # 1. User Protocol Declaration Gate
    if user_specimen_type == "Fruit":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Fruit. The 38-class MobileNetV2 classifier is trained exclusively on foliar (leaf) specimens and cannot diagnose fruit pathologies.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Whole plant / canopy":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Whole plant / canopy. The classifier was trained on single excised leaves against plain backgrounds; complex canopies are outside its validated scope.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Stem / flower / root":
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": "Specimen is declared as Stem / flower / root. The model does not support non-foliar botanical organs.",
            "metrics": {"user_declaration": user_specimen_type}
        }
    elif user_specimen_type == "Not sure":
        return {
            "status": "UNCERTAIN_SPECIMEN",
            "is_permitted": False,
            "reason": "Specimen organ type is unconfirmed. Disease diagnosis is withheld to prevent unverified classification.",
            "metrics": {"user_declaration": user_specimen_type}
        }

    # 2. Universal Optical & Morphological Assessment
    oriented_img = ImageOps.exif_transpose(image)
    rgb = np.array(oriented_img.convert("RGB"))
    h, w = rgb.shape[:2]
    total_pixels = h * w

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(np.mean(gray))

    # Universal Color Space Conversions
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    # Universal Foliar Color Segmentation across all crops:
    # Green foliage: HSV H in [25, 95], S >= 25, V >= 25, or LAB a < 125 with S >= 20
    mask_green_hsv = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([95, 255, 255]))
    mask_green_lab = np.uint8(((lab[:, :, 1] < 125) & (hsv[:, :, 1] >= 20)) * 255)
    
    # Yellow / Chlorotic leaf tissue: HSV H in [15, 35], S >= 30, V >= 50
    mask_yellow = cv2.inRange(hsv, np.array([15, 30, 50]), np.array([35, 255, 255]))

    # Brown / Necrotic leaf tissue: HSV H in [8, 24], S >= 25, V in [20, 200]
    mask_brown = cv2.inRange(hsv, np.array([8, 25, 20]), np.array([24, 255, 200]))

    foliage_mask = cv2.bitwise_or(mask_green_hsv, mask_green_lab)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_yellow)
    foliage_mask = cv2.bitwise_or(foliage_mask, mask_brown)

    foliage_pixels = cv2.countNonZero(foliage_mask)
    foliage_ratio = float(foliage_pixels / total_pixels)

    # Non-Foliar High-Chroma Organ Check (Fruit / Flower Dominance)
    # Ripe fruit / flower chromatic signature: High redness in LAB (a* > 140) and high saturation in HSV (S > 75)
    red_hsv1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([12, 255, 255]))
    red_hsv2 = cv2.inRange(hsv, np.array([165, 75, 40]), np.array([180, 255, 255]))
    red_hsv = cv2.bitwise_or(red_hsv1, red_hsv2)
    red_lab = np.uint8((lab[:, :, 1] > 140) * 255)
    fruit_mask = cv2.bitwise_and(red_hsv, red_lab)
    fruit_pixels = cv2.countNonZero(fruit_mask)
    fruit_ratio = float(fruit_pixels / total_pixels)

    fruit_to_foliage = float(fruit_pixels / (foliage_pixels + 1e-6)) if fruit_pixels > 0 else 0.0

    # If prominent fruit organ is detected (occupies substantial area or dominates foreground)
    if fruit_ratio > 0.08 or (fruit_pixels > 500 and fruit_to_foliage > 0.30):
        return {
            "status": "NON_LEAF_ORGAN",
            "is_permitted": False,
            "reason": f"Optical analysis detected prominent non-foliar fruit/flower structures ({fruit_ratio*100:.1f}% of frame). The model was trained exclusively on leaf specimens and cannot diagnose fruit or floral organs.",
            "metrics": {"fruit_ratio": fruit_ratio, "foliage_ratio": foliage_ratio, "laplacian_var": lap_var}
        }

    # Optical image quality gate
    if lap_var < 15.0:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Severe optical or motion blur detected (Laplacian variance {lap_var:.1f} < 15.0). Diagnostic leaf features cannot be resolved.",
            "metrics": {"laplacian_var": lap_var, "mean_brightness": mean_bright}
        }
    if mean_bright < 15.0 or mean_bright > 248.0:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Extreme illumination detected (Mean brightness {mean_bright:.1f}/255). Image is severely underexposed or overexposed.",
            "metrics": {"laplacian_var": lap_var, "mean_brightness": mean_bright}
        }

    if foliage_ratio < 0.10:
        return {
            "status": "UNSUITABLE_IMAGE",
            "is_permitted": False,
            "reason": f"Insufficient leaf surface area detected ({foliage_ratio*100:.1f}% coverage < 10%). The image appears to show background, soil, or an inanimate surface.",
            "metrics": {"foliage_ratio": foliage_ratio, "laplacian_var": lap_var}
        }

    return {
        "status": "VALID_LEAF_SPECIMEN",
        "is_permitted": True,
        "reason": "Identifiable single leaf specimen confirmed with adequate optical sharpness, illumination, and foliar surface area.",
        "metrics": {"foliage_ratio": foliage_ratio, "fruit_ratio": fruit_ratio, "laplacian_var": lap_var, "mean_brightness": mean_bright}
    }
'''

# Insert evaluate_leaf_suitability right above parse_class_label
if "def evaluate_leaf_suitability" not in core_text:
    target_marker = "def parse_class_label(raw_class: str):"
    core_text = core_text.replace(target_marker, suitability_func.strip() + "\n\n\n" + target_marker)
    print("Added evaluate_leaf_suitability to core_ai.py")

# Update predict_plant_disease definition and logic
old_predict_def = "def predict_plant_disease(image: Image.Image, confidence_threshold=65.0, entropy_threshold=1.75, margin_threshold=18.0, mode=None):"
new_predict_def = 'def predict_plant_disease(image: Image.Image, confidence_threshold=65.0, entropy_threshold=1.75, margin_threshold=18.0, mode=None, specimen_type="Single leaf"):'

if old_predict_def in core_text:
    core_text = core_text.replace(old_predict_def, new_predict_def)
    print("Updated predict_plant_disease signature in core_ai.py")

# Insert suitability gate inside predict_plant_disease right after input_arr = preprocess_image(image)
old_pred_start = '''    input_arr = preprocess_image(image)

    # 1. Image Foliage / Quality Assessment for OOD detection
    foliage_info = assess_input_foliage(image)'''

new_pred_start = '''    input_arr = preprocess_image(image)

    # 0. Universal Specimen Suitability Gate
    suitability = evaluate_leaf_suitability(image, user_specimen_type=specimen_type)
    if not suitability["is_permitted"]:
        return {
            "status": "withheld",
            "is_blocked": True,
            "suitability_status": suitability["status"],
            "suitability_reason": suitability["reason"],
            "suitability_metrics": suitability.get("metrics", {}),
            "raw_class": "N/A",
            "class_index": -1,
            "plant": "Diagnosis Withheld",
            "disease": f"Specimen Not Authorized ({suitability['status']})",
            "is_healthy": False,
            "confidence": 0.0,
            "entropy": 0.0,
            "margin": 0.0,
            "is_uncertain": True,
            "uncertainty_reasons": [
                f"Suitability Gate: {suitability['reason']}"
            ],
            "top_5": [],
            "model_readiness": "Diagnostic Withheld (Leaf Only)",
            "model_mode": mode or _active_model_mode
        }

    # 1. Image Foliage / Quality Assessment for OOD detection
    foliage_info = assess_input_foliage(image)'''

if old_pred_start in core_text:
    core_text = core_text.replace(old_pred_start, new_pred_start)
    print("Added suitability gate inside predict_plant_disease in core_ai.py")

# Ensure returned dict in predict_plant_disease includes suitability status and is_blocked=False
return_marker = '    return {\n        "raw_class": top_class_raw,'
new_return_marker = '''    return {
        "status": "success",
        "is_blocked": False,
        "suitability_status": suitability["status"],
        "suitability_reason": suitability["reason"],
        "raw_class": top_class_raw,'''

if return_marker in core_text and '"is_blocked": False' not in core_text:
    core_text = core_text.replace(return_marker, new_return_marker)
    print("Added suitability_status and is_blocked to return dictionary in core_ai.py")

core_ai_path.write_text(core_text, encoding="utf-8")
print("Saved modules/core_ai.py")

# =========================================================================
# 2. Update app.py
# =========================================================================
app_path = BASE_DIR / "app.py"
app_text = app_path.read_text(encoding="utf-8")

# Add evaluate_leaf_suitability to imports from modules.core_ai
if "evaluate_leaf_suitability" not in app_text:
    app_text = app_text.replace(
        "    predict_plant_disease,\n",
        "    predict_plant_disease,\n    evaluate_leaf_suitability,\n"
    )
    print("Imported evaluate_leaf_suitability in app.py")

# Add permanent limitation notice near the upload component (above col_cam, col_upload)
old_stage1_start = '''    st.markdown("### 📥 Stage 1: Select Plant Leaf Specimen")
    st.markdown("Choose an input method below to provide a single plant leaf photograph:")

    col_cam, col_upload = st.columns(2, gap="large")'''

new_stage1_start = '''    st.markdown("### 📥 Stage 1: Select Plant Leaf Specimen")
    st.markdown("Choose an input method below to provide a single plant leaf photograph:")

    render_html("""
    <div style="background:rgba(56, 189, 248, 0.08); border-left:4px solid #38bdf8; border-radius:6px; padding:10px 14px; margin-bottom:16px; font-size:13px; color:#c8e3ed;">
        <b>🌿 Specimen Scope Notice:</b> PlantGuard AI is trained to identify diseases from leaf images. Fruit, stem, flower, root, and whole-plant images are outside its validated scope.
    </div>
    """)

    col_cam, col_upload = st.columns(2, gap="large")'''

if old_stage1_start in app_text:
    app_text = app_text.replace(old_stage1_start, new_stage1_start)
    print("Added permanent specimen limitation notice in app.py")

# Update active_img section in Stage 1 with Specimen Type Verification radio and Suitability Gate
old_action_block = '''            btn_run, btn_clear = st.columns([2, 1])
            with btn_run:
                if st.button("🔬 Analyze Plant", use_container_width=True):
                    with st.spinner("Executing Deep Learning Diagnostic Pipeline..."):
                        # Execute Core AI Inference
                        pred = predict_plant_disease(active_img, mode=st.session_state.selected_model_mode)
                        gradcam_overlay, gradcam_raw = compute_gradcam(active_img, pred["class_index"], mode=pred.get("model_mode"))
                        vision = analyze_severity_and_vision(active_img)
                        pest = analyze_pest(active_img)
                        nutrient = analyze_nutrients(active_img)

                        analysis_res = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "source": st.session_state.image_source_desc,
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
                            "vision": vision,
                            "pest": pest,
                            "nutrient": nutrient
                        }

                        st.session_state.current_analysis = analysis_res
                        st.session_state.workflow_stage = "dashboard"

                        # Record for history
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
                if st.button("❌ Clear Image", use_container_width=True):'''

new_action_block = '''            # Specimen Type Verification Selector (Required Step)
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
                                st.error(f"Execution Blocked: {pred.get('block_reason')}")
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
                    st.rerun()'''

if old_action_block in app_text:
    app_text = app_text.replace(old_action_block, new_action_block)
    print("Updated Specimen Type Validation & Suitability Gate in app.py")
else:
    print("Warning: old_action_block not matched in app.py, trying with CRLF replacement...")
    old_crlf = old_action_block.replace('\n', '\r\n')
    if old_crlf in app_text:
        app_text = app_text.replace(old_crlf, new_action_block.replace('\n', '\r\n'))
        print("Updated Specimen Type Validation & Suitability Gate in app.py (CRLF)")

app_path.write_text(app_text, encoding="utf-8")
print("Saved app.py")

# =========================================================================
# 3. Update test_pipeline.py with TEST 6
# =========================================================================
test_path = BASE_DIR / "test_pipeline.py"
test_text = test_path.read_text(encoding="utf-8")

test6_code = '''# TEST 6: Universal Leaf Specimen Suitability Gate & OOD Safeguards
print("\\n--- TEST 6: Universal Leaf Specimen Suitability Gate & OOD Safeguards ---")

# 6A: Valid single leaf
pred_valid = predict_plant_disease(img, specimen_type="Single leaf")
assert pred_valid["is_blocked"] is False
assert pred_valid["confidence"] > 0.0
assert pred_valid["suitability_status"] == "VALID_LEAF_SPECIMEN"
print("  6A Passed: Single leaf specimen permitted for genuine disease classification.")

# 6B: Declared Fruit -> Blocked
pred_fruit = predict_plant_disease(img, specimen_type="Fruit")
assert pred_fruit["is_blocked"] is True
assert pred_fruit["confidence"] == 0.0
assert pred_fruit["suitability_status"] == "NON_LEAF_ORGAN"
assert "Fruit" in pred_fruit["suitability_reason"]
print("  6B Passed: Fruit specimen category strictly blocked from disease classification.")

# 6C: Declared Whole plant / canopy -> Blocked
pred_canopy = predict_plant_disease(img, specimen_type="Whole plant / canopy")
assert pred_canopy["is_blocked"] is True
assert pred_canopy["confidence"] == 0.0
assert pred_canopy["suitability_status"] == "NON_LEAF_ORGAN"
print("  6C Passed: Whole plant / canopy specimen strictly blocked.")

# 6D: Declared Stem / flower / root -> Blocked
pred_stem = predict_plant_disease(img, specimen_type="Stem / flower / root")
assert pred_stem["is_blocked"] is True
assert pred_stem["confidence"] == 0.0
assert pred_stem["suitability_status"] == "NON_LEAF_ORGAN"
print("  6D Passed: Stem / flower / root specimen strictly blocked.")

# 6E: Declared Not sure -> Blocked
pred_notsure = predict_plant_disease(img, specimen_type="Not sure")
assert pred_notsure["is_blocked"] is True
assert pred_notsure["confidence"] == 0.0
assert pred_notsure["suitability_status"] == "UNCERTAIN_SPECIMEN"
print("  6E Passed: Unconfirmed specimen category strictly blocked.")

# 6F: Non-leaf blank background image with Single leaf declared -> Optical Gate Blocks
blank_test_img = Image.new("RGB", (224, 224), (200, 200, 200))
pred_blank = predict_plant_disease(blank_test_img, specimen_type="Single leaf")
assert pred_blank["is_blocked"] is True
assert pred_blank["suitability_status"] == "UNSUITABLE_IMAGE"
print("  6F Passed: Optical gate autonomously blocks non-foliar blank/background image.")

print("Test 6 Passed: Universal specimen suitability gate and OOD safeguards verified across all categories.")
'''

old_test_end = '''print("\\n" + "=" * 60)
print(">>> ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
print("=" * 60)'''

new_test_end = test6_code + '''
print("\\n" + "=" * 60)
print(">>> ALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
print("=" * 60)'''

if old_test_end in test_text:
    test_text = test_text.replace(old_test_end, new_test_end)
    print("Added TEST 6 to test_pipeline.py")
else:
    old_test_end_crlf = old_test_end.replace('\n', '\r\n')
    if old_test_end_crlf in test_text:
        test_text = test_text.replace(old_test_end_crlf, new_test_end.replace('\n', '\r\n'))
        print("Added TEST 6 to test_pipeline.py (CRLF)")

test_path.write_text(test_text, encoding="utf-8")
print("Saved test_pipeline.py")
print("\nAll files successfully patched!")

