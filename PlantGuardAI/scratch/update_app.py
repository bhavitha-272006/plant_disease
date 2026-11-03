from pathlib import Path

app_path = Path("PlantGuardAI/app.py")
text = app_path.read_text(encoding="utf-8")
text_lf = text.replace("\r\n", "\n")

# 1. Update navigation options and current index
old_nav_block = """    nav_options = [
        "🏠 Home",
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
    current_nav_index = 0
    if st.session_state.workflow_stage == "dashboard" and st.session_state.current_analysis is not None:
        current_nav_index = 3

    nav_selection = st.radio(
        "Navigation Menu",
        nav_options,
        index=current_nav_index,
        label_visibility="collapsed"
    )

    # Sync navigation with workflow stage
    if nav_selection in ["🏠 Home", "📷 Live Camera", "⬆️ Upload Image"]:
        if st.session_state.workflow_stage != "input":
            st.session_state.workflow_stage = "input"
    elif nav_selection == "📊 Analysis Dashboard":
        if st.session_state.current_analysis is not None:
            st.session_state.workflow_stage = "dashboard\""""

new_nav_block = """    nav_options = [
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
            st.session_state.workflow_stage = "dashboard\""""

assert old_nav_block in text_lf, "old_nav_block not found"
text_lf = text_lf.replace(old_nav_block, new_nav_block, 1)

# 2. Stage 1 nav_selection check
old_stage1_check = 'if nav_selection in ["🏠 Home", "📷 Live Camera", "⬆️ Upload Image"] and st.session_state.workflow_stage == "input":'
new_stage1_check = 'if nav_selection in ["📷 Live Camera", "⬆️ Upload Image"] and st.session_state.workflow_stage == "input":'
assert old_stage1_check in text_lf, "old_stage1_check not found"
text_lf = text_lf.replace(old_stage1_check, new_stage1_check, 1)

# 3. Predict button execution block
old_pred_exec = """                            # Execute Core AI Inference with specimen validation
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
                                st.rerun()"""

new_pred_exec = """                            # Execute Core AI Inference with specimen validation
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
                                st.rerun()"""

assert old_pred_exec in text_lf, "old_pred_exec not found"
text_lf = text_lf.replace(old_pred_exec, new_pred_exec, 1)

# 4. Return to home strings
old_home_btn = """        st.info("ℹ️ No leaf specimen currently analyzed. Please return to the Home page to capture or upload an image.")
        if st.button("⬅️ Return to Home Page"):"""
new_home_btn = """        st.info("ℹ️ No leaf specimen currently analyzed. Please return to Image Input to capture or upload an image.")
        if st.button("⬅️ Return to Image Input"):"""
assert old_home_btn in text_lf, "old_home_btn not found"
text_lf = text_lf.replace(old_home_btn, new_home_btn, 1)

old_hist_notice = 'Run leaf analyses in the Home or Camera views to populate records.'
new_hist_notice = 'Run leaf analyses in the Upload Image or Live Camera views to populate records.'
assert old_hist_notice in text_lf, "old_hist_notice not found"
text_lf = text_lf.replace(old_hist_notice, new_hist_notice, 1)

# 5. Left column Grad-CAM tabs handling when blocked
old_tabs = """            tab_orig, tab_cam, tab_overlay = st.tabs([
                "🌿 Original Leaf",
                "🔥 Grad-CAM Heatmap",
                "🎯 Superimposed Focus"
            ])
            with tab_orig:
                st.image(active_img, caption="Original Input Specimen", use_container_width=True)
            with tab_cam:
                st.image(cur["gradcam_raw"], caption="Convolutional Activation Heatmap (out_relu)", use_container_width=True)
            with tab_overlay:
                st.image(cur["gradcam_image"], caption="Grad-CAM Activation Superimposed on Specimen", use_container_width=True)"""

new_tabs = """            if cur.get("gradcam_raw") is not None and cur.get("gradcam_image") is not None:
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
                st.info("ℹ️ Grad-CAM neural attention heatmap is disabled because diagnosis was withheld (Domain Shift / OOD safeguard).")"""

assert old_tabs in text_lf, "old_tabs not found"
text_lf = text_lf.replace(old_tabs, new_tabs, 1)

# 6. Right column Diagnosis card when blocked vs normal
old_right_col = """        # ----------------------------------------------------
        # RIGHT COLUMN: DIAGNOSIS RESULTS
        # ----------------------------------------------------
        with col_diagnosis:
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
                uncertainty_html = f\"\"\"
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
                \"\"\"

            render_html(f\"\"\"
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
            \"\"\")

            with st.expander("📊 View Top-5 Class Probability Distribution", expanded=False):
                top_5_data = cur["top_5"]
                df_top5 = pd.DataFrame([
                    {"Class": item["display_name"], "Confidence (%)": round(item["percentage"], 1)}
                    for item in top_5_data
                ])
                st.dataframe(df_top5, hide_index=True, use_container_width=True)"""

new_right_col = """        # ----------------------------------------------------
        # RIGHT COLUMN: DIAGNOSIS RESULTS
        # ----------------------------------------------------
        with col_diagnosis:
            _is_blocked = cur.get("is_blocked", False)
            if _is_blocked:
                proto_sim = cur.get("prototype_similarity", 0.0)
                reasons = cur.get("uncertainty_reasons", [])
                reasons_li = "".join(f"<li style='margin-bottom:3px;'>{r}</li>" for r in reasons)
                render_html(f\"\"\"
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
                \"\"\")
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
                    uncertainty_html = f\"\"\"
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
                    \"\"\"

                proto_sim_val = cur.get('prototype_similarity')
                proto_metric_html = f"<div class='metric-box'><div class='metric-label'>Prototype Cosine Sim</div><div class='metric-value' style='color:#10b981;'>{proto_sim_val:.2f}</div></div>" if proto_sim_val is not None else ""

                render_html(f\"\"\"
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
                \"\"\")

                with st.expander("📊 View Top-5 Class Probability Distribution", expanded=False):
                    top_5_data = cur["top_5"]
                    df_top5 = pd.DataFrame([
                        {"Class": item["display_name"], "Confidence (%)": round(item["percentage"], 1)}
                        for item in top_5_data
                    ])
                    st.dataframe(df_top5, hide_index=True, use_container_width=True)"""

assert old_right_col in text_lf, "old_right_col not found"
text_lf = text_lf.replace(old_right_col, new_right_col, 1)

# 7. Check if _is_blocked for supporting analysis sections A, B, C, D
old_support_header = """        # ====================================================
        # BELOW THE MAIN DASHBOARD: SUPPORTING ANALYSIS
        # ====================================================
        st.markdown("<hr style='border-color:#0d384d; margin:24px 0 20px 0;'>", unsafe_allow_html=True)"""

start_idx = text_lf.find(old_support_header)
end_marker = "# VIEW 2: DISEASE PROGRESSION TRACKING"
end_idx = text_lf.find(end_marker)

assert start_idx != -1, "start_idx not found"
assert end_idx != -1, "end_idx not found"

support_section = text_lf[start_idx + len(old_support_header):end_idx]

# Indent all lines of support_section by 4 spaces
indented_support = "\n".join("    " + line if line.strip() else line for line in support_section.split("\n"))

new_support_block = f"""{old_support_header}

        if _is_blocked:
            st.info("ℹ️ Secondary diagnostic modules (Grad-CAM neural attention heatmaps, heuristic damage contours, severity scoring, and agronomic treatment protocols) are withheld for this specimen to prevent unverified clinical guidance on out-of-distribution inputs.")
            st.markdown("<hr style='border-color:#0d384d; margin:20px 0 14px 0;'>", unsafe_allow_html=True)
            st.caption("ℹ️ Longitudinal Progression: Specimen diagnosis was withheld due to domain shift or non-leaf features. Not recorded in persistent diagnostic history.")
        else:
{indented_support}"""

text_lf = text_lf[:start_idx] + new_support_block + text_lf[end_idx:]

app_path.write_text(text_lf.replace("\n", "\r\n"), encoding="utf-8", errors="replace")
print("Successfully updated app.py with all fixes!")
