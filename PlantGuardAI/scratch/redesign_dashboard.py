"""
Redesign PlantGuard AI Analysis Dashboard in app.py.
Applies:
1. Updated render_html function to safely render pure HTML without markdown code block parsing.
2. Complete Stage 2 Dashboard redesign according to reference layout:
   - Balanced two-column main dashboard:
     * Left Column: Specimen Analysis (Uploaded image, metadata, dimensions, quality, 3 views)
     * Right Column: Diagnosis Results (Single clean card with crop, disease, model, confidence, uncertainty, top-5)
   - Supporting Analysis Sections below main dashboard:
     * Visual Explanation (Grad-CAM technical scope)
     * Heuristic Damage Analysis & Experimental Severity Estimate
     * Compact Additional AI Modules (Pest & Nutrient - pending training)
     * Consolidated Diagnosis Summary & Biosecurity Advisory
     * Analysis History & JSON Audit Report Download
"""

from pathlib import Path

app_path = Path(r"c:\Users\chinn\OneDrive\Desktop\plant_cv\PlantGuardAI\app.py")
content = app_path.read_text(encoding="utf-8")

# 1. Update render_html
old_render_html = '''def render_html(snippet: str):
    """Safely renders HTML in Streamlit without triggering Markdown indented code blocks."""
    st.markdown(textwrap.dedent(snippet).strip(), unsafe_allow_html=True)'''

new_render_html = '''def render_html(snippet: str):
    """Safely renders HTML in Streamlit natively without markdown code block parsing."""
    cleaned = "\\n".join(line.strip() for line in snippet.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)'''

if old_render_html in content:
    content = content.replace(old_render_html, new_render_html, 1)
    print("Updated render_html definition.")
else:
    print("Warning: old render_html definition not matched exactly, checking alternative...")
    # Try with CRLF
    old_crlf = old_render_html.replace('\n', '\r\n')
    if old_crlf in content:
        content = content.replace(old_crlf, new_render_html.replace('\n', '\r\n'), 1)
        print("Updated render_html definition (CRLF).")

# 2. Locate Stage 2 start and end
stage2_start_marker = '# STAGE 2: PROFESSIONAL ANALYSIS DASHBOARD\n# =========================================='
if stage2_start_marker not in content:
    stage2_start_marker = stage2_start_marker.replace('\n', '\r\n')

view2_marker = '# VIEW 2: DISEASE PROGRESSION TRACKING\n# =========================================='
if view2_marker not in content:
    view2_marker = view2_marker.replace('\n', '\r\n')

start_pos = content.find(stage2_start_marker)
end_pos = content.find(view2_marker)

if start_pos == -1 or end_pos == -1:
    raise ValueError(f"Could not locate Stage 2 block. start_pos={start_pos}, end_pos={end_pos}")

print(f"Located Stage 2 block from char {start_pos} to {end_pos}.")

# Construct the new Stage 2 block
stage2_replacement = '''# STAGE 2: PROFESSIONAL ANALYSIS DASHBOARD
# ==========================================

elif nav_selection == "📊 Analysis Dashboard" or st.session_state.workflow_stage == "dashboard":

    cur = st.session_state.current_analysis

    if cur is None and st.session_state.selected_image is not None:
        st.session_state.workflow_stage = "input"
        st.rerun()

    if cur is None:
        st.info("ℹ️ No leaf specimen currently analyzed. Please return to the Home page to capture or upload an image.")
        if st.button("⬅️ Return to Home Page"):
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

'''

# Perform the replacement
new_content = content[:start_pos] + stage2_replacement + content[end_pos:]
app_path.write_text(new_content, encoding="utf-8")
print("Successfully replaced Stage 2 block in app.py!")

