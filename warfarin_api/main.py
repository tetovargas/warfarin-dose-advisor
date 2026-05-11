"""
Warfarin Dose Recommendation API
Run: uvicorn main:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import time
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    PredictRequest, PredictResponse, ClinicalAlert, AlertSeverity,
    InteractionRequest, InteractionResponse, InteractionMatch,
    PatientHistoryResponse, WeekRecord,
    HealthResponse, ModelInfoResponse, TABULAR_FEATURES
)
from predictor import predictor, MODEL_VERSION, TRAINING_METRICS
from interactions import search_interactions

START_TIME = time.time()
TARGET_INR_MIN, TARGET_INR_MAX = 2.5, 3.5

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n⚕  Warfarin Dose API — starting up...")
    predictor.train()
    print("⚕  API ready.\n")
    yield

app = FastAPI(
    title="Warfarin Dose Recommendation API",
    description=(
        "Clinical decision-support API for warfarin dosing in mechanical mitral valve patients. "
        "Powered by XGBoost trained on a synthetic cohort of 250 patients (5,000 weekly records). "
        "Target INR: **2.5–3.5**. "
        "⚠️ For research/educational use only — not validated for clinical deployment."
    ),
    version=MODEL_VERSION, lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Safety guardrails ─────────────────────────────────────────────────────────
def run_safety_checks(req: PredictRequest, predicted_dose: float) -> List[ClinicalAlert]:
    alerts = []
    inr = req.inr_result

    if inr > 5.0:
        alerts.append(ClinicalAlert(severity=AlertSeverity.critical, code="INR_CRITICAL_HIGH",
            message=f"INR {inr} is critically elevated (>5.0). Major bleeding risk.",
            recommended_action="HOLD warfarin. Consider Vitamin K 1–2.5 mg PO. Urgent clinical review."))
    elif inr > TARGET_INR_MAX:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="INR_SUPRATHERAPEUTIC",
            message=f"INR {inr} above therapeutic range ({TARGET_INR_MIN}–{TARGET_INR_MAX}).",
            recommended_action="Reduce warfarin. Recheck INR in 3–5 days. Review dietary/drug changes."))
    elif inr < 1.5:
        alerts.append(ClinicalAlert(severity=AlertSeverity.critical, code="INR_CRITICALLY_LOW",
            message=f"INR {inr} critically low (<1.5). High thromboembolic risk for mechanical valve patient.",
            recommended_action="Increase dose urgently. Consider bridging therapy (heparin). Urgent cardiology review."))
    elif inr < TARGET_INR_MIN:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="INR_SUBTHERAPEUTIC",
            message=f"INR {inr} below therapeutic range. Stroke/thrombosis risk elevated.",
            recommended_action="Increase warfarin dose. Recheck INR in 5–7 days."))

    if predicted_dose > 15.0:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="DOSE_UNUSUALLY_HIGH",
            message=f"Predicted dose {predicted_dose} mg/day exceeds typical maximum (15 mg/day).",
            recommended_action="Verify weight, genotype, and drug interactions before dispensing."))
    if predicted_dose < 1.0 and predicted_dose > 0:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="DOSE_UNUSUALLY_LOW",
            message=f"Predicted dose {predicted_dose} mg/day below typical minimum (1.0 mg/day).",
            recommended_action="Confirm high warfarin sensitivity (CYP2C9 PM + VKORC1 AA). Check interactions."))

    change_pct = abs((predicted_dose - req.current_dose_mg) / req.current_dose_mg) * 100
    if change_pct > 30:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="DOSE_LARGE_CHANGE",
            message=f"Predicted dose change {change_pct:.1f}% — unusually large adjustment.",
            recommended_action="Review precipitating factors (new drug, diet, illness). Consider staged titration."))

    if req.medication_inr_effect.value == "Increase" and req.medication_category in ["Antifungal","Antibiotic","Antiarrhythmic"]:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="DRUG_INTERACTION_POTENTIATOR",
            message=f"Concurrent {req.medication_category} with INR-potentiating effect detected.",
            recommended_action="Recheck INR 3–5 days after starting medication. Monitor for bleeding."))
    if req.medication_inr_effect.value == "Decrease":
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="DRUG_INTERACTION_REDUCER",
            message="Concurrent medication reducing INR detected (enzyme inducer).",
            recommended_action="INR may drop significantly. Check INR in 5–7 days; plan dose increase accordingly."))

    if req.missed_doses_this_week >= 2:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="ADHERENCE_CONCERN",
            message=f"Patient reported {req.missed_doses_this_week} missed dose(s) this week.",
            recommended_action="Counsel on adherence. Do NOT double-dose. Recheck INR in 5–7 days."))
    if req.illness_event.value in ["Moderate","Severe"] or req.diarrhea_vomiting:
        alerts.append(ClinicalAlert(severity=AlertSeverity.warning, code="ILLNESS_INR_RISK",
            message="Active illness or GI disturbance may unpredictably alter INR.",
            recommended_action="Monitor INR every 3–5 days until clinically stable."))
    return alerts


def classify_action(current, predicted):
    pct = (predicted - current) / current * 100
    if predicted == 0: return "Hold", "Warfarin held — INR critically elevated. Resume at reduced dose after review."
    if pct >  5: return "Increase", "INR expected to rise toward therapeutic range"
    if pct < -5: return "Decrease", "INR expected to fall toward therapeutic range"
    return "Maintain", "INR expected to remain stable"


def confidence(req):
    flags = sum([req.medication_inr_effect.value != "None", req.missed_doses_this_week >= 2,
                 req.illness_event.value in ["Moderate","Severe"], req.diarrhea_vomiting,
                 req.vitk_level.value == "High", req.alcohol_units_per_week > 14, req.cranberry_juice])
    return "High" if flags == 0 else ("Moderate" if flags <= 2 else "Low")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
def health():
    """Liveness check — confirm API and model are operational."""
    return HealthResponse(status="ok", model_loaded=predictor.is_ready,
        dataset_loaded=predictor.dataset is not None,
        uptime_seconds=round(time.time()-START_TIME,1), version=MODEL_VERSION)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Infrastructure"])
def model_info():
    """Return model metadata, features, training performance, and safety configuration."""
    return ModelInfoResponse(
        model_type="XGBoost Regressor (tabular snapshot)", version=MODEL_VERSION,
        training_patients=200, training_weeks=20,
        target_variable="New_Warfarin_Dose_mg_day", features=TABULAR_FEATURES,
        performance=TRAINING_METRICS,
        inr_target_range={"min":TARGET_INR_MIN,"max":TARGET_INR_MAX,"indication":"Mechanical Mitral Valve"},
        safety_guardrails=[
            "Rejects INR values outside 0.5–10.0",
            "Flags predicted dose >15 mg/day or <1.0 mg/day",
            "Alerts when dose change exceeds 30% of current dose",
            "CRITICAL alert for INR >5.0 — hold + Vitamin K reversal guidance",
            "CRITICAL alert for INR <1.5 — bridging therapy recommendation",
            "Drug interaction severity flags (INFO / WARNING / CRITICAL)",
            "Adherence alert for ≥2 missed doses this week",
            "Illness/GI disturbance INR instability warning",
        ])


@app.post("/predict", response_model=PredictResponse, tags=["Clinical"])
def predict_dose(req: PredictRequest):
    """
    **Warfarin Dose Recommendation**

    Provide current INR, dose, patient demographics, and weekly clinical factors.
    Returns the recommended next dose, dose action, confidence, and clinical safety alerts.
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Retry in a moment.")

    result = predictor.predict(req)
    pred_dose = result["raw_dose"]

    if req.inr_result > 5.0:
        pred_dose = 0.0

    action, direction = classify_action(req.current_dose_mg, pred_dose)
    dose_change = round(pred_dose - req.current_dose_mg, 2)
    dose_change_pct = round((dose_change / req.current_dose_mg) * 100, 1) if req.current_dose_mg > 0 else 0.0
    alerts = run_safety_checks(req, pred_dose)

    notes = []
    if req.vitk_level.value == "High":
        notes.append("High Vitamin K intake — counsel patient on dietary consistency.")
    if req.alcohol_units_per_week > 14:
        notes.append(f"Alcohol {req.alcohol_units_per_week} units/week exceeds safe threshold. Counsel reduction.")
    if req.cyp2c9.value in ["*2/*3","*3/*3"]:
        notes.append("Poor CYP2C9 metabolizer — inherently high warfarin sensitivity. Use lowest effective dose.")
    if req.vkorc1.value == "AA":
        notes.append("VKORC1 AA genotype — high warfarin sensitivity. Dose with extra caution.")
    if req.has_liver_disease:
        notes.append("Liver disease — impaired clotting factor synthesis amplifies anticoagulant effect.")
    if req.has_hyperthyroidism:
        notes.append("Hyperthyroidism increases catabolism of clotting factors — INR may be labile.")
    if not notes:
        notes.append("No additional clinical notes. Routine monitoring applies.")

    return PredictResponse(
        patient_id=req.patient_id, recommended_dose_mg=pred_dose,
        dose_change_mg=dose_change, dose_change_pct=dose_change_pct,
        dose_action=action, predicted_inr_direction=direction,
        confidence=confidence(req), model_used=f"XGBoost v{MODEL_VERSION}",
        clinical_alerts=alerts, clinical_notes=notes)


@app.post("/interaction-check", response_model=InteractionResponse, tags=["Clinical"])
def interaction_check(req: InteractionRequest):
    """
    **Drug & Food Interaction Checker**

    Enter any drug name, food, or supplement. Returns interaction details,
    mechanism, severity, and recommended clinical action.
    """
    raw = search_interactions(req.query)
    if not raw:
        return InteractionResponse(query=req.query, matches=[],
            summary=f"No known warfarin interactions found for '{req.query}'. "
                    "This does not guarantee safety — verify with a clinical pharmacist.")

    matches = [InteractionMatch(name=m["name"].title(), category=m["category"],
        effect_on_inr=m["effect_on_inr"], magnitude=m["magnitude"],
        mechanism=m["mechanism"], clinical_action=m["clinical_action"],
        severity=m["severity"]) for m in raw]

    if any(m.severity == AlertSeverity.critical for m in matches):
        summary = f"⚠️ CRITICAL interaction for '{req.query}'. Immediate INR monitoring and dose adjustment required."
    elif any(m.severity == AlertSeverity.warning for m in matches):
        summary = f"Moderate interaction for '{req.query}'. Monitor INR closely and consider dose adjustment."
    else:
        summary = f"Low-level interaction noted for '{req.query}'. Routine monitoring is sufficient."

    return InteractionResponse(query=req.query, matches=matches, summary=summary)


@app.get("/patient/{patient_id}", response_model=PatientHistoryResponse, tags=["Clinical"])
def get_patient(patient_id: str = Path(..., description="Patient ID e.g. P001")):
    """
    **Patient History & TTR Report**

    Full 20-week INR trajectory, time-in-therapeutic-range, dose history,
    and adverse event summary for any patient in the synthetic cohort (P001–P250).
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Dataset not yet loaded.")

    demo, weekly = predictor.get_patient_data(patient_id.upper())
    if demo is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found. Use P001–P250.")

    weekly = weekly.fillna("")
    inrs = weekly["INR_Result"].tolist()
    stats = weekly["INR_Status"].tolist() if "INR_Status" in weekly.columns else []
    mean_inr  = round(sum(inrs)/len(inrs), 2) if inrs else 0.0
    pct_ther  = round(stats.count("Therapeutic")/len(stats)*100, 1) if stats else 0.0
    pct_sub   = round(stats.count("Subtherapeutic")/len(stats)*100, 1) if stats else 0.0
    pct_supra = round((stats.count("Supratherapeutic")+stats.count("Critical High"))/len(stats)*100, 1) if stats else 0.0
    dose_adjs = int((weekly["Dose_Action"] != "Maintain").sum()) if "Dose_Action" in weekly.columns else 0
    bleeds    = int((weekly["Bleeding_Event"] != "None").sum()) if "Bleeding_Event" in weekly.columns else 0
    clots     = int((weekly["Thromboembolic_Event"] == "Yes").sum()) if "Thromboembolic_Event" in weekly.columns else 0

    records = [WeekRecord(
        week=int(r.get("Week_Number",0)), date=str(r.get("Visit_Date","")),
        inr=float(r.get("INR_Result",0)), status=str(r.get("INR_Status","")),
        dose_mg=float(r.get("Current_Warfarin_Dose_mg_day",0)),
        dose_action=str(r.get("Dose_Action","")),
        medication=str(r.get("Concurrent_Medication","None")),
        vitk_level=str(r.get("VitK_Intake_Level","")),
        missed_doses=int(r.get("Missed_Doses_This_Week",0)),
        bleeding_event=str(r.get("Bleeding_Event","None")),
        clot_event=str(r.get("Thromboembolic_Event","No")),
    ) for _,r in weekly.iterrows()]

    return PatientHistoryResponse(
        patient_id=patient_id.upper(), age=int(demo.get("Age",0)),
        sex=str(demo.get("Sex","")), cyp2c9=str(demo.get("CYP2C9_Genotype","")),
        vkorc1=str(demo.get("VKORC1_Genotype","")),
        comorbidities=str(demo.get("Comorbidities","None")),
        baseline_dose_mg=float(demo.get("Baseline_Warfarin_Dose_mg_day",0)),
        target_inr_min=TARGET_INR_MIN, target_inr_max=TARGET_INR_MAX,
        weeks_observed=len(records), mean_inr=mean_inr,
        pct_time_therapeutic=pct_ther, pct_time_subtherapeutic=pct_sub,
        pct_time_supratherapeutic=pct_supra, total_dose_adjustments=dose_adjs,
        bleeding_events=bleeds, thromboembolic_events=clots, weekly_records=records)
