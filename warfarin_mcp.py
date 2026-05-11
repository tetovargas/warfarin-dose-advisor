#!/usr/bin/env python3
"""
Warfarin Dose Advisor — MCP Server for Claude Code

Exposes three tools directly to Claude Code (no separate API server needed):
  - get_patient_history        → loads demographics + INR history from cohort
  - check_drug_food_interaction → queries the 23-entry interaction database
  - get_dose_recommendation    → runs XGBoost prediction + safety guardrails

Configure in Claude Code  (~/.claude.json  or  claude_desktop_config.json):

  {
    "mcpServers": {
      "warfarin-advisor": {
        "command": "python3",
        "args": ["/absolute/path/to/warfarin_mcp.py"]
      }
    }
  }

Then open Claude Code in the project directory and use natural language:
  "Run a warfarin check-in for patient P045"
  "Check if amiodarone interacts with warfarin"
  "What dose should P120 be on given INR 2.1 and one missed dose?"
"""

import asyncio
import json
import sys
import math
from pathlib import Path

# Ensure project root is on sys.path regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from predictor import predictor
from interactions import search_interactions

# ── Boot the model once at startup ───────────────────────────────────────────
import logging
logging.basicConfig(level=logging.WARNING)   # suppress uvicorn-style INFO noise

TARGET_INR_MIN, TARGET_INR_MAX = 2.5, 3.5

predictor.train()

server = Server("warfarin-advisor")


# ── Tool definitions ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_patient_history",
            description=(
                "Load a patient's demographics and 20-week INR history from the synthetic cohort. "
                "Returns age, sex, weight, CYP2C9/VKORC1 genotype, comorbidities, "
                "time-in-therapeutic-range, and the last 5 weekly records. "
                "Valid patient IDs: P001 to P250."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "Patient ID, e.g. P045"
                    }
                },
                "required": ["patient_id"]
            }
        ),
        Tool(
            name="check_drug_food_interaction",
            description=(
                "Check whether a drug, supplement, or food interacts with warfarin. "
                "Returns effect on INR (Increase/Decrease/None), magnitude, severity "
                "(INFO/WARNING/CRITICAL), mechanism, and recommended clinical action. "
                "Use this for any medication, herbal remedy, or food the patient mentions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Drug, supplement, or food name to look up"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_dose_recommendation",
            description=(
                "Compute the recommended warfarin dose using the XGBoost model and run all "
                "8 safety guardrail checks. Returns recommended_dose_mg, dose_action "
                "(Increase/Decrease/Maintain/Hold), confidence (High/Moderate/Low), "
                "clinical_alerts (INFO/WARNING/CRITICAL), and clinical_notes. "
                "Target INR for mechanical mitral valve: 2.5–3.5."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_id":              {"type": "string"},
                    "age":                     {"type": "integer", "minimum": 18, "maximum": 100},
                    "sex":                     {"type": "string", "enum": ["M", "F"]},
                    "weight_kg":               {"type": "number"},
                    "bmi":                     {"type": "number"},
                    "cyp2c9":                  {"type": "string", "enum": ["*1/*1","*1/*2","*1/*3","*2/*2","*2/*3","*3/*3"]},
                    "vkorc1":                  {"type": "string", "enum": ["GG","AG","AA"]},
                    "inr_result":              {"type": "number", "minimum": 0.5, "maximum": 10.0},
                    "current_dose_mg":         {"type": "number", "minimum": 0.5, "maximum": 20.0},
                    "inr_lag1":                {"type": "number"},
                    "inr_lag2":                {"type": "number"},
                    "dose_lag1":               {"type": "number"},
                    "vitk_level":              {"type": "string", "enum": ["Low","Normal","High"]},
                    "missed_doses_this_week":  {"type": "integer", "minimum": 0, "maximum": 7},
                    "medication_inr_effect":   {"type": "string", "enum": ["None","Increase","Decrease"]},
                    "medication_category":     {"type": "string"},
                    "illness_event":           {"type": "string", "enum": ["None","Mild","Moderate","Severe"]},
                    "diarrhea_vomiting":       {"type": "boolean"},
                    "alcohol_units_per_week":  {"type": "number"},
                    "cranberry_juice":         {"type": "boolean"},
                    "grapefruit":              {"type": "boolean"},
                    "garlic_supplement":       {"type": "boolean"},
                    "exercise_level":          {"type": "string", "enum": ["Sedentary","Moderate","Active"]},
                    "has_hypertension":        {"type": "boolean"},
                    "has_diabetes":            {"type": "boolean"},
                    "has_heart_failure":       {"type": "boolean"},
                    "has_atrial_fibrillation": {"type": "boolean"},
                    "has_ckd":                 {"type": "boolean"},
                    "has_liver_disease":       {"type": "boolean"},
                    "has_hypothyroidism":      {"type": "boolean"},
                    "has_hyperthyroidism":     {"type": "boolean"},
                    "has_copd":                {"type": "boolean"},
                    "has_cad":                 {"type": "boolean"},
                },
                "required": ["age", "sex", "weight_kg", "bmi", "cyp2c9", "vkorc1",
                             "inr_result", "current_dose_mg"]
            }
        )
    ]


# ── Tool execution ─────────────────────────────────────────────────────────────

def _clean(obj):
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    # ── get_patient_history ───────────────────────────────────────────────────
    if name == "get_patient_history":
        pid = arguments["patient_id"].upper()
        demo, weekly = predictor.get_patient_data(pid)
        if demo is None:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Patient {pid} not found. Valid range: P001 to P250."
            }))]

        weekly = weekly.fillna(0)
        inrs   = weekly["INR_Result"].tolist()
        stats  = weekly["INR_Status"].tolist() if "INR_Status" in weekly.columns else []
        ttr    = round(stats.count("Therapeutic") / len(stats) * 100, 1) if stats else 0.0

        last5  = []
        for _, r in weekly.tail(5).iterrows():
            last5.append({
                "week":          int(r.get("Week_Number", 0)),
                "inr":           float(r.get("INR_Result", 0)),
                "status":        str(r.get("INR_Status", "")),
                "dose_mg":       float(r.get("Current_Warfarin_Dose_mg_day", 0)),
                "dose_action":   str(r.get("Dose_Action", "")),
                "missed_doses":  int(r.get("Missed_Doses_This_Week", 0)),
                "medication":    str(r.get("Concurrent_Medication", "None")),
            })

        result = {
            "patient_id":          pid,
            "age":                 int(demo.get("Age", 0)),
            "sex":                 str(demo.get("Sex", "")),
            "weight_kg":           float(demo.get("Weight_kg", 0)),
            "bmi":                 float(demo.get("BMI", 0)),
            "cyp2c9":              str(demo.get("CYP2C9_Genotype", "")),
            "vkorc1":              str(demo.get("VKORC1_Genotype", "")),
            "comorbidities":       str(demo.get("Comorbidities", "None")),
            "baseline_dose_mg":    float(demo.get("Baseline_Warfarin_Dose_mg_day", 0)),
            "mean_inr":            round(sum(inrs) / len(inrs), 2) if inrs else 0.0,
            "pct_time_therapeutic": ttr,
            "target_inr":          "2.5 – 3.5 (Mechanical Mitral Valve)",
            "last_5_weeks":        last5,
        }
        return [TextContent(type="text", text=json.dumps(_clean(result), indent=2))]

    # ── check_drug_food_interaction ───────────────────────────────────────────
    elif name == "check_drug_food_interaction":
        matches = search_interactions(arguments["query"])
        if not matches:
            result = {
                "query":   arguments["query"],
                "matches": [],
                "summary": (
                    f"No known warfarin interaction found for '{arguments['query']}'. "
                    "This does not guarantee safety — verify with a clinical pharmacist."
                )
            }
        else:
            severities = [m["severity"] for m in matches]
            if "CRITICAL" in severities:
                summary = f"CRITICAL interaction for '{arguments['query']}'. Immediate INR monitoring required."
            elif "WARNING" in severities:
                summary = f"Moderate interaction for '{arguments['query']}'. Monitor INR closely."
            else:
                summary = f"Low-level interaction for '{arguments['query']}'. Routine monitoring sufficient."
            result = {"query": arguments["query"], "matches": matches, "summary": summary}
        return [TextContent(type="text", text=json.dumps(_clean(result), indent=2))]

    # ── get_dose_recommendation ───────────────────────────────────────────────
    elif name == "get_dose_recommendation":
        from schemas import (
            PredictRequest, CYP2C9, VKORC1, VitKLevel,
            IllnessLevel, ExerciseLevel, MedEffect
        )

        # Map string values to enums with safe defaults
        def e(cls, val, default):
            try:
                return cls(val) if val else default
            except Exception:
                return default

        req = PredictRequest(
            patient_id              = arguments.get("patient_id"),
            age                     = arguments["age"],
            sex                     = arguments["sex"],
            weight_kg               = arguments["weight_kg"],
            bmi                     = arguments["bmi"],
            cyp2c9                  = e(CYP2C9,        arguments.get("cyp2c9"),               CYP2C9.wt),
            vkorc1                  = e(VKORC1,        arguments.get("vkorc1"),               VKORC1.mid),
            inr_result              = arguments["inr_result"],
            current_dose_mg         = arguments["current_dose_mg"],
            inr_lag1                = arguments.get("inr_lag1"),
            inr_lag2                = arguments.get("inr_lag2"),
            dose_lag1               = arguments.get("dose_lag1"),
            vitk_level              = e(VitKLevel,     arguments.get("vitk_level"),           VitKLevel.normal),
            missed_doses_this_week  = arguments.get("missed_doses_this_week", 0),
            medication_inr_effect   = e(MedEffect,     arguments.get("medication_inr_effect"), MedEffect.none),
            medication_category     = arguments.get("medication_category", "None"),
            illness_event           = e(IllnessLevel,  arguments.get("illness_event"),        IllnessLevel.none),
            diarrhea_vomiting       = arguments.get("diarrhea_vomiting", False),
            alcohol_units_per_week  = arguments.get("alcohol_units_per_week", 0.0),
            cranberry_juice         = arguments.get("cranberry_juice", False),
            grapefruit              = arguments.get("grapefruit", False),
            garlic_supplement       = arguments.get("garlic_supplement", False),
            exercise_level          = e(ExerciseLevel, arguments.get("exercise_level"),       ExerciseLevel.moderate),
            has_atrial_fibrillation = arguments.get("has_atrial_fibrillation", False),
            has_heart_failure       = arguments.get("has_heart_failure", False),
            has_hypertension        = arguments.get("has_hypertension", False),
            has_diabetes            = arguments.get("has_diabetes", False),
            has_ckd                 = arguments.get("has_ckd", False),
            has_hypothyroidism      = arguments.get("has_hypothyroidism", False),
            has_hyperthyroidism     = arguments.get("has_hyperthyroidism", False),
            has_liver_disease       = arguments.get("has_liver_disease", False),
            has_copd                = arguments.get("has_copd", False),
            has_cad                 = arguments.get("has_cad", False),
        )

        pred = predictor.predict(req)
        raw_dose = pred["raw_dose"]
        if req.inr_result > 5.0:
            raw_dose = 0.0

        # Inline safety checks (mirrors main.py run_safety_checks)
        alerts = []
        inr = req.inr_result
        if inr > 5.0:
            alerts.append({"severity": "CRITICAL", "code": "INR_CRITICAL_HIGH",
                "message": f"INR {inr} critically elevated (>5.0). Major bleeding risk.",
                "action": "HOLD warfarin. Consider Vitamin K 1-2.5 mg PO. Urgent clinical review."})
        elif inr > TARGET_INR_MAX:
            alerts.append({"severity": "WARNING", "code": "INR_SUPRATHERAPEUTIC",
                "message": f"INR {inr} above therapeutic range (2.5-3.5).",
                "action": "Reduce dose. Recheck INR in 3-5 days."})
        elif inr < 1.5:
            alerts.append({"severity": "CRITICAL", "code": "INR_CRITICALLY_LOW",
                "message": f"INR {inr} critically low (<1.5). High thromboembolic risk.",
                "action": "Increase dose urgently. Consider bridging therapy. Urgent cardiology review."})
        elif inr < TARGET_INR_MIN:
            alerts.append({"severity": "WARNING", "code": "INR_SUBTHERAPEUTIC",
                "message": f"INR {inr} below therapeutic range. Stroke/thrombosis risk elevated.",
                "action": "Increase dose. Recheck INR in 5-7 days."})
        if raw_dose > 15.0:
            alerts.append({"severity": "WARNING", "code": "DOSE_UNUSUALLY_HIGH",
                "message": f"Predicted dose {raw_dose} mg/day exceeds typical maximum.",
                "action": "Verify weight, genotype, and interactions before dispensing."})
        change_pct = abs((raw_dose - req.current_dose_mg) / req.current_dose_mg) * 100
        if change_pct > 30:
            alerts.append({"severity": "WARNING", "code": "DOSE_LARGE_CHANGE",
                "message": f"Dose change {change_pct:.1f}% — unusually large adjustment.",
                "action": "Review precipitating factors. Consider staged titration."})
        if req.missed_doses_this_week >= 2:
            alerts.append({"severity": "WARNING", "code": "ADHERENCE_CONCERN",
                "message": f"{req.missed_doses_this_week} missed dose(s) this week.",
                "action": "Counsel on adherence. Do NOT double-dose."})
        if req.medication_inr_effect.value == "Increase":
            alerts.append({"severity": "WARNING", "code": "DRUG_INTERACTION_POTENTIATOR",
                "message": "Concurrent medication with INR-potentiating effect detected.",
                "action": "Recheck INR 3-5 days after starting medication."})
        elif req.medication_inr_effect.value == "Decrease":
            alerts.append({"severity": "WARNING", "code": "DRUG_INTERACTION_REDUCER",
                "message": "Concurrent medication reducing INR detected.",
                "action": "INR may drop. Check INR in 5-7 days; plan dose increase."})

        pct = (raw_dose - req.current_dose_mg) / req.current_dose_mg * 100 if req.current_dose_mg else 0
        if raw_dose == 0:
            action = "Hold"
        elif pct > 5:
            action = "Increase"
        elif pct < -5:
            action = "Decrease"
        else:
            action = "Maintain"

        flags = sum([req.medication_inr_effect.value != "None",
                     req.missed_doses_this_week >= 2,
                     req.illness_event.value in ["Moderate", "Severe"],
                     req.diarrhea_vomiting])
        confidence = "High" if flags == 0 else ("Moderate" if flags <= 2 else "Low")

        result = {
            "patient_id":        req.patient_id,
            "recommended_dose_mg": round(raw_dose, 2),
            "dose_change_mg":    round(raw_dose - req.current_dose_mg, 2),
            "dose_change_pct":   round(pct, 1),
            "dose_action":       action,
            "confidence":        confidence,
            "model":             "XGBoost v1.0 (synthetic cohort, 3800 training records)",
            "inr_target":        "2.5 – 3.5 (Mechanical Mitral Valve)",
            "clinical_alerts":   alerts,
            "disclaimer":        "Research prototype only. Confirm all dose changes with anticoagulation clinician."
        }
        return [TextContent(type="text", text=json.dumps(_clean(result), indent=2))]

    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
