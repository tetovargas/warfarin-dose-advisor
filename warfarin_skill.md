# Warfarin Dose Advisor — Agent Skill

## Overview

This skill exposes the Warfarin Dose Recommendation system as a callable tool
for Claude Code and other agent harnesses (ReAct, agentic pipelines, etc.).

It combines two layers:
- An XGBoost ML model for evidence-based dose prediction (RMSE 0.128 mg, R^2 0.998)
- A Claude claude-sonnet-4-6 conversational layer that conducts the patient intake in
  natural language and calls the ML engine as a structured tool

This is the answer to "where does GenAI fit?": Claude handles the unstructured
patient conversation; the ML model handles the numeric prediction; the safety
guardrail engine handles clinical validation.

---

## When to invoke this skill

Use this skill when:
- A clinician or nurse wants to run a patient's weekly warfarin check-in
- An agent needs to compute a dose recommendation from raw clinical text
- A downstream agent needs to check drug or food interactions with warfarin
- You need to retrieve a patient's full INR history and time-in-therapeutic-range

---

## Prerequisites

1. Start the Warfarin API (from the project directory):

   uvicorn main:app --port 8000

2. Set the Anthropic API key:

   export ANTHROPIC_API_KEY=sk-ant-...

3. Install dependencies (first time only):

   pip install anthropic httpx fastapi uvicorn xgboost pandas scikit-learn

---

## Tool 1: Conversational Patient Intake (Claude-powered)

Run a full weekly check-in session with natural language intake:

   python3 intake_agent.py              # new patient
   python3 intake_agent.py P045         # load history for patient P045

What Claude does inside this session (ReAct loop):

   Think  -> Decides what clinical information is still needed
   Act    -> Calls one of three API tools (see below)
   Observe -> Reads the tool result
   Repeat -> Until a dose recommendation is produced and explained

The session ends with a plain-language explanation of the recommended dose,
any safety alerts in everyday language, and a reminder to confirm with the
anticoagulation nurse before adjusting.

---

## Tool 2: Direct API Endpoints (for programmatic agent use)

These endpoints can be called directly by any agent or script.

### Dose recommendation (POST /predict)

   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "patient_id": "P045",
       "age": 67, "sex": "F",
       "weight_kg": 72.0, "bmi": 26.5,
       "cyp2c9": "*1/*1", "vkorc1": "AG",
       "inr_result": 2.1,
       "current_dose_mg": 5.0,
       "missed_doses_this_week": 1,
       "vitk_level": "Normal"
     }'

Response fields:
   recommended_dose_mg    -> new dose in mg/day
   dose_action            -> Increase / Decrease / Maintain / Hold
   confidence             -> High / Moderate / Low
   clinical_alerts        -> list of INFO / WARNING / CRITICAL alerts
   clinical_notes         -> plain-language clinical guidance

### Drug and food interaction check (POST /interaction-check)

   curl -X POST http://localhost:8000/interaction-check \
     -H "Content-Type: application/json" \
     -d '{"query": "amiodarone"}'

### Patient INR history (GET /patient/{id})

   curl http://localhost:8000/patient/P045

Returns: demographics, genotype, 20-week INR trajectory, time-in-therapeutic-range,
bleeding events, and thromboembolic events.

### Other endpoints

   GET /health         -> liveness check
   GET /model-info     -> model metadata, features, performance, guardrail list
   GET /docs           -> interactive Swagger UI

---

## Tools exposed to Claude inside the intake agent

| Tool name                  | Maps to endpoint         | Purpose                              |
|----------------------------|--------------------------|--------------------------------------|
| get_patient_history        | GET /patient/{id}        | Load demographics and INR history    |
| check_drug_food_interaction| POST /interaction-check  | Warfarin drug/food interaction check |
| get_dose_recommendation    | POST /predict            | XGBoost dose prediction + guardrails |

---

## Safety guardrails (always enforced, regardless of GenAI output)

- INR > 5.0: HOLD warfarin (CRITICAL), Vitamin K reversal guidance
- INR < 1.5: Increase urgently (CRITICAL), bridging therapy recommendation
- Predicted dose > 15 mg/day: WARNING, verify inputs before dispensing
- Dose change > 30% from current: WARNING, consider staged titration
- 2 or more missed doses this week: Adherence counseling alert
- Drug interaction detected (Antifungal, Antibiotic, Antiarrhythmic): severity-tiered alert
- Active illness or GI disturbance: INR instability monitoring alert

---

## Integration with Claude Code (example ReAct harness)

Step 1 (Think):  Patient P045 needs a weekly warfarin check-in.
Step 2 (Act):    Run python3 intake_agent.py P045
Step 3 (Observe): Read the recommended dose and any CRITICAL alerts.
Step 4 (Think):  If a CRITICAL alert is present, escalate to the clinician.
Step 5 (Act):    Notify the anticoagulation nurse with the alert details.

---

## Course concept alignment (JHU Carey GenAI for Business)

Week 2 (Model and Provider Selection):
   XGBoost selected over Random Forest and LSTM based on RMSE, 100% within
   tolerance, and sub-5ms inference with no GPU requirement. Claude claude-sonnet-4-6
   selected over Opus for the conversational layer: tool calling and structured
   extraction do not require deep reasoning; Sonnet provides lower latency and
   cost with equivalent accuracy for this task profile.

Week 4 (Tool Use / Function Calling):
   The three API endpoints are defined as typed JSON schemas and passed to Claude
   as callable tools. Claude decides when to call each tool based on the
   conversation state. This is the agent skill pattern described in the ReAct
   framework: the LLM reasons over natural language, acts by calling typed tools,
   and observes structured results before producing its next response.

---

## Disclaimer

Research and educational prototype only. Trained on synthetic data.
Not validated for clinical deployment. All recommendations must be reviewed
by a licensed anticoagulation clinician before any dose adjustment.
