# ClinicalDose — Warfarin Dose Advisor

**BU.330.760 Generative AI for Business | JHU Carey Business School | Spring II 2026**

AI-powered warfarin dose recommendation for mechanical mitral valve patients, designed to run as a live skill inside Claude Code via the Model Context Protocol (MCP).

---

## 1. Context, User, and Problem

### Who the user is

The primary user is an **anticoagulation clinic nurse or clinical pharmacist** who manages warfarin therapy for patients with mechanical mitral valve replacements. A secondary user is the **patient themselves**, who interacts with the conversational intake agent before or during a visit.

### What workflow is being improved

Every 1–4 weeks, each patient presents with a new INR (International Normalized Ratio) lab result. The clinician must decide whether to increase, decrease, maintain, or hold the warfarin dose. For mechanical mitral valve patients the target INR is **2.5–3.5**. Too low risks thromboembolic stroke; too high risks major bleeding.

This decision requires simultaneously weighing:
- Current and recent INR trajectory (lag features)
- Pharmacogenomics: CYP2C9 and VKORC1 genotype determine how fast the patient metabolizes warfarin
- Concurrent medications and food interactions (23+ known interactions)
- Weekly adherence, dietary changes, illness, and alcohol intake

Manual adjustment with paper nomograms is slow, variable across clinicians, and ignores pharmacogenomics entirely.

### Why it matters

Warfarin is the most common cause of serious adverse drug events in outpatient care. Mechanical valve patients face lifelong anticoagulation with narrow therapeutic windows. A consistent, evidence-based, pharmacogenomics-aware decision-support tool directly reduces preventable harm.

---

## 2. Solution and Design

### Architecture

```
Patient natural language
        ↓
Claude (claude-sonnet-4-6) — conversational reasoning via Claude Code
        ↓  MCP tool calls
┌──────────────────────────────────────────┐
│  warfarin_mcp.py  (MCP Server)           │
│  ┌────────────────────────────────────┐  │
│  │ XGBoost regression model           │  │
│  │ RMSE 0.128 mg  |  R² 0.998         │  │
│  │ 100% predictions within ±1 mg      │  │
│  ├────────────────────────────────────┤  │
│  │ Safety guardrail engine (8 checks) │  │
│  │ INFO / WARNING / CRITICAL tiers    │  │
│  ├────────────────────────────────────┤  │
│  │ Drug/food interaction database     │  │
│  │ 23 entries, fuzzy matching         │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        ↓
Dose recommendation + safety alerts + clinical notes
```

**GenAI (Claude) handles:** natural language intake, clinical reasoning, tool orchestration, plain-language explanation.
**XGBoost handles:** numeric dose prediction from 33 tabular features.
**Safety engine handles:** clinical threshold validation (always enforced, independent of LLM output).

### What was built

**Five components, all tested and running:**

| Component | File | Description |
|---|---|---|
| Synthetic dataset | `generate_warfarin_db.py` | 250 patients × 20 weeks = 5,000 rows |
| ML pipeline | `warfarin_ml_pipeline.py` | RF / XGBoost / LSTM comparison |
| REST API | `main.py` | FastAPI, 5 endpoints, Pydantic v2 |
| MCP server | `warfarin_mcp.py` | Claude Code integration, 3 tools |
| Standalone chatbot | `intake_agent.py` | Claude-powered terminal chatbot |

### MCP tools exposed to Claude Code

| Tool | Endpoint | Purpose |
|---|---|---|
| `get_patient_history` | `GET /patient/{id}` | Demographics + 20-week INR trajectory |
| `check_drug_food_interaction` | `POST /interaction-check` | Drug/food interaction lookup |
| `get_dose_recommendation` | `POST /predict` | XGBoost prediction + safety guardrails |

### Key design choices

**Model selection (Week 2 — Cost/Quality/Latency framework):**
XGBoost was selected over Random Forest and LSTM after head-to-head comparison on the same synthetic cohort. RMSE 0.128 mg vs 0.261 mg (RF) and 0.194 mg (LSTM). LSTM was eliminated on latency grounds (~150 ms vs ~5 ms). No GPU required.

**LLM selection:** Claude claude-sonnet-4-6 over Opus. The intake task is structured extraction and tool calling, not open-ended reasoning. Sonnet provides equivalent accuracy at lower latency and cost for this task profile.

**Tool use / ReAct pattern (Week 4):** Claude Code drives a Think → Act → Observe → Repeat loop. It decides which tools to call and in what order based on conversation state. The MCP server exposes typed JSON schemas; Claude never sees raw model internals.

**Safety-first design:** The guardrail engine runs independently of the LLM. CRITICAL alerts (INR > 5.0, INR < 1.5) are always enforced regardless of what the model predicts or Claude infers.

---

## 3. Evaluation and Results

### ML model performance (held-out test set, 1,000 rows)

| Metric | Value |
|---|---|
| RMSE | 0.128 mg/week |
| R² | 0.998 |
| % within ±1 mg | 100% |
| % within ±2 mg | 100% |
| Inference latency | ~5 ms |

All metrics against synthetic ground truth. Nomogram comparison not yet conducted (noted as a limitation).

### Automated API test suite (10 cases, all passing)

| # | Test | Result |
|---|---|---|
| 1 | GET /health | API running, model loaded ✓ |
| 2 | GET /model-info | 3,800 training records, 33 features ✓ |
| 3 | POST /predict — stable patient (INR 2.8) | MAINTAIN, High confidence, no alerts ✓ |
| 4 | POST /predict — critical INR 5.8 + fluconazole | CRITICAL alert, Hold action ✓ |
| 5 | POST /predict — poor metabolizer (CYP2C9 *3/*3) | Low dose, Moderate confidence ✓ |
| 6 | POST /predict — enzyme inducer (rifampin) | Large dose increase, drug flag ✓ |
| 7 | POST /interaction-check — fluconazole | CRITICAL potentiator returned ✓ |
| 8 | POST /interaction-check — rifampin | CRITICAL reducer returned ✓ |
| 9 | GET /patient/P042 | 20-week history returned ✓ |
| 10 | POST /predict — malformed input | 422 validation error ✓ |

### Limitations

- Trained on synthetic data only; not validated on real patient records
- 23-entry interaction database: drugs not listed pass silently
- No formal baseline comparison against WDS nomogram yet

---

## 4. Artifact Snapshot

### Demo 1 — Stable patient check-in (Claude Code, patient P045)

**Prompt:** `Run a warfarin check-in for patient P045`

**Claude Code response (abridged):**
```
Patient P045 — Check-In Summary

Demographics
  Age / Sex:    49 M
  Weight / BMI: 83.7 kg / 29.8
  Genetics:     CYP2C9 *1/*1 · VKORC1 AG
  Comorbidities: CKD Stage 2
  Target INR:   2.5 – 3.5

This Week's Visit
  INR Result:    3.0 — Therapeutic
  Current Dose:  11.5 mg/day
  Missed Doses:  1
  Medications:   None

Dose Recommendation
  XGBoost Predicted Dose: 11.5 mg/day
  Action:  MAINTAIN (no change)
  Alerts:  None

Clinical Summary
  P045 is well-controlled this week. INR 3.0 sits comfortably in the
  therapeutic window. Continue 11.5 mg/day. Next INR in 4 weeks.
```

### Demo 2 — Critical alert: drug interaction + supratherapeutic INR

**Prompt:** `Patient P045 just told me their INR is 5.8 this week and they started fluconazole for a fungal infection`

**Claude Code response (abridged):**
```
🚨 CRITICAL ALERT — Patient P045

Alert 1 — INR 5.8: Dangerously Supratherapeutic
  HOLD warfarin — do not administer today's dose
  Vitamin K 2.5 mg PO to partially reverse anticoagulation
  Urgent clinical review within 24 hours

Alert 2 — Fluconazole: CRITICAL Drug Interaction
  Effect on INR:  Strong Increase (+1.5–3.0 INR points)
  Mechanism:      Potent CYP2C9 & CYP3A4 inhibitor
  Severity:       CRITICAL

Dose Recommendation
  Current dose:   11.5 mg/day
  Predicted dose: 8.5 mg/day  (−26.1%)
  Action:         HOLD today, restart at 8.5 mg/day when INR < 4.0

Management Plan
  Today:          Hold warfarin. Vitamin K 2.5 mg PO.
  24–48 hrs:      Recheck INR. Restart at 8.5 mg/day if trending down.
  During course:  INR checks every 2–3 days.
  After course:   Retitrate back toward 11.5 mg/day with close monitoring.
```

Claude Code called **5 tools in sequence** for the second demo: patient history load, fluconazole interaction check, dose prediction, safety guardrail evaluation, and management plan synthesis — all from one natural language sentence.

---

## Setup and Usage

### Requirements

- Python 3.10+
- Claude Code (claude.ai/code) with an active Anthropic API key

### Installation

```bash
git clone https://github.com/tetovargas/warfarin-dose-advisor.git
cd warfarin-dose-advisor
pip install -r requirements.txt
```

### Option A: Run as a Claude Code skill (recommended)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "warfarin-advisor": {
      "command": "python3",
      "args": ["/absolute/path/to/warfarin-dose-advisor/warfarin_mcp.py"]
    }
  }
}
```

Open Claude Code in the `warfarin-dose-advisor` folder and try:

```
Run a warfarin check-in for patient P045
Check if amiodarone interacts with warfarin
Patient P120 has INR 1.9, missed two doses, started fluconazole — what dose?
```

No API server needed. The MCP server loads the XGBoost model on first tool call (~8 seconds).

### Option B: Run the standalone terminal chatbot

```bash
# Terminal 1
python3 -m uvicorn main:app --port 8000

# Terminal 2
export ANTHROPIC_API_KEY=your_key_here
python3 intake_agent.py P045
```

### Option C: Call the REST API directly

```bash
python3 -m uvicorn main:app --port 8000
# Swagger UI: http://127.0.0.1:8000/docs

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 49, "sex": "M", "weight_kg": 83.7, "bmi": 29.8,
    "cyp2c9": "*1/*1", "vkorc1": "AG",
    "inr_result": 3.0, "current_dose_mg": 11.5
  }'
```

### Run the test suite

```bash
python3 -m uvicorn main:app --port 8000 &
sleep 10
python3 test_api.py
```

---

## Safety Guardrails

Always enforced regardless of LLM output:

| Condition | Severity | Action |
|---|---|---|
| INR > 5.0 | CRITICAL | HOLD + Vitamin K guidance |
| INR < 1.5 | CRITICAL | Increase urgently + bridging therapy |
| Dose change > 30% | WARNING | Staged titration recommended |
| ≥ 2 missed doses | WARNING | Adherence counseling |
| Drug interaction (potentiator) | WARNING/CRITICAL | INR recheck in 3–5 days |
| Active illness or GI disturbance | WARNING | Monitor every 3–5 days |

## Disclaimer

Research and educational prototype. Trained on synthetic data only (250 patients, 20 weeks). Not validated for clinical deployment. All recommendations require review by a licensed anticoagulation clinician before any dose adjustment.
