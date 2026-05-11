# Warfarin Dose Advisor — Claude Code Skill

## What this project does

ClinicalDose is a warfarin dose recommendation system for patients with mechanical mitral valve replacements. It combines an XGBoost ML model (RMSE 0.128 mg, R² 0.998) with a clinical safety guardrail engine and a 23-entry drug/food interaction database.

This repository is designed to run as a **live skill inside Claude Code** via the MCP server (`warfarin_mcp.py`). Claude Code becomes the conversational clinical intake layer — you use natural language, and Claude calls the structured tools.

## Quick start for Claude Code

### Step 1 — install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — register the MCP server

Add this to your `~/.claude.json` (Claude Code) or `claude_desktop_config.json` (Claude Desktop):

```json
{
  "mcpServers": {
    "warfarin-advisor": {
      "command": "python3",
      "args": ["/absolute/path/to/warfarin_mcp.py"]
    }
  }
}
```

Replace `/absolute/path/to/` with the actual path to your cloned repo.

### Step 3 — open Claude Code in this directory and start talking

```
Run a weekly warfarin check-in for patient P045
```

```
Patient P120 says their INR this week is 1.9, they missed two doses, and started amiodarone. What dose should they be on?
```

```
Check if fluconazole interacts with warfarin
```

```
Pull up the full INR history for patient P089
```

No API server needs to be running. The MCP server loads the XGBoost model directly on startup (takes about 8 seconds the first time).

## Available tools (exposed via MCP)

| Tool | Description |
|---|---|
| `get_patient_history` | Demographics + 20-week INR trajectory for any patient P001–P250 |
| `check_drug_food_interaction` | Warfarin interaction check for any drug, food, or supplement |
| `get_dose_recommendation` | XGBoost dose prediction + 8 safety guardrail checks |

## Alternative: run the standalone chatbot

If you prefer a terminal chatbot instead of Claude Code:

```bash
# Terminal 1
python3 -m uvicorn main:app --port 8000

# Terminal 2
export ANTHROPIC_API_KEY=your_key_here
python3 intake_agent.py P045
```

## Alternative: run the REST API directly

```bash
python3 -m uvicorn main:app --port 8000
# Swagger UI at http://127.0.0.1:8000/docs
```

## Project structure

```
warfarin_mcp.py          MCP server — Claude Code entry point
intake_agent.py          Standalone Claude-powered chatbot
main.py                  FastAPI REST API (5 endpoints)
predictor.py             XGBoost model + training pipeline
interactions.py          23-entry drug/food interaction database
schemas.py               Pydantic v2 request/response schemas
test_api.py              10-case automated API test suite
warfarin_ml_pipeline.py  RF / XGBoost / LSTM comparison pipeline
generate_warfarin_db.py  Synthetic cohort generator (250 patients x 20 weeks)
warfarin_cohort_MMV.xlsx Synthetic dataset (5,000 rows)
warfarin_skill.md        Skill documentation
```

## Safety guardrails (always enforced)

- INR > 5.0: HOLD warfarin — CRITICAL alert + Vitamin K reversal guidance
- INR < 1.5: CRITICAL alert + bridging therapy recommendation
- Dose change > 30%: WARNING — staged titration recommended
- ≥ 2 missed doses: Adherence counseling alert
- Drug interaction detected: severity-tiered alert (INFO / WARNING / CRITICAL)
- Active illness or GI disturbance: INR instability monitoring alert

## Disclaimer

Research and educational prototype. Trained on synthetic data only. Not validated for clinical use. All recommendations must be confirmed by a licensed anticoagulation clinician before any dose adjustment.
