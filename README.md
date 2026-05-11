# ClinicalDose — Warfarin Dose Advisor

AI-powered warfarin dose recommendation for mechanical mitral valve patients.
Built for BU.330.760 Generative AI for Business, JHU Carey Business School, Spring II 2026.

## Architecture

```
Patient natural language
        ↓
Claude (claude-sonnet-4-6) — conversational intake via Claude Code or chatbot
        ↓  tool calls
┌─────────────────────────────────────┐
│  MCP Server  (warfarin_mcp.py)      │
│  ┌─────────────────────────────┐    │
│  │ XGBoost model  (R²=0.998)   │    │
│  │ Safety guardrail engine     │    │
│  │ Drug/food interaction DB    │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
        ↓
Dose recommendation + clinical alerts
```

GenAI (Claude) handles the natural language interface. The XGBoost engine handles numeric prediction. The safety engine handles clinical validation.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/warfarin-dose-advisor.git
cd warfarin-dose-advisor
pip install -r requirements.txt
```

## Usage — Claude Code (recommended)

Add to `~/.claude.json`:

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

Then in Claude Code:
```
Run a weekly warfarin check-in for patient P045
Check if amiodarone interacts with warfarin
Patient P120 has INR 1.9, missed two doses, started fluconazole — what dose?
```

See [CLAUDE.md](CLAUDE.md) for full Claude Code setup instructions.

## Usage — Standalone chatbot

```bash
# Terminal 1: start the API
python3 -m uvicorn main:app --port 8000

# Terminal 2: run the chatbot
export ANTHROPIC_API_KEY=your_key_here
python3 intake_agent.py P045
```

## Usage — REST API

```bash
python3 -m uvicorn main:app --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
```

Dose recommendation endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67, "sex": "F", "weight_kg": 72, "bmi": 26.5,
    "cyp2c9": "*1/*1", "vkorc1": "AG",
    "inr_result": 2.1, "current_dose_mg": 5.0,
    "missed_doses_this_week": 1, "vitk_level": "Normal"
  }'
```

## Run tests

```bash
python3 -m uvicorn main:app --port 8000 &
sleep 10
python3 test_api.py
```

All 10 test cases should pass.

## Model performance (synthetic cohort, held-out test set)

| Metric | Value |
|---|---|
| RMSE | 0.128 mg/week |
| R² | 0.998 |
| % within ±1 mg | 100% |
| Inference latency | ~5 ms |

## Disclaimer

Research and educational prototype. Trained on synthetic data only (250 patients, 20 weeks). Not validated for clinical deployment. All recommendations require review by a licensed anticoagulation clinician.
