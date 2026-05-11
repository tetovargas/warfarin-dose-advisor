#!/usr/bin/env python3
"""
Warfarin Intake Agent — Claude-powered patient check-in chatbot
Conducts a conversational weekly intake and calls the Warfarin Dose API as tools.

This script is the GenAI layer on top of the XGBoost dose engine:
  - Claude claude-sonnet-4-6 handles natural language intake (missed doses,
    dietary changes, new medications, illness, etc.)
  - Three FastAPI endpoints are exposed as callable tools (Tool Use / Week 4)
  - Claude drives the ReAct loop: ask -> check interactions -> recommend -> explain

Usage:
    python3 intake_agent.py              # new patient, no history
    python3 intake_agent.py P045         # load demographics from synthetic cohort

Prerequisites:
    1. Warfarin API running:  uvicorn main:app --port 8000
    2. ANTHROPIC_API_KEY set in environment
"""

import sys
import json
import os
import httpx
import anthropic

API_BASE = os.getenv("WARFARIN_API_URL", "http://localhost:8000")
MODEL    = "claude-sonnet-4-6"

client = anthropic.Anthropic()


# ── Tool definitions exposed to Claude ───────────────────────────────────────

TOOLS = [
    {
        "name": "get_patient_history",
        "description": (
            "Load a patient's demographics and 20-week INR history from the cohort database. "
            "Call this at the start of the session when the patient provides their ID (e.g. P045). "
            "Returns age, sex, weight, genotype, comorbidities, and last 20 weekly records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier, e.g. P045 (P001 to P250)"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "check_drug_food_interaction",
        "description": (
            "Check whether a specific drug, supplement, or food interacts with warfarin. "
            "Call this whenever the patient mentions taking a new medication, supplement, "
            "herbal remedy, or consuming specific foods such as cranberry juice, grapefruit, "
            "or garlic. Returns effect on INR, severity, mechanism, and clinical action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The drug, food, or supplement name to look up"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_dose_recommendation",
        "description": (
            "Submit all collected patient data to the XGBoost dose engine and receive "
            "the recommended warfarin dose with safety alerts and clinical notes. "
            "Call this once you have gathered the week's clinical information from the patient. "
            "Use defaults for any factor not discussed (vitk_level=Normal, illness_event=None, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_id":              {"type": "string"},
                "age":                     {"type": "integer", "minimum": 18, "maximum": 100},
                "sex":                     {"type": "string", "enum": ["M", "F"]},
                "weight_kg":               {"type": "number"},
                "bmi":                     {"type": "number"},
                "cyp2c9":                  {"type": "string", "enum": ["*1/*1","*1/*2","*1/*3","*2/*2","*2/*3","*3/*3"]},
                "vkorc1":                  {"type": "string", "enum": ["GG","AG","AA"]},
                "inr_result":              {"type": "number", "description": "This week's INR lab result"},
                "current_dose_mg":         {"type": "number", "description": "Current warfarin dose mg/day"},
                "inr_lag1":                {"type": "number", "description": "Last week's INR (optional)"},
                "inr_lag2":                {"type": "number", "description": "Two weeks ago INR (optional)"},
                "dose_lag1":               {"type": "number", "description": "Last week's dose mg/day (optional)"},
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
    }
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a warm, friendly clinical intake assistant at a warfarin anticoagulation clinic specializing in patients with mechanical mitral valves. Your role is to conduct a brief weekly check-in with the patient, gather the clinical information needed for their dose recommendation, and explain the result in plain, reassuring language.

Target INR range for mechanical mitral valve patients: 2.5 to 3.5.

## Workflow

1. Greet the patient warmly and explain this is a quick weekly check-in.

2. If the patient provides an ID (e.g. P045), call get_patient_history immediately to load their demographics and recent chart. Use the most recent weekly record for inr_lag1 and dose_lag1. Confirm weight and current medications are still up to date.

3. If no patient ID is provided, ask for: age, sex, approximate weight, and their current warfarin dose.

4. Ask about this week's clinical factors, one or two questions at a time:
   - INR result from this week's lab test
   - Current warfarin dose (mg/day)
   - Missed doses this week (how many?)
   - Diet changes, especially leafy greens (spinach, kale, broccoli, Brussels sprouts contain Vitamin K and lower INR)
   - Any new medications, supplements, vitamins, or over-the-counter drugs
   - Any illness, fever, diarrhea, or vomiting this week
   - Alcohol intake this week (how many drinks?)
   - Changes in physical activity

5. For every specific drug, supplement, herb, or food the patient mentions, call check_drug_food_interaction. Use the result to set medication_inr_effect and medication_category before calling the dose recommendation.

6. When you have gathered all relevant information, call get_dose_recommendation with the full data. Apply sensible defaults for anything not discussed: vitk_level="Normal", illness_event="None", diarrhea_vomiting=false, alcohol_units_per_week=0, exercise_level="Moderate".

7. Explain the result in simple, friendly language:
   - New recommended dose (plain mg/day)
   - Whether to increase, decrease, or stay the same, and why in one sentence
   - Any safety alerts translated to everyday language (no medical codes)
   - When to come back for the next INR test
   - Always close with: "Please confirm any dose change with your anticoagulation nurse or physician before adjusting."

## Tone rules
- Warm, calm, reassuring. No jargon. No acronyms unexplained.
- One or two questions per turn. Never fire a list of questions at once.
- If INR is critically high (above 5.0) or critically low (below 1.5), be clear about urgency and recommend calling the clinic today.

## Limits
- You are a clinical decision-support tool, not a prescriber.
- Never override or dismiss safety alerts from the system.
- Always recommend clinical review before any dose change."""


# ── Tool execution ─────────────────────────────────────────────────────────────

def run_tool(name: str, inputs: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    try:
        if name == "get_patient_history":
            pid = inputs["patient_id"].upper()
            r = httpx.get(f"{API_BASE}/patient/{pid}", timeout=10)
            if r.status_code == 404:
                return json.dumps({"error": f"Patient {pid} not found (valid range: P001 to P250)."})
            r.raise_for_status()
            return r.text

        elif name == "check_drug_food_interaction":
            r = httpx.post(
                f"{API_BASE}/interaction-check",
                json={"query": inputs["query"]},
                timeout=10
            )
            r.raise_for_status()
            return r.text

        elif name == "get_dose_recommendation":
            r = httpx.post(f"{API_BASE}/predict", json=inputs, timeout=10)
            if r.status_code == 422:
                return json.dumps({"error": "Validation error", "detail": r.json()})
            r.raise_for_status()
            return r.text

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except httpx.ConnectError:
        return json.dumps({
            "error": (
                f"Cannot reach the Warfarin API at {API_BASE}. "
                "Please start it first with:  uvicorn main:app --port 8000"
            )
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_session(patient_id: str | None = None) -> None:
    """
    Run a full patient intake session using the ReAct pattern.

    Think  -> Claude decides what information is still needed
    Act    -> Claude calls a tool (load history, check interaction, get dose)
    Observe -> Tool result is fed back into the conversation
    Repeat -> Until Claude delivers a final plain-language recommendation
    """
    print("\n" + "=" * 60)
    print("  Warfarin Intake Agent  |  Claude claude-sonnet-4-6")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")

    messages = []

    first_msg = (
        f"Hi, my patient ID is {patient_id.upper()}."
        if patient_id
        else "Hi, I'm here for my weekly warfarin check-in."
    )
    print(f"Patient: {first_msg}\n")
    messages.append({"role": "user", "content": first_msg})

    ending_phrases = (
        "have a good", "take care", "see you", "goodbye",
        "bye", "next visit", "next time", "feel better"
    )

    while True:
        # ── Think: call Claude ────────────────────────────────────────────────
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # ── Act: handle tool calls ────────────────────────────────────────────
        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                args_preview = json.dumps(block.input)[:80]
                print(f"  [tool: {block.name}({args_preview}...)]\n")
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # ── Observe: send tool results back ───────────────────────────────────
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            continue  # loop back to Think

        # ── Show text response ────────────────────────────────────────────────
        text = "\n".join(b.text for b in assistant_content if hasattr(b, "text")).strip()
        if text:
            print(f"Agent: {text}\n")

        if response.stop_reason == "end_turn" and any(p in text.lower() for p in ending_phrases):
            break

        # ── Get next patient input ────────────────────────────────────────────
        try:
            user_input = input("Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nSession ended. Goodbye!")
            break

        if not user_input:
            continue

        print()
        messages.append({"role": "user", "content": user_input})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("       export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    patient_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_session(patient_id=patient_id)
