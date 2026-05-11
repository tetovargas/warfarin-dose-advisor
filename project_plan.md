# ClinicalDose

AI-Powered Warfarin Dose Recommendation System with GenAI Patient Intake Agent

BU.330.760 Generative AI for Business | JHU Carey Business School | Spring II 2026

April-May 2026 | First-Prototype Concept

---

## 1. Project Title

**ClinicalDose:** An AI-Powered Warfarin Dose Recommendation System with Claude-Powered Conversational Patient Intake for Mechanical Mitral Valve Patients

---

## 2. Target User, Workflow, and Business Value

### Who the users are

The system serves two user types. The primary user is an anticoagulation clinic nurse or clinical pharmacist who manages warfarin therapy for patients with mechanical mitral valve replacements. The secondary user is the patient themselves, who interacts directly with the conversational intake agent before or during a clinic visit.

### What task is being improved

The task is weekly warfarin dose adjustment: given a patient's current INR, recent dose history, pharmacogenomic profile (CYP2C9/VKORC1 genotype), comorbidities, and concurrent medications, the clinician must decide whether to increase, maintain, decrease, or hold the dose. For mechanical mitral valve patients, the target INR is 2.5-3.5. Under-anticoagulation risks thromboembolic stroke; over-anticoagulation risks major bleeding.

The intake step is particularly time-consuming: gathering missed dose counts, dietary changes, new medications, and illness events from patients in natural conversation requires clinical attention that could be partially automated.

### Where the workflow begins and ends

- **Begins:** Patient arrives with a new INR result from a laboratory draw or point-of-care device and begins the intake conversation with the agent.
- **Ends:** The clinician reviews the structured recommendation (dose action, confidence, safety alerts) and documents the new weekly dose.

### Why this matters

Manual dose adjustment with paper-based nomograms is time-consuming, highly variable across clinicians, and does not systematically account for pharmacogenomics or drug interactions. The conversational intake layer additionally reduces the burden on clinical staff to manually extract structured data from patient self-reports. This prototype explores whether an LLM-powered conversational front end combined with an ML prediction engine can support that decision faster and more consistently.

---

## 3. Problem Statement and GenAI Fit

### The exact task the system performs

Given a patient's natural language description of their week (missed doses, dietary changes, new medications, illness), Claude extracts the structured clinical variables needed for dose prediction, checks drug and food interactions through the knowledge base, calls the XGBoost prediction engine, and explains the recommendation back to the patient and clinician in plain language.

### Where GenAI fits

GenAI (Claude claude-sonnet-4-6) is the conversational interface layer between the patient and the ML engine. Patients do not report their week in structured JSON. They say things like "I forgot my pill on Tuesday and I've been eating a lot of salads." Claude's role is to:

1. Conduct the patient intake as a warm, natural conversation
2. Extract structured clinical variables from unstructured language (missed_doses_this_week=1, vitk_level=High)
3. Recognize drug and food names from casual mentions and call the interaction checker for each one
4. Assemble the complete PredictRequest payload and call the dose recommendation engine
5. Translate the structured JSON recommendation and safety alerts back into plain, patient-friendly language

The XGBoost model handles the numeric prediction. The safety guardrail engine handles clinical validation. Claude handles everything that requires natural language understanding. This division of responsibility is the core architectural insight: use the right tool for each sub-task.

### Why a simpler tool would not be enough

Standard nomograms use 3-5 variables, have no pharmacogenomic adjustment, no drug interaction layer, and no confidence calibration. A rule-based intake form cannot conduct a flexible conversation or recognize that "I started a new antibiotic this week" requires an interaction check before proceeding. The combination of conversational AI for intake and ML for prediction is what allows the system to handle the full complexity of the clinical scenario.

Prototype scope note: The system was built and tested on synthetic data only. It has not been validated against real patients or compared against a clinical baseline in a formal study.

---

## 4. Planned System Design and Baseline

### What was built

The prototype consists of five components, all implemented and running locally.

#### Component 1: Synthetic patient dataset

- **Script:** generate_warfarin_db.py
- **Output:** warfarin_cohort_MMV.xlsx, 250 patients x 20 weeks = 5,000 rows
- **Features:** INR, weekly dose, CYP2C9/VKORC1 genotype, BMI, creatinine, comorbidity flags, drug/food interaction deltas, illness and adherence flags
- **Simulation:** INR modeled as a function of dose, genotypic sensitivity, and interaction effects with added Gaussian noise

#### Component 2: ML pipeline and model selection

- **Script:** warfarin_ml_pipeline.py
- **Split:** 80/20 train/test (3,800 training records, 1,000 test records)
- **Features engineered:** 33 tabular features including lag INR (1-3 weeks), lag dose, genotype encodings, and binary comorbidity flags
- **Results saved to:** model_comparison_results.csv

| Model | RMSE (mg/wk) | R² | % within ±1 mg | Inference latency |
|---|---|---|---|---|
| Random Forest | 0.261 mg | 0.993 | 97.4% | ~12 ms |
| **XGBoost (selected)** | **0.128 mg** | **0.998** | **100%** | **~5 ms** |
| LSTM | 0.194 mg | 0.996 | 98.8% | ~150 ms |

#### Component 3: FastAPI REST API

- **File:** main.py
- **Server:** Uvicorn at http://127.0.0.1:8000, confirmed running; Swagger docs at /docs
- **Startup:** Model trains on launch, confirmed: "[Predictor] Ready. 3,800 training records."
- **5 endpoints:** GET /health, GET /model-info, POST /predict, POST /interaction-check, GET /patient/{patient_id}
- **Input validation:** Pydantic v2 schemas; invalid inputs return 422 with field-level errors

#### Component 4: Safety guardrail engine and interaction database

- **8 safety checks:** INR_CRITICAL_HIGH (>5.0), INR_CRITICALLY_LOW (<1.5), DOSE_UNUSUALLY_HIGH, DOSE_LARGE_CHANGE (>30%), DRUG_INTERACTION_POTENTIATOR, DRUG_INTERACTION_REDUCER, ADHERENCE_CONCERN, ILLNESS_INR_RISK
- **Severity tiers:** INFO / WARNING / CRITICAL
- **Interaction database:** 23 entries (14 drug, 9 food) in interactions.py; exact and fuzzy (first-4-character) matching
- **Confidence scoring:** High / Moderate / Low based on count of active perturbing factors

#### Component 5: Claude-powered patient intake agent

- **Script:** intake_agent.py
- **Model:** Claude claude-sonnet-4-6 via Anthropic Python SDK
- **Agent skill definition:** warfarin_skill.md (for use in Claude Code or any ReAct harness)

The agent exposes three of the FastAPI endpoints as typed tools that Claude can call:

| Tool name | Maps to endpoint | Purpose |
|---|---|---|
| get_patient_history | GET /patient/{id} | Load demographics and 20-week INR history |
| check_drug_food_interaction | POST /interaction-check | Warfarin drug and food interaction check |
| get_dose_recommendation | POST /predict | XGBoost dose prediction with safety guardrails |

The session runs as a ReAct loop (Module 4):

- **Think:** Claude determines what clinical information is still missing
- **Act:** Claude calls a tool (load history, check interaction, or get dose recommendation)
- **Observe:** Tool result is read and incorporated into the conversation state
- **Repeat:** Until a complete dose recommendation is produced and explained

A patient-facing example session:

- Patient: "Hi, my ID is P045."
- Agent calls get_patient_history(P045), loads demographics, last INR, last dose
- Agent: "Good to see you. Your last INR was 2.6 and your current dose is 5 mg/day. How did this week go, any missed doses?"
- Patient: "I missed one on Tuesday. And I started a new antibiotic."
- Agent calls check_drug_food_interaction("antibiotic"), reviews severity
- Agent: "Which antibiotic is it? Some can raise your INR significantly."
- Patient: "Fluconazole."
- Agent calls check_drug_food_interaction("fluconazole"), confirms CRITICAL potentiator
- Agent calls get_dose_recommendation with all assembled fields
- Agent: "Your recommended dose is actually being flagged as a hold today. Fluconazole significantly raises your warfarin effect, and combined with your current INR, the system is recommending you pause your dose and contact the clinic today. Please confirm with your nurse before making any changes."

Usage:

- New patient: python3 intake_agent.py
- Existing cohort patient: python3 intake_agent.py P045

### Course concepts integrated

#### Concept 1: Model and Provider Selection, Cost / Latency / Quality Trade-offs (Week 2)

This concept was applied twice in the prototype. First, three ML architectures were evaluated on the same synthetic cohort. XGBoost was selected because it achieved the lowest RMSE (0.128 mg), the highest clinical tolerance (100% of predictions within +/-1 mg), and the fastest inference (~5 ms) without requiring GPU infrastructure. LSTM was eliminated on latency grounds, 150 ms is too slow for interactive clinical use.

Second, Claude claude-sonnet-4-6 was selected over Claude Opus for the conversational intake layer. Sonnet is sufficient because the intake task is structured extraction and tool calling, not open-ended clinical reasoning. The API already defines the exact schema of what needs to be collected. Using Opus would add latency and cost with no meaningful accuracy gain for this specific task profile. Both selections follow the same Week 2 framework: match the model to the quality, latency, and cost requirements of the task.

#### Concept 2: Tool Use / Function Calling and the ReAct Agent Pattern (Week 4)

The five REST API endpoints were designed from the start as callable tools with defined input schemas (Pydantic v2), typed JSON outputs, and scoped responsibilities. In Component 5, three of those endpoints are formally registered as tools in the Anthropic tool use protocol, each with a name, description, and JSON Schema input definition. Claude decides when to call each tool based on conversation state, not a fixed script. This is direct implementation of the Week 4 tool call protocol and the ReAct pattern: the LLM reasons over natural language, acts by calling typed structured tools, observes the results, and continues reasoning until the task is complete.

The agent skill file (warfarin_skill.md) packages the full system as a reusable skill that can be invoked from Claude Code or any compatible agent harness, matching the professor's suggestion exactly.

### Baseline

The intended comparison is against manual dose adjustment using the Warfarin Dosing Service (WDS) nomogram. That comparison has not been run yet. It is included as a planned next step, not a completed one. The current prototype establishes the AI side of that comparison only.

---

## 5. Evaluation Plan

### What was actually tested

A 10-case automated test suite (test_api.py) was run against the live API using the httpx library. All 10 tests passed.

1. **GET /health,** confirmed API is running and model is loaded
2. **GET /model-info,** confirmed training record count, feature count, and model type
3. **POST /predict (stable patient),** INR 2.8, dose 30 mg/week; returned Maintain, High confidence, no alerts
4. **POST /predict (critical INR + antifungal),** INR 5.8 + fluconazole; returned CRITICAL alert, Hold action
5. **POST /predict (poor metabolizer),** CYP2C9 \*3/\*3 + VKORC1 AA; returned low dose, Moderate confidence
6. **POST /predict (enzyme inducer),** rifampin; returned significant dose increase, drug interaction flag
7. **POST /interaction-check (fluconazole),** confirmed HIGH severity potentiator entry returned
8. **POST /interaction-check (rifampin),** confirmed HIGH severity reducer entry returned
9. **GET /patient/P042,** confirmed last 5 weeks of visit history returned
10. **422 validation error,** confirmed Pydantic rejects malformed input with field-level errors

The intake agent (intake_agent.py) was reviewed for syntax and structural correctness. A live end-to-end session requires the API server running and an ANTHROPIC_API_KEY; that test is conducted interactively.

### ML model performance on held-out test set

The XGBoost model was evaluated on the 1,000-row held-out set (20% of the synthetic cohort). These metrics reflect performance against the synthetic data generator, not real clinical outcomes.

- RMSE: 0.128 mg/week
- R-squared: 0.998
- % predictions within +/-1 mg of ground truth: 100%
- % predictions within +/-2 mg of ground truth: 100%

All metrics are measured against synthetic ground truth only. The nomogram comparison has not been conducted.

---

## 6. Example Inputs and Failure Cases

### Test cases run through the API

**Case 1, Stable, in-range patient**
- **Input:** INR 2.8, dose 30 mg/week, CYP2C9 \*1/\*1, VKORC1 GG, no interactions
- **Result:** Maintain, High confidence, no safety alerts

**Case 2, Critical high INR with antifungal (fluconazole)**
- **Input:** INR 5.8, dose 32.5 mg/week, concurrent fluconazole
- **Result:** Hold, CRITICAL alerts fired (INR_CRITICAL_HIGH + DRUG_INTERACTION_POTENTIATOR)

**Case 3, Poor metabolizer**
- **Input:** CYP2C9 \*3/\*3, VKORC1 AA (high genetic sensitivity)
- **Result:** Low dose recommendation, Moderate confidence, genotypic sensitivity flagged

**Case 4, Enzyme inducer (rifampin)**
- **Input:** INR 1.8, dose 35 mg/week, concurrent rifampin
- **Result:** Significant dose increase recommended, DRUG_INTERACTION_REDUCER flagged

**Case 5, Invalid input (validation test)**
- **Input:** Malformed request body missing required fields
- **Result:** 422 Unprocessable Entity returned with field-level Pydantic error detail

### Anticipated failure cases

**Failure 1, Multiple simultaneous drug interactions**

The engine applies drug effects additively. A patient on both fluconazole and amiodarone may have a combined INR impact larger than the sum of individual effects. The prototype may underestimate this and miss a CRITICAL threshold.

**Failure 2, Drug not in the 23-entry knowledge base**

Any medication not catalogued in interactions.py returns no interaction flag, a silent false negative. The prototype currently has no mechanism to warn the clinician when a submitted drug name is unrecognized.

**Failure 3, LLM misextraction from ambiguous patient language**

If a patient describes a medication vaguely ("a blood thinner my doctor added") Claude may call check_drug_food_interaction with an incomplete query, miss the match, and pass medication_inr_effect=None to the prediction engine. The safety guardrail engine does not catch extraction errors, only clinical threshold violations.

**Failure 4, Synthetic data distribution vs. real patients**

The model was trained on data generated from pharmacological equations. Real patient INR trajectories include noise from adherence variability, dietary inconsistency, and measurement error that the synthetic generator does not model. Prototype performance on real data is unknown.

---

## 7. Risks and Governance

### Where the system could fail

- **Synthetic training data:** The model has never seen real patient data. Its confidence on real-world inputs is unknown and likely overstated.
- **Interaction database coverage:** Only 23 drug/food entries. Any agent not in the list passes through without a flag.
- **LLM extraction errors:** Claude may misinterpret ambiguous patient language and pass incorrect field values to the prediction engine. There is no extraction validation layer between the LLM and the API.
- **Prompt injection via patient input:** A patient could theoretically phrase their input in a way that attempts to manipulate Claude's behavior. The system prompt and tool schema provide a degree of constraint, but this risk is not formally mitigated in the prototype.
- **No real baseline comparison:** The prototype cannot yet demonstrate improvement over manual dose adjustment.
- **Input quality:** Data entry errors propagate directly into recommendations with no detection.

### Where the system should not be trusted

- Any clinical decision with a real patient, this is a first prototype on synthetic data only
- Pediatric patients (model trained on adult pharmacokinetics)
- Pregnant patients (warfarin is contraindicated in pregnancy)
- Patients with mechanical aortic valves (different INR target: 2.0-3.0)
- Patients with severe hepatic or renal impairment
- Any drug not in the 23-entry knowledge base

### Controls in the current prototype

- **CRITICAL alert classification:** INR > 5.0 and INR < 1.5 are flagged as CRITICAL in the API response.
- **Dose-change threshold flag:** Recommendations that change the prior dose by more than 30% trigger DOSE_LARGE_CHANGE in the alert list.
- **Clinician-in-the-loop requirement:** The agent always closes with a reminder to confirm any dose change with the anticoagulation nurse or physician. The system is a copilot, not an autonomous prescriber.
- **No write access:** The API is read-only and runs locally. It does not connect to any EHR or patient record system.

### Data and privacy

The prototype uses only synthetic data. No real patient health information (PHI) was used at any stage. In a future real-world deployment, all patient inputs would constitute protected health information and require HIPAA-compliant infrastructure. Patient conversations with the intake agent would additionally require specific consent disclosure under HIPAA guidelines.

---

## 8. Plan for the Week 6 Check-in

### What is running now (as of May 2026)

- **warfarin_cohort_MMV.xlsx,** 5,000-row synthetic cohort generated and saved to the project folder
- **warfarin_ml_pipeline.py,** RF / XGBoost / LSTM compared; results saved to model_comparison_results.csv
- **main.py,** FastAPI running at http://127.0.0.1:8000, all 5 endpoints live, Swagger docs at /docs
- **interactions.py,** 23-entry drug/food interaction database with fuzzy matching
- **test_api.py,** 10-case automated test suite; all tests passed
- **intake_agent.py,** Claude claude-sonnet-4-6 conversational intake agent with ReAct tool-use loop; syntax validated, ready for live session with API key and running server
- **warfarin_skill.md,** agent skill definition for use in Claude Code or compatible agent harness

### What evaluation is in place

- ML performance metrics on 1,000-row held-out set: RMSE 0.128 mg, R2 0.998, 100% within +/-1 mg
- 10 live API test cases covering stable patient, critical INR + drug interaction, poor metabolizer, enzyme inducer, invalid input
- Safety alert firing verified for CRITICAL scenarios (fluconazole + high INR, rifampin)
- Input validation verified via 422 error test
- Intake agent syntax validation passed; end-to-end conversational session requires live API and ANTHROPIC_API_KEY

### What is still needed

- Live end-to-end intake agent session testing with real ANTHROPIC_API_KEY and running FastAPI server
- Formal baseline comparison against WDS nomogram on matched patient scenarios, not yet run
- Unrecognized drug name warning in /predict response when query misses the interaction database
- Explicit prototype disclaimer in all /predict responses
- Expanded test set covering edge cases: polypharmacy, renal impairment, missed doses
