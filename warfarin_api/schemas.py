from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum

class CYP2C9(str, Enum):
    wt="*1/*1"; im1="*1/*2"; im2="*1/*3"; pm1="*2/*2"; pm2="*2/*3"; pm3="*3/*3"
class VKORC1(str, Enum):
    low="GG"; mid="AG"; high="AA"
class VitKLevel(str, Enum):
    low="Low"; normal="Normal"; high="High"
class IllnessLevel(str, Enum):
    none="None"; mild="Mild"; moderate="Moderate"; severe="Severe"
class ExerciseLevel(str, Enum):
    sedentary="Sedentary"; moderate="Moderate"; active="Active"
class MedEffect(str, Enum):
    none="None"; increase="Increase"; decrease="Decrease"
class AlertSeverity(str, Enum):
    info="INFO"; warning="WARNING"; critical="CRITICAL"

TABULAR_FEATURES = [
    "INR_Result","INR_lag1","INR_lag2","INR_delta",
    "Current_Warfarin_Dose_mg_day","Dose_lag1",
    "VitK_enc","Alcohol_Units_Per_Week","Cranberry_enc","Grapefruit_enc",
    "Garlic_enc","MedEffect_enc","DrugCat_enc",
    "Missed_Doses_This_Week","Illness_enc","Diarrhea_enc","Exercise_enc",
    "Age","Sex_enc","Weight_kg","BMI","CYP2C9_enc","VKORC1_enc",
    "has_Atrial_Fibrillation","has_Heart_Failure","has_Hypertension",
    "has_Diabetes_Mellitus","has_CKD","has_Hypothyroidism","has_Hyperthyroidism",
    "has_Liver_Disease","has_COPD","has_Coronary_Artery_Disease",
]

class PredictRequest(BaseModel):
    patient_id: Optional[str]=None
    age: int=Field(...,ge=18,le=100)
    sex: str=Field(...,pattern="^[MF]$")
    weight_kg: float=Field(...,ge=30,le=200)
    bmi: float=Field(...,ge=10,le=70)
    cyp2c9: CYP2C9
    vkorc1: VKORC1
    has_atrial_fibrillation: bool=False
    has_heart_failure: bool=False
    has_hypertension: bool=False
    has_diabetes: bool=False
    has_ckd: bool=False
    has_hypothyroidism: bool=False
    has_hyperthyroidism: bool=False
    has_liver_disease: bool=False
    has_copd: bool=False
    has_cad: bool=False
    inr_result: float=Field(...,ge=0.5,le=10.0)
    current_dose_mg: float=Field(...,ge=0.5,le=20.0)
    inr_lag1: Optional[float]=Field(None,ge=0.5,le=10.0)
    inr_lag2: Optional[float]=Field(None,ge=0.5,le=10.0)
    dose_lag1: Optional[float]=Field(None,ge=0.5,le=20.0)
    vitk_level: VitKLevel=VitKLevel.normal
    alcohol_units_per_week: float=Field(0.0,ge=0,le=50)
    cranberry_juice: bool=False
    grapefruit: bool=False
    garlic_supplement: bool=False
    medication_inr_effect: MedEffect=MedEffect.none
    medication_category: Optional[str]="None"
    missed_doses_this_week: int=Field(0,ge=0,le=7)
    illness_event: IllnessLevel=IllnessLevel.none
    diarrhea_vomiting: bool=False
    exercise_level: ExerciseLevel=ExerciseLevel.moderate

class ClinicalAlert(BaseModel):
    severity: AlertSeverity
    code: str
    message: str
    recommended_action: str

class PredictResponse(BaseModel):
    patient_id: Optional[str]
    recommended_dose_mg: float
    dose_change_mg: float
    dose_change_pct: float
    dose_action: str
    predicted_inr_direction: str
    confidence: str
    model_used: str
    clinical_alerts: List[ClinicalAlert]
    clinical_notes: List[str]

class InteractionRequest(BaseModel):
    query: str=Field(...,min_length=2)

class InteractionMatch(BaseModel):
    name: str; category: str; effect_on_inr: str; magnitude: str
    mechanism: str; clinical_action: str; severity: AlertSeverity

class InteractionResponse(BaseModel):
    query: str; matches: List[InteractionMatch]; summary: str

class WeekRecord(BaseModel):
    week: int; date: str; inr: float; status: str; dose_mg: float
    dose_action: str; medication: str; vitk_level: str
    missed_doses: int; bleeding_event: str; clot_event: str

class PatientHistoryResponse(BaseModel):
    patient_id: str; age: int; sex: str; cyp2c9: str; vkorc1: str
    comorbidities: str; baseline_dose_mg: float
    target_inr_min: float; target_inr_max: float; weeks_observed: int
    mean_inr: float; pct_time_therapeutic: float
    pct_time_subtherapeutic: float; pct_time_supratherapeutic: float
    total_dose_adjustments: int; bleeding_events: int
    thromboembolic_events: int; weekly_records: List[WeekRecord]

class HealthResponse(BaseModel):
    status: str; model_loaded: bool; dataset_loaded: bool
    uptime_seconds: float; version: str

class ModelInfoResponse(BaseModel):
    model_type: str; version: str; training_patients: int
    training_weeks: int; target_variable: str; features: List[str]
    performance: dict; inr_target_range: dict; safety_guardrails: List[str]
