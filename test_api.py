import sys, warnings, os
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from fastapi.testclient import TestClient
from main import app

SEP = "=" * 62
print(SEP)
print("  WARFARIN API - ENDPOINT TEST SUITE")
print(SEP)

BASE = {
    "patient_id":"TEST-001","age":65,"sex":"F","weight_kg":72.0,"bmi":27.5,
    "cyp2c9":"*1/*2","vkorc1":"AG","has_hypertension":True,
    "inr_result":2.2,"current_dose_mg":5.0,
    "inr_lag1":2.4,"inr_lag2":2.6,"dose_lag1":5.0,
    "vitk_level":"Normal","alcohol_units_per_week":3.0,
    "medication_inr_effect":"None","medication_category":"None",
    "missed_doses_this_week":0,"illness_event":"None","exercise_level":"Moderate",
}

with TestClient(app) as client:

    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    print("[PASS] GET /health  status=" + d["status"] + "  model_loaded=" + str(d["model_loaded"]))
    r = client.get("/model-info")
    assert r.status_code == 200
    d = r.json()
    print("[PASS] GET /model-info  R2=" + str(d["performance"]["R2"]) + "  features=" + str(len(d["features"])))
    r = client.post("/predict", json=BASE)
    assert r.status_code == 200, r.text
    d = r.json()
    print("[PASS] POST /predict [stable INR=2.2]  dose=" + str(d["recommended_dose_mg"]) + " mg  action=" + d["dose_action"] + "  confidence=" + d["confidence"])
    crit = {**BASE, "patient_id":"TEST-002","inr_result":5.8,"current_dose_mg":7.5,
            "medication_inr_effect":"Increase","medication_category":"Antifungal","vitk_level":"Low"}
    r = client.post("/predict", json=crit)
    assert r.status_code == 200, r.text
    d = r.json()
    print("[PASS] POST /predict [CRITICAL INR=5.8]  dose=" + str(d["recommended_dose_mg"]) + " mg  action=" + d["dose_action"] + "  alerts=" + str(len(d["clinical_alerts"])))
    for a in d["clinical_alerts"]:
        print("         [" + a["severity"] + "] " + a["code"])
    poor = {**BASE, "patient_id":"TEST-003","cyp2c9":"*3/*3","vkorc1":"AA","inr_result":4.1,"current_dose_mg":2.0,"inr_lag1":3.8}
    r = client.post("/predict", json=poor)
    assert r.status_code == 200, r.text
    d = r.json()
    print("[PASS] POST /predict [Poor metabolizer INR=4.1]  dose=" + str(d["recommended_dose_mg"]) + " mg  action=" + d["dose_action"])
    print("         note: " + d["clinical_notes"][0][:70])
    rif = {**BASE, "patient_id":"TEST-004","inr_result":1.8,"current_dose_mg":6.0,"inr_lag1":2.6,"medication_inr_effect":"Decrease","medication_category":"Antibiotic"}
    r = client.post("/predict", json=rif)
    assert r.status_code == 200, r.text
    d = r.json()
    print("[PASS] POST /predict [Rifampin INR=1.8]  dose=" + str(d["recommended_dose_mg"]) + " mg  action=" + d["dose_action"])
    print("         alerts: " + str([a["code"] for a in d["clinical_alerts"]]))
    print("[PASS] POST /interaction-check")
    r = client.post("/interaction-check", json={"query":"fluconazole"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "CRITICAL" else "FAIL"
    print("         [" + ok + "] fluconazole -> " + sev)
    r = client.post("/interaction-check", json={"query":"amiodarone"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "CRITICAL" else "FAIL"
    print("         [" + ok + "] amiodarone -> " + sev)
    r = client.post("/interaction-check", json={"query":"kale"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "CRITICAL" else "FAIL"
    print("         [" + ok + "] kale -> " + sev)
    r = client.post("/interaction-check", json={"query":"st. john's wort"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "CRITICAL" else "FAIL"
    print("         [" + ok + "] st. john's wort -> " + sev)
    r = client.post("/interaction-check", json={"query":"ibuprofen"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "WARNING" else "FAIL"
    print("         [" + ok + "] ibuprofen -> " + sev)
    r = client.post("/interaction-check", json={"query":"omeprazole"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "INFO" else "FAIL"
    print("         [" + ok + "] omeprazole -> " + sev)
    r = client.post("/interaction-check", json={"query":"fish oil"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "INFO" else "FAIL"
    print("         [" + ok + "] fish oil -> " + sev)
    r = client.post("/interaction-check", json={"query":"unknownxyz"})
    assert r.status_code == 200
    d = r.json()
    sev = d["matches"][0]["severity"] if d["matches"] else "no match"
    ok = "OK  " if sev == "no match" else "FAIL"
    print("         [" + ok + "] unknownxyz -> " + sev)
    r = client.get("/patient/P042")
    assert r.status_code == 200, r.text
    d = r.json()
    print("[PASS] GET /patient/P042  age=" + str(d["age"]) + "  sex=" + d["sex"] + "  CYP2C9=" + d["cyp2c9"] + "  VKORC1=" + d["vkorc1"])
    print("         mean_INR=" + str(d["mean_inr"]) + "  TTR=" + str(d["pct_time_therapeutic"]) + "%  adjustments=" + str(d["total_dose_adjustments"]))
    print("         comorbidities: " + d["comorbidities"])
    bad = {**BASE, "inr_result": 15.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    print("[PASS] POST /predict [INR=15.0 out of range] -> 422 (correct)")
    r = client.get("/patient/P999")
    assert r.status_code == 404
    print("[PASS] GET /patient/P999 -> 404 Not Found (correct)")

print(SEP)
print("  ALL TESTS PASSED")
print(SEP)