import os
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import libsql_client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

def analyze_telemetry(sensor_type: str, model_type: str, history: list, metrics: dict) -> tuple[bool, str]:
    # --- 1. HEART RATE ---
    if sensor_type == "heart_rate":
        hr = metrics["bpm"]
        if model_type == "threshold":
            if hr > 120: return True, f"Tachycardia Alert ({hr} BPM > 120)"
            if hr < 50: return True, f"Bradycardia Alert ({hr} BPM < 50)"
        elif model_type == "zscore" and len(history) >= 5:
            vals = [h["bpm"] for h in history]
            mean, std = np.mean(vals), np.std(vals)
            if std > 0 and abs(hr - mean) / std > 2.5:
                return True, f"Sudden HR Spike/Drop (Z-Score > 2.5 from avg {int(mean)})"
        return False, f"Normal ({hr} BPM)"

    # --- 2. BLOOD SUGAR (CGM) ---
    elif sensor_type == "blood_sugar":
        glucose = metrics["mg_dl"]
        if model_type == "threshold":
            if glucose < 70: return True, f"Hypoglycemia Alert ({glucose} mg/dL < 70)"
            if glucose > 180: return True, f"Hyperglycemia Alert ({glucose} mg/dL > 180)"
        elif model_type == "ema_drift" and len(history) >= 4:
            # Exponential Moving Average drift check for rapid drops/spikes
            recent = [h["mg_dl"] for h in history[-4:]]
            rate_of_change = (glucose - recent[0]) / len(recent)
            if rate_of_change < -8:
                return True, f"Rapid Glucose Drop ({rate_of_change:.1f} mg/dL per tick)"
            if rate_of_change > 10:
                return True, f"Rapid Glucose Surge (+{rate_of_change:.1f} mg/dL per tick)"
        return False, f"Stable Glucose ({glucose} mg/dL)"

    # --- 3. BRAIN WAVES (EEG) ---
    elif sensor_type == "brain_waves":
        alpha = metrics["alpha_hz"] # Relaxed/focused awareness
        beta = metrics["beta_hz"]   # Active stress/high workload
        theta = metrics["theta_hz"] # Drowsiness/micro-sleep
        
        if model_type == "cognitive_load":
            # High Beta relative to Alpha indicates extreme stress/overload
            ratio = beta / max(alpha, 0.1)
            if ratio > 2.5:
                return True, f"Cognitive Overload Alert (Beta/Alpha Ratio: {ratio:.2f})"
            if theta > 8.0 and beta < 12.0:
                return True, f"Micro-Sleep / Fatigue Detected (High Theta: {theta} Hz)"
        elif model_type == "isolation_forest" and len(history) >= 8:
            X = np.array([[h["alpha_hz"], h["beta_hz"], h["theta_hz"]] for h in history])
            clf = IsolationForest(contamination=0.15, random_state=42).fit(X)
            pred = clf.predict([[alpha, beta, theta]])[0]
            if pred == -1:
                return True, f"Abnormal EEG Pattern (Isolation Forest Outlier)"
        return False, f"Nominal Brain Activity (α:{alpha} β:{beta} θ:{theta} Hz)"

    # --- 4. BLOOD OXYGEN (SpO2) & CABIN CO2 ---
    elif sensor_type == "spo2_co2":
        spo2 = metrics["spo2_pct"]
        co2 = metrics["cabin_co2_ppm"]
        
        if model_type == "hypoxia_env":
            if spo2 < 92 and co2 > 4000:
                return True, f"CRITICAL: Cabin Hypercapnia ({co2} ppm) + Hypoxia ({spo2}%)"
            if spo2 < 92:
                return True, f"Hypoxia Alert: Low Blood Oxygen ({spo2}% < 92%)"
            if co2 > 5000:
                return True, f"Cabin CO2 Scrubber Warning ({co2} ppm > 5000)"
        elif model_type == "isolation_forest" and len(history) >= 8:
            X = np.array([[h["spo2_pct"], h["cabin_co2_ppm"]] for h in history])
            clf = IsolationForest(contamination=0.15, random_state=42).fit(X)
            pred = clf.predict([[spo2, co2]])[0]
            if pred == -1:
                return True, f"Multivariate SpO2/CO2 Anomaly Detected"
        return False, f"Nominal O2/CO2 (SpO2: {spo2}%, CO2: {co2} ppm)"

    return False, "Normal"

@app.websocket("/ws/{session_id}/{sensor_type}/{model_type}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, sensor_type: str, model_type: str):
    await websocket.accept()
    history = []
    
    http_url = TURSO_URL.replace("libsql://", "https://")
    client = libsql_client.create_client(url=http_url, auth_token=TURSO_TOKEN)

    try:
        while True:
            data = await websocket.receive_text()
            metrics = json.loads(data)

            # 1. Run AI Analysis
            is_anomaly, message = analyze_telemetry(sensor_type, model_type, history, metrics)
            history.append(metrics)
            if len(history) > 25:
                history.pop(0)

            # 2. Save to Turso
            await client.execute(
                "INSERT INTO vitals (session_id, sensor_type, metrics_json, is_anomaly) VALUES (?, ?, ?, ?)",
                [session_id, sensor_type, json.dumps(metrics), is_anomaly]
            )

            # 3. Send response back to frontend
            await websocket.send_json({
                "sensor_type": sensor_type,
                "metrics": metrics,
                "is_anomaly": is_anomaly,
                "alert_message": message
            })

    except WebSocketDisconnect:
        await client.execute("DELETE FROM vitals WHERE session_id = ?", [session_id])
        print(f"Wiped session {session_id}")
    finally:
        await client.close()
