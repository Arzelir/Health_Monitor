import os
import json
import numpy as np
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

# Built-in AI / Anomaly detection logic
def analyze_heart_rate(model_type: str, history: list, current_hr: int) -> tuple[bool, str]:
    if model_type == "threshold":
        if current_hr > 120:
            return True, "High Heart Rate Alert (>120 BPM)"
        if current_hr < 50:
            return True, "Low Heart Rate Alert (<50 BPM)"
            
    elif model_type == "zscore" and len(history) >= 5:
        mean = np.mean(history)
        std = np.std(history)
        if std > 0 and abs(current_hr - mean) / std > 2.5:
            return True, f"Sudden Spike/Drop Detected (Z-Score > 2.5 from avg {int(mean)})"
            
    return False, "Normal"

@app.websocket("/ws/{session_id}/{model_type}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, model_type: str):
    await websocket.accept()
    history = []
    
    # Connect to Turso using HTTP protocol for serverless compatibility
    http_url = TURSO_URL.replace("libsql://", "https://")
    client = libsql_client.create_client(url=http_url, auth_token=TURSO_TOKEN)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            hr = int(payload["heart_rate"])

            # 1. Run AI Analysis
            is_anomaly, message = analyze_heart_rate(model_type, history, hr)
            history.append(hr)
            if len(history) > 20:
                history.pop(0)

            # 2. Save to Turso
            await client.execute(
                "INSERT INTO vitals (session_id, heart_rate, is_anomaly) VALUES (?, ?, ?)",
                [session_id, hr, is_anomaly]
            )

            # 3. Send response/alert back to frontend
            await websocket.send_json({
                "heart_rate": hr,
                "is_anomaly": is_anomaly,
                "alert_message": message
            })

    except WebSocketDisconnect:
        # 4. Cleanup: Delete all session data when user closes tab or ends session
        await client.execute("DELETE FROM vitals WHERE session_id = ?", [session_id])
        print(f"Cleaned up database records for session: {session_id}")
    finally:
        await client.close()
