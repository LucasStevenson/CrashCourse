from fastapi import FastAPI, UploadFile, File, Body, Form
from pydantic import BaseModel
import numpy as np, cv2
import json
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import YoloDetector, estimate_lead_distance_px
from rules import ScoringState, Telemetry

app=FastAPI()

# Use absolute path for model file
model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
if not os.path.exists(model_path):
    # If not found, try downloading or use default
    model_path = "yolov8n.pt"  # YOLO will auto-download if needed

det=YoloDetector(model_path, conf=0.25, imgsz=640)
scorer=ScoringState()

class TelemetryIn(BaseModel):
    t: float; speed_mps: float; speed_limit_mps: float
    throttle: float; brake: float; steer_deg: float
    lane_offset_m: float|None=None; tl_state: str|None=None
    in_stop_zone: bool|None=None; collision: bool=False

def px_to_ttc(px_proxy: float|None, speed_mps: float)->float|None:
    if px_proxy is None or speed_mps<0.1: return None
    return (40.0*px_proxy)/max(speed_mps,0.1)

# Heuristic distance conversion: px_proxy -> meters
# px_proxy comes from estimate_lead_distance_px: 1 / bbox_height_pixels
# Use same scale as TTC helper (40.0) so dist_m ~= 40 * (1/box_h)
def px_to_dist_m(px_proxy: float|None) -> float|None:
    if px_proxy is None: return None
    return 40.0 * px_proxy

@app.post("/infer_frame")
async def infer_frame(
    image: UploadFile = File(...),
    telemetry: str = Form(...),   # <-- accept as string from multipart
):
    # Parse telemetry JSON string into the Pydantic model
    telemetry_obj = TelemetryIn.model_validate_json(telemetry)

    data = await image.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    dets = det.infer(bgr)
    lead_proxy = estimate_lead_distance_px(dets, bgr.shape)
    ttc = px_to_ttc(lead_proxy, telemetry_obj.speed_mps)
    lead_dist_m = px_to_dist_m(lead_proxy)

    # Simple collision heuristic: very close or extremely low TTC
    COLLISION_DIST_M = 0.6   # tune as needed
    TTC_COLLISION_S  = 0.25  # seconds
    collided = False
    if lead_dist_m is not None and lead_dist_m < COLLISION_DIST_M:
        collided = True
    if ttc is not None and ttc < TTC_COLLISION_S:
        collided = True

    tel = Telemetry(**telemetry_obj.model_dump())
    if collided:
        tel.collision = True

    cues = scorer.step(tel, ttc)
    return {
        "cues": cues,
        "ttc": ttc,
        "lead_distance_m": lead_dist_m,
        "collision": tel.collision,
        "detections": len(dets),
    }

@app.post("/end_session")
async def end_session():
    return scorer.finalize()
