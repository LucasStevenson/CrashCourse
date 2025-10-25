from fastapi import FastAPI, UploadFile, File, Body, Form
from pydantic import BaseModel
import numpy as np, cv2
from .detector import YoloDetector, estimate_lead_distance_px
from .rules import ScoringState, Telemetry
import json

app=FastAPI()
det=YoloDetector("yolov8n.pt", conf=0.25, imgsz=640)
scorer=ScoringState()

class TelemetryIn(BaseModel):
    t: float; speed_mps: float; speed_limit_mps: float
    throttle: float; brake: float; steer_deg: float
    lane_offset_m: float|None=None; tl_state: str|None=None
    in_stop_zone: bool|None=None; collision: bool=False

def px_to_ttc(px_proxy: float|None, speed_mps: float)->float|None:
    if px_proxy is None or speed_mps<0.1: return None
    return (40.0*px_proxy)/max(speed_mps,0.1)

# New function to process image and telemetry, returning cues
def process_image_and_telemetry(image_bytes: bytes, telemetry_json: str):
    telemetry_obj = TelemetryIn.model_validate_json(telemetry_json)
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    dets = det.infer(bgr)
    lead_proxy = estimate_lead_distance_px(dets, bgr.shape)
    ttc = px_to_ttc(lead_proxy, telemetry_obj.speed_mps)

    cues = scorer.step(Telemetry(**telemetry_obj.model_dump()), ttc)
    return {"cues": cues, "ttc": ttc, "detections": len(dets)}

@app.post("/infer_frame")
async def infer_frame(
    image: UploadFile = File(...),
    telemetry: str = Form(...),   # <-- accept as string from multipart
):
    # This endpoint can now call the new function
    image_data = await image.read()
    return process_image_and_telemetry(image_data, telemetry)

@app.post("/end_session")
async def end_session():
    return scorer.finalize()
