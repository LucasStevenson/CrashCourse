# video_only.py
import cv2, numpy as np, time
from typing import Optional, Tuple, Dict, Any, List
from .lane_simple import estimate_lane_offset_m

class FlowSpeedEstimator:
    """Relative speed from optical flow magnitude over road ROI; scale_k maps mag->m/s."""
    def __init__(self, scale_k: float = 2.5):
        self.prev = None
        self.scale_k = scale_k

    def step(self, bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi = gray[int(0.55*h):int(0.95*h), int(0.15*w):int(0.85*w)]
        if self.prev is None:
            self.prev = roi
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(self.prev, roi, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        self.prev = roi
        mag = np.linalg.norm(flow, axis=2).mean()
        return float(self.scale_k * mag)  # m/s (rough, calibrate scale_k with one known segment)

class LeadTTC:
    """TTC via looming: bbox height h(t); TTC ≈ h / (dh/dt)."""
    def __init__(self):
        self.prev_h = None
        self.prev_t = None

    def step(self, lead_box: Optional[List[float]], t: float) -> Optional[float]:
        if lead_box is None: 
            self.prev_h = None; self.prev_t = None
            return None
        x1, y1, x2, y2 = lead_box
        h = max(1.0, y2 - y1)
        if self.prev_h is None or self.prev_t is None:
            self.prev_h, self.prev_t = h, t
            return None
        dt = max(1e-3, t - self.prev_t)
        dh = h - self.prev_h
        self.prev_h, self.prev_t = h, t
        if dh <= 0: 
            return None  # not approaching
        ttc = h / (dh/dt)
        return float(ttc)

def classify_traffic_light_color(bgr_crop: np.ndarray) -> Optional[str]:
    """Very simple HSV threshold: returns 'red','green', or None."""
    if bgr_crop is None or bgr_crop.size == 0: return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    # masks
    red1 = cv2.inRange(hsv, (0,80,80), (10,255,255))
    red2 = cv2.inRange(hsv, (160,80,80), (179,255,255))
    red = cv2.countNonZero(red1|red2)
    green = cv2.countNonZero(cv2.inRange(hsv, (35,60,60), (90,255,255)))
    if red==0 and green==0: return None
    return "red" if red >= green*1.2 else ("green" if green >= red*1.2 else None)

def pick_lead_vehicle(dets: List[Dict[str,Any]], frame_shape) -> Optional[List[float]]:
    """Choose largest forward vehicle bbox."""
    h, w = frame_shape[:2]; cx = w/2
    best = None; best_h = -1
    for d in dets:
        cls = d["cls_name"]
        if cls not in ("car","truck","bus","motorcycle","bicycle"): continue
        x1,y1,x2,y2 = d["xyxy"]
        box_h = y2-y1
        if abs((x1+x2)/2 - cx) < 0.22*w and box_h > best_h:
            best_h = box_h; best = [x1,y1,x2,y2]
    return best

def crop_bbox(bgr, box):
    if box is None: return None
    x1,y1,x2,y2 = map(int, box)
    h,w = bgr.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    if x2<=x1 or y2<=y1: return None
    return bgr[y1:y2, x1:x2]
