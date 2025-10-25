from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO

COCO = {"person":0,"bicycle":1,"car":2,"motorcycle":3,"bus":5,"truck":7,"traffic light":10,"stop sign":13}
INTEREST = {COCO["person"],COCO["car"],COCO["bicycle"],COCO["motorcycle"],COCO["bus"],COCO["truck"],COCO["traffic light"],COCO["stop sign"]}

class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, bgr_frame: np.ndarray) -> List[Dict[str, Any]]:
        res = self.model.predict(bgr_frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out: List[Dict[str, Any]] = []
        if res.boxes is None or res.boxes.xyxy is None:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        names = self.model.names
        for (x1,y1,x2,y2), c, p in zip(boxes, clss, confs):
            if c not in INTEREST: continue
            out.append({"cls_id":int(c),"cls_name":names[int(c)],"conf":float(p),"xyxy":[float(x1),float(y1),float(x2),float(y2)],
                        "center":[float((x1+x2)/2), float((y1+y2)/2)]})
        return out

def estimate_lead_distance_px(det: List[Dict[str, Any]], frame_shape: Tuple[int,int,int]) -> Optional[float]:
    h, w = frame_shape[:2]; cx = w/2; best=None
    for d in det:
        if d["cls_id"] not in [COCO["car"],COCO["bus"],COCO["truck"],COCO["motorcycle"],COCO["bicycle"]]: continue
        x1,y1,x2,y2 = d["xyxy"]; box_h = (y2-y1); center_x = (x1+x2)/2
        if abs(center_x-cx) < w*0.22:
            if best is None or box_h>best[0]: best=(box_h,d)
    if best is None: return None
    return 1.0/max(best[0],1.0)  # bigger box -> closer
