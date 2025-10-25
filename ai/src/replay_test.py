import cv2, time, math, numpy as np
from .detector import YoloDetector, estimate_lead_distance_px
from .rules import ScoringState, Telemetry

VIDEO_PATH = "src/sample_drive.mp4"
IMG_SIZE = 640
FPS_INFER = 12
SPEED_LIMIT_MPS = 13.4  # ~30 mph

def synthetic_telemetry(t: float) -> Telemetry:
    speed = 13 + 5*math.sin(t*0.2)
    brake = max(0.0, 0.4*math.sin(t*1.3))
    throttle = max(0.0, 0.6*math.cos(t*0.7))
    steer = 5*math.sin(t*0.5)
    lane = 0.25*math.sin(t*0.15)
    tl = "red" if 20.0 <= t%60.0 <= 25.0 else "green"
    in_stop = 19.0 <= t%60.0 <= 26.0
    return Telemetry(t=t, speed_mps=max(0.0,speed), speed_limit_mps=SPEED_LIMIT_MPS,
                     throttle=throttle, brake=brake, steer_deg=steer,
                     lane_offset_m=lane, tl_state=tl, in_stop_zone=in_stop, collision=False)

def px_to_ttc(px_proxy: float|None, speed_mps: float) -> float|None:
    if px_proxy is None or speed_mps<0.1: return None
    k=40.0; dist=k*px_proxy; return dist/max(speed_mps,0.1)

def main():
    cap=cv2.VideoCapture(VIDEO_PATH); 
    if not cap.isOpened(): raise SystemExit(f"Cannot open {VIDEO_PATH}")
    det=YoloDetector("yolov8n.pt", conf=0.25, imgsz=IMG_SIZE)
    scorer=ScoringState()
    frame_period=1.0/FPS_INFER; next_tick=time.time(); t0=time.time()

    while True:
        now=time.time()
        if now<next_tick: time.sleep(max(0.0, next_tick-now))
        next_tick+=frame_period
        ok, frame=cap.read()
        if not ok: break
        h,w=frame.shape[:2]; scale=max(w,h)/720
        if scale>1.0: frame=cv2.resize(frame,(int(w/scale),int(h/scale)))

        dets=det.infer(frame)
        lead_proxy=estimate_lead_distance_px(dets, frame.shape)

        t=time.time()-t0
        tel=synthetic_telemetry(t)
        ttc=px_to_ttc(lead_proxy, tel.speed_mps)

        cues=scorer.step(tel, ttc)

        for d in dets:
            x1,y1,x2,y2=map(int,d["xyxy"])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{d['cls_name']} {d['conf']:.2f}",(x1,max(12,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

        hud=f"spd={tel.speed_mps*2.236:.1f}mph lim={tel.speed_limit_mps*2.236:.0f}  lane={tel.lane_offset_m:+.2f}m  TTC={'{:.2f}s'.format(ttc) if ttc else 'NA'}"
        cv2.putText(frame,hud,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(50,200,255),2)
        y=45
        for cue in cues[:2]:
            cv2.putText(frame,f"CUE: {cue['cue']} {cue['level']:.2f}",(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2); y+=22
        cv2.imshow("Scoring Replay (q to quit)", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    print("\n=== SCORECARD ===")
    print(scorer.finalize())

if __name__=="__main__": main()
