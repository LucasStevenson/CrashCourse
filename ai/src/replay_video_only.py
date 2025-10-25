# replay_video_only.py
import argparse, cv2, time
from .detector import YoloDetector
from .rules import ScoringState, Telemetry
from .video_only import FlowSpeedEstimator, LeadTTC, classify_traffic_light_color, pick_lead_vehicle
from .lane_simple import estimate_lane_offset_m

parser = argparse.ArgumentParser()
parser.add_argument("--video", default="data/sample_drive.mp4", help="path or 0 for webcam")
parser.add_argument("--limit_mph", type=float, default=30.0, help="assumed speed limit for demo")
parser.add_argument("--scale_k", type=float, default=2.5, help="optical flow scale to m/s")
args = parser.parse_args()
VIDEO_PATH = 0 if args.video == "0" else args.video
SPEED_LIMIT_MPS = args.limit_mph * 0.44704

det = YoloDetector("yolov8n.pt", conf=0.25, imgsz=640)
scorer = ScoringState()
flow_speed = FlowSpeedEstimator(scale_k=args.scale_k)
lead_ttc = LeadTTC()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO_PATH}")

t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok: break

    # Downscale lightly for speed
    h,w = frame.shape[:2]
    scale = max(w,h)/720
    if scale > 1.0:
        frame = cv2.resize(frame, (int(w/scale), int(h/scale)))

    t = time.time() - t0

    # Perception
    dets = det.infer(frame)
    lead_box = pick_lead_vehicle(dets, frame.shape)

    # Derived signals from video
    speed_mps = flow_speed.step(frame)                         # relative m/s
    lane_off_m, lane_dbg = estimate_lane_offset_m(frame)       # may be None
    tl_crop = None
    for d in dets:
        if d["cls_name"] == "traffic light":
            tl_crop = frame[int(d["xyxy"][1]):int(d["xyxy"][3]), int(d["xyxy"][0]):int(d["xyxy"][2])]
            break
    tl_state = classify_traffic_light_color(tl_crop)           # 'red'|'green'|None
    ttc = lead_ttc.step(lead_box, t)                           # seconds, or None

    # Build Telemetry from estimates
    tel = Telemetry(
        t=t,
        speed_mps=speed_mps,
        speed_limit_mps=SPEED_LIMIT_MPS,
        throttle=0.0,                 # unknown from video; keep 0
        brake=0.0,                    # unknown; rules still handle TTC/lanes
        steer_deg=0.0,                # unknown
        lane_offset_m=lane_off_m,
        tl_state=tl_state,            # None if unknown
        in_stop_zone=False,           # we don't know stop zone without a map
        collision=False
    )

    # Update scoring & get sticky cues
    scorer.step(tel, ttc)
    display_cues = scorer.get_display_cues()

    # ---- HUD overlays (debug) ----
    for d in dets:
        x1,y1,x2,y2 = map(int, d["xyxy"])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, f"{d['cls_name']} {d['conf']:.2f}", (x1,max(12,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),1)
    if lead_box:
        x1,y1,x2,y2 = map(int, lead_box)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)

    spd_txt = f"spd≈{speed_mps*2.236:.1f}mph (video) lim={SPEED_LIMIT_MPS*2.236:.0f}"
    lane_txt = f"lane={lane_off_m:+.2f}m" if lane_off_m is not None else "lane=NA"
    ttc_txt = f"TTC={ttc:.2f}s" if ttc is not None else "TTC=NA"
    light_txt = f"light={tl_state or 'NA'}"
    cv2.putText(frame, f"{spd_txt}  {lane_txt}  {ttc_txt}  {light_txt}", (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,(50,200,255),2)

    y=46
    for cue in display_cues:
        cv2.putText(frame, f"CUE: {cue['cue']} {cue['level']:.2f}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,0,255),2)
        y+=24

    cv2.imshow("Video-only Scoring (q to quit)", frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()
print("\n=== SCORECARD (video-only) ===")
print(scorer.finalize())
