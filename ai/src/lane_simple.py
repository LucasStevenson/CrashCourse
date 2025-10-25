# lane_simple.py
import cv2, numpy as np
from typing import Optional, Tuple

def _roi_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    pts = np.array([[
        (int(0.10*w), int(0.95*h)),
        (int(0.45*w), int(0.62*h)),
        (int(0.55*w), int(0.62*h)),
        (int(0.90*w), int(0.95*h))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(img, mask)

def _fit_line(points):
    if len(points) < 2: return None
    vx, vy, x0, y0 = cv2.fitLine(np.array(points, np.float32), cv2.DIST_L2,0,0.01,0.01)
    return float(vx), float(vy), float(x0), float(y0)

def _x_at_y(line, y):
    if line is None: return None
    vx, vy, x0, y0 = line
    if abs(vy) < 1e-6: return None
    t = (y - y0) / vy
    return x0 + vx * t

def estimate_lane_offset_m(bgr: np.ndarray, lane_width_m: float = 3.7):
    """Return (offset_m, dbg) where + is right of center; None if cannot estimate."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 150)
    edges = _roi_mask(edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=50)
    left_pts, right_pts = [], []
    cx = w/2

    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            if y2==y1: continue
            slope = (x2-x1)/(y2-y1)
            if abs(slope) < 0.2:  # reject near-horizontal
                continue
            x_bottom = x1 if y1>y2 else x2
            (left_pts if x_bottom < cx else right_pts).extend([(x1,y1),(x2,y2)])

    L = _fit_line(left_pts)  if left_pts  else None
    R = _fit_line(right_pts) if right_pts else None
    y_eval = int(h*0.90)
    xl = _x_at_y(L, y_eval) if L else None
    xr = _x_at_y(R, y_eval) if R else None

    dbg = {"xl": xl, "xr": xr, "y_eval": y_eval}
    if xl is None or xr is None or xr <= xl: return (None, dbg)

    lane_center_x = 0.5*(xl + xr)
    lane_width_px = xr - xl
    meters_per_px = lane_width_m / lane_width_px
    offset_m = (lane_center_x - cx) * meters_per_px
    return (float(offset_m), dbg)
