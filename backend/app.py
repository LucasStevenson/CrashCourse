import asyncio
import websockets
import json
import numpy as np
import cv2
import aiohttp
import os
import time
import uuid
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

# Store frames and telemetry for each connection
connections = {}

# Load environment variables from a .env file (e.g., backend/.env)
load_dotenv()

# Toolhouse agent config (set via environment)
TOOLHOUSE_URL = os.getenv("TOOLHOUSE_URL", "")
TOOLHOUSE_API_KEY = os.getenv("TOOLHOUSE_API_KEY", "")
# To conserve runs: throttle forwards and only send when cue changes
FORWARD_MIN_INTERVAL_S = float(os.getenv("TOOLHOUSE_MIN_INTERVAL_S", "1.0"))
# Max seconds to wait for final coach response before falling back locally
TOOLHOUSE_FINAL_TIMEOUT_S = float(os.getenv("TOOLHOUSE_FINAL_TIMEOUT_S", "20.0"))
# How to send payload to Toolhouse: 'wrapped' (default, uses 'message'), 'wrapped_input' (uses 'input'), or 'raw'
PAYLOAD_STYLE = os.getenv("TOOLHOUSE_PAYLOAD_STYLE", "wrapped").lower()


def _bucket(val, step):
    try:
        return None if val is None else round(float(val) / step) * step
    except Exception:
        return None


def _cue_fingerprint(obs):
    cue = obs.get("cue") or ""
    lvl = _bucket(obs.get("cue_level"), 0.2)
    return f"{cue}|{lvl}"


async def forward_to_toolhouse(session: aiohttp.ClientSession, payload: dict, timeout_s: float = 5.0) -> dict | None:
    if not TOOLHOUSE_URL:
        return None
    headers = {"Content-Type": "application/json"}
    if TOOLHOUSE_API_KEY:
        headers["Authorization"] = f"Bearer {TOOLHOUSE_API_KEY}"

    # Build a wrapped text prompt if requested, improves deterministic JSON replies
    body = payload
    if PAYLOAD_STYLE != "raw":
        obs = json.dumps(payload, ensure_ascii=False)
        if payload.get("event") == "session_end":
            # Session summary prompt
            prompt = (
                "You are VR Driving Coach AI. Summarize the driver's overall session based on this final score JSON and respond with a SINGLE compact JSON object only, no prose. "
                "Final: " + obs + ". "
                "Output schema: {\"summary\": string, \"tips\": [string,string,string], \"drills\": [string,string,string], \"priority\": \"speeding|lane|headway|smooth|compliance\"}. "
                "Rules: (1) Keep summary <= 160 chars. (2) Tips must be short, actionable, and specific. (3) Choose priority based on weakest weighted dimension."
            )
        else:
            # Realtime observation prompt
            prompt = (
                "You are VR Driving Coach AI. Evaluate this driving observation JSON and respond with a SINGLE compact JSON object only, no prose. "
                "Observation: " + obs + ". "
                "Output schema: {\"cue\": string|null, \"cue_level\": number|null, \"message\": string, \"notes\": string|null}. "
                "Rules: (1) Choose at most one cue unless imminent danger. (2) If no issue, set cue=null and write a short positive message. (3) Keep message under 140 chars."
            )
        if PAYLOAD_STYLE == "wrapped_input":
            body = {"input": prompt}
        else:
            # default 'wrapped' uses the 'message' field (matches your working curl)
            body = {"message": prompt}
    try:
        async with session.post(TOOLHOUSE_URL, json=body, headers=headers, timeout=timeout_s) as resp:
            status = resp.status
            try:
                body = await resp.json()
            except Exception:
                body = {"text": await resp.text()}
            print(f"[coach] forwarded event={payload.get('event')} status={status}")
            body["status"] = status
            return body
    except Exception as e:
        print(f"Coach forward error: {e}")
        return None

async def safe_send(ws, obj: dict) -> bool:
    if getattr(ws, "closed", False):
        return False
    try:
        await ws.send(json.dumps(obj))
        return True
    except (ConnectionClosed, ConnectionClosedOK):
        return False
    except Exception as e:
        print(f"Send error: {e}")
        return False


def _fallback_final_coach(final_result: dict) -> dict:
    subs = final_result.get("subscores", {})
    final = final_result.get("final", 0)
    pri = min(subs, key=subs.get) if subs else "headway"
    tips_map = {
        "speeding": [
            "Match speed to posted limit",
            "Lift early when approaching slower traffic",
            "Use cruise control to avoid creep",
        ],
        "lane": [
            "Center the car between lines",
            "Look farther ahead to stabilize steering",
            "Ease steering inputs—avoid ping‑pong",
        ],
        "headway": [
            "Open following gap to 2–3s",
            "Brake earlier, lighter when closing",
            "Avoid tailgating after lane changes",
        ],
        "smooth": [
            "Feather brake before stopping",
            "Plan ahead—no hard stabs",
            "Keep throttle steady out of turns",
        ],
        "compliance": [
            "Full stop at reds/stop lines",
            "Scan for pedestrians before turning",
            "Approach intersections off‑throttle",
        ],
    }
    drills_map = {
        "speeding": ["30‑mph road—hold ±1 mph for 2 min", "Practice coasting to limit signs", "Use speed checks every 10s"],
        "lane": ["Empty lot—center between cones", "Highway—hands light, eyes far", "2‑min no‑correction challenge"],
        "headway": ["Count 3‑second gap to lead", "Close/open gap smoothly", "Brake at 0.2g then release"],
        "smooth": ["Stop without ABS engagement", "No throttle spikes for 2 min", "Brake‑to‑zero with no head toss"],
        "compliance": ["Full stop 1s at line x5", "Red‑light scan left‑center‑right", "Yield practice in empty lot"],
    }
    summary = f"Score {final:.0f}/100. Weakest: {pri}. Focus on clean {pri} to lift overall." if subs else "Session complete. See tips."
    return {
        "summary": summary,
        "tips": tips_map.get(pri, tips_map["headway"]),
        "drills": drills_map.get(pri, drills_map["headway"]),
        "priority": pri,
        "source": "fallback",
    }

async def handler(websocket):
    connection_id = id(websocket)
    connections[connection_id] = {
        'frames': [],
        'telemetry': [],
        'session_id': str(uuid.uuid4()),
        'last_forward_ts': 0.0,
        'last_cue_fp': None,
    }

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    message = await websocket.recv()
                except (ConnectionClosed, ConnectionClosedOK):
                    # client closed the socket gracefully
                    break
                if isinstance(message, bytes):
                    # Image data
                    img = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                    connections[connection_id]['frames'].append(img)
                else:
                    # Telemetry JSON data or control message (`DONE`)
                    try:
                        data = json.loads(message)
                        connections[connection_id]['telemetry'].append(data)
                        print("Received telemetry:", data)

                        # Send frame + telemetry to inference API
                        if connections[connection_id]['frames']:
                            frame = connections[connection_id]['frames'][-1]  # Use latest frame
                            _, img_encoded = cv2.imencode('.jpg', frame)
                            img_bytes = img_encoded.tobytes()

                            form_data = aiohttp.FormData()
                            form_data.add_field('image', img_bytes, filename='frame.jpg', content_type='image/jpeg')
                            form_data.add_field('telemetry', json.dumps(data))

                            try:
                                async with session.post('http://localhost:8000/infer_frame', data=form_data) as resp:
                                    result = await resp.json()
                                    print("Inference result:", result)

                                    # Optionally forward a reduced observation to Toolhouse (rate-limited)
                                    now = time.time()
                                    tel = data
                                    cues = result.get('cues') or []
                                    top = cues[0] if cues else None
                                    obs = {
                                        "event": "observations",
                                        "session_id": connections[connection_id]['session_id'],
                                        "t": tel.get("t"),
                                        "lane_offset_m": tel.get("lane_offset_m"),
                                        "ttc": result.get("ttc"),
                                        "speed_mps": tel.get("speed_mps"),
                                        "speed_limit_mps": tel.get("speed_limit_mps"),
                                        "cue": (top.get('cue') if top else None),
                                        "cue_level": (top.get('level') if top else None),
                                        "detections": result.get("detections"),
                                    }

                                    cue_fp = _cue_fingerprint(obs)
                                    changed_cue = cue_fp != connections[connection_id]['last_cue_fp']
                                    last_ts = connections[connection_id]['last_forward_ts']
                                    first_send_ok = (last_ts == 0.0)
                                    interval_ok = (now - last_ts) >= FORWARD_MIN_INTERVAL_S
                                    should_send = changed_cue and (first_send_ok or interval_ok)
                                    coach_reply = None
                                    if should_send:
                                        coach_reply = await forward_to_toolhouse(session, obs)
                                        connections[connection_id]['last_forward_ts'] = now
                                        connections[connection_id]['last_cue_fp'] = cue_fp

                                    # Send cues back to client, include optional coach reply
                                    out = dict(result)
                                    out["type"] = "inference"
                                    if coach_reply is not None:
                                        out["coach"] = coach_reply
                                    await safe_send(websocket, out)
                            except Exception as e:
                                print(f"Error calling inference API: {e}")

                    except json.JSONDecodeError:
                        if message == "DONE":
                            # Request final score
                            try:
                                async with session.post('http://localhost:8000/end_session') as resp:
                                    final_result = await resp.json()

                                    # Prepare final payload immediately
                                    out = dict(final_result)
                                    out["type"] = "final"

                                    # Try to fetch coach quickly; don't block too long
                                    # Always attempt to fetch coach; wait up to TOOLHOUSE_FINAL_TIMEOUT_S
                                    final_payload = {
                                        "event": "session_end",
                                        "session_id": connections[connection_id]['session_id'],
                                        "final": final_result,
                                    }
                                    coach_final = await forward_to_toolhouse(session, final_payload, timeout_s=TOOLHOUSE_FINAL_TIMEOUT_S)
                                    if not coach_final or int(coach_final.get("status", 0)) >= 400:
                                        coach_final = _fallback_final_coach(final_result)
                                    out["coach"] = coach_final

                                    await safe_send(websocket, out)
                            except Exception as e:
                                print(f"Error getting final score: {e}")
                                # Fallback to dummy result
                                result = {"grade": "A", "confidence": 0.97, "type": "final"}
                                await safe_send(websocket, result)

                            connections[connection_id]['frames'].clear()
                            connections[connection_id]['telemetry'].clear()
                            break
                        else:
                            print("Unknown message type:", message)
    finally:
        # Clean up connection data
        if connection_id in connections:
            del connections[connection_id]


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

 
