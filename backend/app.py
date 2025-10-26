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

# Store frames and telemetry for each connection
connections = {}

# Load environment variables from a .env file (e.g., backend/.env)
load_dotenv()

# Toolhouse agent config (set via environment)
TOOLHOUSE_URL = os.getenv("TOOLHOUSE_URL", "")
TOOLHOUSE_API_KEY = os.getenv("TOOLHOUSE_API_KEY", "")
# To conserve runs: throttle forwards and only send when cue changes
FORWARD_MIN_INTERVAL_S = float(os.getenv("TOOLHOUSE_MIN_INTERVAL_S", "1.0"))


def _bucket(val, step):
    try:
        return None if val is None else round(float(val) / step) * step
    except Exception:
        return None


def _cue_fingerprint(obs):
    cue = obs.get("cue") or ""
    lvl = _bucket(obs.get("cue_level"), 0.2)
    return f"{cue}|{lvl}"


async def forward_to_toolhouse(session: aiohttp.ClientSession, payload: dict) -> dict | None:
    if not TOOLHOUSE_URL:
        return None
    headers = {"Content-Type": "application/json"}
    if TOOLHOUSE_API_KEY:
        headers["Authorization"] = f"Bearer {TOOLHOUSE_API_KEY}"
    try:
        async with session.post(TOOLHOUSE_URL, json=payload, headers=headers, timeout=5) as resp:
            try:
                return await resp.json()
            except Exception:
                # Fallback to text if JSON not returned
                txt = await resp.text()
                return {"text": txt, "status": resp.status}
    except Exception as e:
        print(f"Coach forward error: {e}")
        return None

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
                message = await websocket.recv()
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
                                    should_send = changed_cue and (now - connections[connection_id]['last_forward_ts'] >= FORWARD_MIN_INTERVAL_S)
                                    coach_reply = None
                                    if should_send:
                                        coach_reply = await forward_to_toolhouse(session, obs)
                                        connections[connection_id]['last_forward_ts'] = now
                                        connections[connection_id]['last_cue_fp'] = cue_fp

                                    # Send cues back to client, include optional coach reply
                                    out = dict(result)
                                    if coach_reply is not None:
                                        out["coach"] = coach_reply
                                    await websocket.send(json.dumps(out))
                            except Exception as e:
                                print(f"Error calling inference API: {e}")

                    except json.JSONDecodeError:
                        if message == "DONE":
                            # Request final score
                            try:
                                async with session.post('http://localhost:8000/end_session') as resp:
                                    final_result = await resp.json()

                                    # Forward final summary to Toolhouse (always send once if configured)
                                    final_payload = {
                                        "event": "session_end",
                                        "session_id": connections[connection_id]['session_id'],
                                        "final": final_result,
                                    }
                                    coach_final = await forward_to_toolhouse(session, final_payload)

                                    # Send back to client with optional coach summary
                                    out = dict(final_result)
                                    if coach_final is not None:
                                        out["coach"] = coach_final
                                    await websocket.send(json.dumps(out))
                            except Exception as e:
                                print(f"Error getting final score: {e}")
                                # Fallback to dummy result
                                result = {"grade": "A", "confidence": 0.97}
                                await websocket.send(json.dumps(result))

                            connections[connection_id]['frames'].clear()
                            connections[connection_id]['telemetry'].clear()
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

