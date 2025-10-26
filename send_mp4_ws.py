import asyncio
import json
import time
import cv2
import websockets
import argparse
import math


def gen_telemetry(t: float, speed_limit_mps: float = 13.4) -> dict:
    # Simple synthetic telemetry resembling README format
    speed = 13.0 + 5.0 * math.sin(t * 0.4)
    return {
        "t": float(t),
        "speed_mps": float(max(0.0, speed)),
        "speed_limit_mps": float(speed_limit_mps),
        "throttle": 0.3,
        "brake": 0.0,
        "steer_deg": 0.0,
        "lane_offset_m": 0.0,
        "tl_state": "green",
        "in_stop_zone": False,
        "collision": False,
    }


async def stream_video(video_path: str, ws_url: str = "ws://localhost:8765", fps: float = 12.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    frame_period = 1.0 / max(1e-3, fps)
    t0 = time.time()

    async with websockets.connect(ws_url) as ws:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                ok_jpg, enc = cv2.imencode('.jpg', frame)
                if not ok_jpg:
                    print("Warn: JPEG encode failed; skipping frame")
                    await asyncio.sleep(frame_period)
                    continue

                # Send binary frame
                await ws.send(enc.tobytes())

                # Send matching telemetry
                t = time.time() - t0
                tel = gen_telemetry(t)
                await ws.send(json.dumps(tel))

                # Optionally read back inference result (non-blocking)
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    try:
                        data = json.loads(resp)
                        collided = data.get("collision")
                        ttc = data.get("ttc")
                        dist = data.get("lead_distance_m")
                        cues = data.get("cues")
                        kind = data.get("type", "inference")
                        print(f"Result[{kind}]: ttc={ttc}  dist_m={dist}  collided={collided}  cues={cues}")
                        coach = data.get("coach")
                        if coach is not None:
                            print("Coach:", coach)
                    except Exception:
                        print("Result:", resp)
                except asyncio.TimeoutError:
                    pass

                await asyncio.sleep(frame_period)

            # Signal end of session and read final
            await ws.send("DONE")
            # Drain messages until we receive the one marked as final
            received_final = False
            deadline = time.time() + 25.0
            while time.time() < deadline and not received_final:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.time()))
                except asyncio.TimeoutError:
                    break
                try:
                    data = json.loads(msg)
                except Exception:
                    print("Final (raw):", msg)
                    continue
                if data.get("type") == "final":
                    print("Final:", data)
                    if data.get("coach") is not None:
                        print("Coach (final):", data["coach"])
                    received_final = True
                else:
                    # late in-flight inference; print briefly and continue waiting
                    print("Late inference after DONE:", data)
            if not received_final:
                print("Final score response timed out or missing")
        finally:
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="ai/src/sample_drive.mp4", help="Path to MP4 file")
    parser.add_argument("--ws", default="ws://localhost:8765", help="WebSocket URL of backend")
    parser.add_argument("--fps", type=float, default=12.0, help="Send rate (frames per second)")
    args = parser.parse_args()

    asyncio.run(stream_video(args.video, args.ws, args.fps))
