import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8765"
    image_path = "./Screen Shot 2025-10-25 at 12.31.39 PM.png"
    with open(image_path, "rb") as f:
        image = f.read()

    async with websockets.connect(uri) as websocket:
        # Define some dummy telemetry data to send
        telemetry_payload = {
            "t": 0.0,
            "speed_mps": 20.0, # Example speed to trigger SLOW_DOWN cue
            "speed_limit_mps": 5.0,
            "throttle": 0.5,
            "brake": 0.0,
            "steer_deg": 0.0,
            "lane_offset_m": 0.1,
            "tl_state": "green",
            "in_stop_zone": False,
            "collision": False,
        }
        await websocket.send(json.dumps(telemetry_payload))
        await websocket.send(image)
        response = await websocket.recv()
        print(f"Server: {response}")

asyncio.run(test())
