import asyncio
import websockets
import json

TELEM_DATA = {
    "t": 5.3,
    "speed_mps": 20.5,
    "speed_limit_mps": 15.1,
    "throttle": 0.0,                 # unknown from video; keep 0
    "brake": 0.0,                    # unknown; rules still handle TTC/lanes
    "steer_deg": 0.0,                # unknown
    "lane_offset_m": 2.9,
    "tl_state": None,            # None if unknown
    "in_stop_zone": False,           # we don't know stop zone without a map
    "collision": False
}

async def test():
    uri = "ws://localhost:8765"
    image_path = "./Screen Shot 2025-10-26 at 3.47.09 AM.png"
    with open(image_path, "rb") as f:
        image = f.read()

    async with websockets.connect(uri) as websocket:
        # Send image and some telemetry first
        await websocket.send(image)
        await websocket.send(json.dumps(TELEM_DATA))
        
        # Initialize a list to store audio chunks
        audio_data = b''

        while True:
            try:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    audio_data += message
                else:
                    print(f"Received JSON: {message}")
            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed")
                break

        # Save the received audio data as a WAV file
        if audio_data:
            with open("output.mp3", 'wb') as f:
                f.write(audio_data)
            print("Audio saved to output.mp3")

asyncio.run(test())
