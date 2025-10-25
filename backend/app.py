import asyncio
import websockets
import json
import numpy as np
import cv2

image_frames = []

async def handler(websocket):
    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            # Image data
            img = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            image_frames.append(img)
        else:
            # Telemetry JSON data or control message (`DONE`)
            try:
                data = json.loads(message)
                print("Received JSON:", data)
            except json.JSONDecodeError:
                if message == "DONE":
                    result = {"grade": "A", "confidence": 0.97}
                    await websocket.send(json.dumps(result))
                    image_frames.clear()
                else:
                    print("Unknown message type:", message)

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

