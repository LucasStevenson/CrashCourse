import asyncio
import websockets
import json
import numpy as np
import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the new function from ai.src.api
from ai.src.api import process_image_and_telemetry

async def handler(websocket):
    telemetry_data = None

    while True:
        message = await websocket.recv()
        print(message)
        if isinstance(message, bytes):
            # Image data
            img_bytes = message
            if telemetry_data: # Process if we have telemetry
                # Call the new function
                result = process_image_and_telemetry(img_bytes, json.dumps(telemetry_data))
                print(result)
                await websocket.send(json.dumps(result["cues"])) # Send back only the cues
                telemetry_data = None # Reset telemetry after processing
            else:
                print("Received image without corresponding telemetry data.")
        else:
            # Telemetry JSON data or control message (`DONE`)
            try:
                data = json.loads(message)
                print("Received JSON:", data)
                # Store telemetry data, waiting for the next image frame
                telemetry_data = data
            except json.JSONDecodeError:
                if message == "DONE":
                    # You might want to send a final summary cue here if needed
                    pass
                else:
                    print("Unknown message type:", message)


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

