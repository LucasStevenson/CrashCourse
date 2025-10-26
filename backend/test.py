import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8765"
    # image_path = "./Screen Shot 2025-10-25 at 12.31.39 PM.png"
    # with open(image_path, "rb") as f:
    #     image = f.read()

    async with websockets.connect(uri) as websocket:
        # await websocket.send(json.dumps(DATA))
        # await websocket.send(image)
        # response = await websocket.recv()
        await websocket.send("TTS:Testing testing testing testing")
        response = await websocket.recv()
        print(f"Server: {response}")
        # await websocket.send("TTS:testing testing testing testing")
        #
        while True:
            try:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    print("Received audio chunk")
                    response += message
                else:
                    print(f"Received: {message}")
            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed")
                break
        with open("audio_test.mp3", 'wb') as f:
            f.write(response)

asyncio.run(test())
