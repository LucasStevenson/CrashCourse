# The idea is that the unity game sends about 10-15 frames + some textual telemetry data to the livekit python backend, which then sends the data to appropriate AI models. It sends the AI data back to the unity game, which verbally says the output

from livekit import api, rtc
import logging
import asyncio
import aiohttp
import requests
import os
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()  # take environment variables

# Code of your application, which uses environment variables (e.g. from `os.environ` or
# `os.getenv`) as if they came from the actual environment.

URL = os.getenv('LIVEKIT_URL')
TOKEN = api.AccessToken() \
    .with_identity("crashcourse") \
    .with_name("Python Bot") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
    )).to_jwt()

async def main():
    room = rtc.Room()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logging.info(
                "participant connected: %s %s", participant.sid, participant.identity)

    telemetry_cache = {}

    async def receive_video_frames(stream: rtc.VideoStream):
        async with aiohttp.ClientSession() as session:
            async for frame in stream:
                # Convert frame to JPEG bytes properly
                # Convert the frame to numpy array
                arr = frame.to_ndarray(format="bgr24")
                # Encode as JPEG
                _, image_bytes = cv2.imencode('.jpg', arr)
                image_bytes = image_bytes.tobytes()

                telemetry_str = telemetry_cache.get("latest", "{}")
                data = aiohttp.FormData()
                data.add_field('image', image_bytes, filename='frame.jpg', content_type='image/jpeg')
                data.add_field('telemetry', telemetry_str)

                try:
                    async with session.post('http://localhost:8000/infer_frame', data=data) as resp:
                        result = await resp.json()
                        print("Inference result:", result)
                except Exception as e:
                    logging.error(f"Error during inference: {e}")

    async def receive_telemetry_data(track: rtc.Track):
        while True:
            data = await track.read()  # reads the next data packet (bytes)
            if data is None:
                break  # track ended
            telemetry_cache["latest"] = data.decode('utf-8')

    # track_subscribed is emitted whenever the local participant is subscribed to a new track
    @room.on("track_subscribed")
    async def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logging.info("track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_stream = rtc.VideoStream(track)
            await receive_video_frames(video_stream)
        # if track.kind == rtc.TrackKind.TELEMETRY_DATA:
        if track.kind == rtc.TrackKind.KIND_UNKNOWN:
            await receive_telemetry_data(track)

    # By default, autosubscribe is enabled. The participant will be subscribed to
    # all published tracks in the room
    await room.connect(URL, TOKEN)
    logging.info("connected to room %s", room.name)

    # participants and tracks that are already available in the room
    # participant_connected and track_published events will *not* be emitted for them
    for identity, participant in room.remote_participants.items():
        print(f"identity: {identity}")
        print(f"participant: {participant}")
        for tid, publication in participant.track_publications.items():
            print(f"\ttrack id: {publication}")

    # Keep the connection alive
    await asyncio.Future()  # runs forever

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
