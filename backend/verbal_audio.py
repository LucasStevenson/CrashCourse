# Using fish audio AI for the verbal cues sent back to the user
from fish_audio_sdk import WebSocketSession, TTSRequest

class FishTTSStreamer:
    def __init__(self, api_key, voice_model_id=None):
        self.api_key = api_key
        self.voice_model_id = voice_model_id

    async def stream_tts(self, text, send_audio):
        # Generator splits text for lower latency streaming
        def text_chunks():
            for word in text.split():
                yield word + " "
        # Setup Fish Audio session
        with WebSocketSession(self.api_key) as session:
            request = TTSRequest(text="", reference_id=self.voice_model_id)
            # Receive audio chunks and forward to send_audio callback
            for chunk in session.tts(request, text_chunks()):
                await send_audio(chunk)

