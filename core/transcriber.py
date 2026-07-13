"""Deepgram voice-to-text transcriber."""
import os
import json
from config.settings import DEEPGRAM_API_KEY

def transcribe_voice(file_path):
    if not DEEPGRAM_API_KEY:
        print("[Transcriber] No DEEPGRAM_API_KEY found in settings")
        return None

    # Method 1: Deepgram SDK v3/v6 (latest) - with Pydantic v2 fix
    try:
        from deepgram import DeepgramClient
        from deepgram.core.api_error import ApiError

        # FIX: Pass api_key as keyword argument, not positional
        # Also set env var as fallback
        os.environ["DEEPGRAM_API_KEY"] = DEEPGRAM_API_KEY
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        with open(file_path, "rb") as f:
            audio_buffer = f.read()

        # Use transcribe_file with request= keyword
        response = client.listen.v1.media.transcribe_file(
            request=audio_buffer,
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
        )

        transcript = response.results.channels[0].alternatives[0].transcript
        print(f"[Transcriber] Deepgram v3 success: {transcript[:50]}...")
        return transcript

    except ImportError as e:
        print(f"[Transcriber] Deepgram SDK v3 not available: {e}")
    except Exception as e:
        print(f"[Transcriber] Deepgram v3 error: {type(e).__name__}: {e}")

    # Method 2: Deepgram SDK v2 (older)
    try:
        from deepgram import Deepgram

        dg = Deepgram(DEEPGRAM_API_KEY)

        with open(file_path, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/ogg"}
            response = dg.transcription.sync_prerecorded(
                source,
                {"punctuate": True, "language": "en", "model": "nova-2"}
            )

        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        print(f"[Transcriber] Deepgram v2 success: {transcript[:50]}...")
        return transcript

    except ImportError as e:
        print(f"[Transcriber] Deepgram SDK v2 not available: {e}")
    except Exception as e:
        print(f"[Transcriber] Deepgram v2 error: {type(e).__name__}: {e}")

    # Method 3: Direct HTTP API (most reliable fallback)
    try:
        import requests

        url = "https://api.deepgram.com/v1/listen?model=nova-2&language=en&punctuate=true&smart_format=true"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/ogg"
        }

        with open(file_path, "rb") as f:
            response = requests.post(url, headers=headers, data=f, timeout=30)

        response.raise_for_status()
        data = response.json()

        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        print(f"[Transcriber] Direct API success: {transcript[:50]}...")
        return transcript

    except Exception as e:
        print(f"[Transcriber] All methods failed. Final error: {type(e).__name__}: {e}")
        return None
