import os
import tempfile
from gtts import gTTS

async def send_voice_reply(update, context, text, voice_gender="female"):
    try:
        tts = gTTS(text=text[:500], lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts.save(fp.name)
            await update.message.reply_voice(voice=open(fp.name, "rb"))
            os.remove(fp.name)
    except Exception as e:
        print(f"[VoiceReply] Error: {e}")
