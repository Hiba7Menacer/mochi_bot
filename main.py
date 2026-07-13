import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env BEFORE any other imports
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify critical env vars
assert os.getenv("TELEGRAM_BOT_TOKEN"), "TELEGRAM_BOT_TOKEN not set in .env!"
assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY not set in .env!"
assert os.getenv("DEEPGRAM_API_KEY"), "DEEPGRAM_API_KEY not set in .env!"

from bot.telegram_bot import main

if __name__ == "__main__":
    main()