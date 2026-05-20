import time
from openai import OpenAI

# 1. Initialize the client to point to your Docker container
# The API key doesn't matter for local servers, but the library requires the field
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-server-key" 
)

AUDIO_FILE = "/home/scenescribe/scenescribe_exp/recorded_audio.wav"
TARGET_LANGUAGE = "en" # Urdu

print("Sending audio to local Faster-Whisper container...")
start_time = time.time()

try:
    with open(AUDIO_FILE, "rb") as audio:
        # 2. Call the transcription endpoint exactly like cloud OpenAI
        transcription = client.audio.transcriptions.create(
            model="Systran/faster-whisper-base",  # The container will download/load the 'base' model automatically
            file=audio,
            language=TARGET_LANGUAGE
        )
    
    end_time = time.time()
    
    print("\n" + "="*50)
    print(f"⏱️ Inference Time: {end_time - start_time:.2f} seconds")
    print(f"🗣️ Transcribed Text: {transcription.text.strip()}")
    print("="*50)

except Exception as e:
    print(f"❌ API Error: {e}")