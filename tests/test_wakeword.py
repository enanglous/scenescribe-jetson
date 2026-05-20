import pyaudio
import json
from vosk import Model, KaldiRecognizer

# 1. Point this to your extracted Vosk model folder
model_path = "vosk-model-small-en-us-0.15" 
try:
    model = Model(model_path)
except Exception as e:
    print(f"Failed to load model. Did you extract the folder to {model_path}?")
    exit(1)

rec = KaldiRecognizer(model, 16000)

# 2. Setup PyAudio microphone stream
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=16000, 
    input=True, 
    frames_per_buffer=8000
)

print("Microphone active.")
print("Waiting for 'Hey Glasses' to wake up.")
print("Say 'Stop Listening' to exit the program.")

try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        
        if rec.AcceptWaveform(data):
            rec.Result() 
        else:
            # Check the partial results for instant word detection
            partial_result = json.loads(rec.PartialResult())
            partial_text = partial_result.get("partial", "").lower()
            
            # --- WAKE WORD LOGIC ---
            if "hey glasses" in partial_text:
                print("\n✅ Wake word 'Hey Glasses' detected!")
                
                # Reset the recognizer so it doesn't trigger multiple times
                rec.Reset()
                
                # --> ADD YOUR WAKE ACTION HERE <--
            
            # --- STOP WORD LOGIC ---
            elif "stop listening" in partial_text:
                print("\n🛑 Stop word 'Stop Listening' detected! Shutting down...")
                break  # This breaks the while loop and moves to the 'finally' block
                
except KeyboardInterrupt:
    print("\nManual interrupt. Stopping script...")

finally:
    # Clean up hardware resources
    print("Cleaning up audio streams...")
    stream.stop_stream()
    stream.close()
    p.terminate()