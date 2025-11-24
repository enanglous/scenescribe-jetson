import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

sd.default.device[0] = 25
print(sd.query_devices(0))


def record_with_sounddevice_until_keypress(filename, sample_rate=16000):
    """
    Record audio using sounddevice library until a key is pressed
    """
    while True:
        key = input()
        if key.lower() == 'd':
            break
        time.sleep(0.1)
    print("Recording... Press 'd' to stop recording.")
    
    
    # List to store audio chunks
    audio_chunks = []
    
    # Flag to control recording
    recording = True
    
    def callback(indata, frames, time, status):
        """Callback function to process audio chunks"""
        if status:
            print(status)
        audio_chunks.append(indata.copy())
    
    # Start recording in a separate thread
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        callback=callback,
        blocksize=1024  # Adjust block size as needed
    )
    with stream:
        while True:
            key = input()
            if key.lower() == 'd':
                break
            time.sleep(0.1)
    
    print("Recording stopped.")
    
    # Combine all audio chunks
    if audio_chunks:
        audio_data = np.vstack(audio_chunks)
        
        # Save as WAV file
        wav.write(filename, sample_rate, audio_data)
        print(f"Audio saved as: {filename}")
        print(f"Recording duration: {len(audio_data) / sample_rate:.2f} seconds")
    else:
        print("No audio data recorded.")


record_with_sounddevice_until_keypress("recorded_audio.wav")