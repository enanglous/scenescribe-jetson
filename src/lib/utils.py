from openai import OpenAI
import sys
import cv2
import os
import base64
# from picamera2 import Picamera2, Preview
import pyttsx3
import speech_recognition as sr
import base64
from io import BytesIO
from PIL import Image
import io
import matplotlib.pyplot as plt
import threading
import time
import copy
import joblib
from utils.misc import extract_positions_from_string, generate_navigation
import whisper
import time
import soundfile as sf
import sounddevice as sd
import numpy as np
import wave
# import webrtcvad
# import firebase_admin
# from firebase_admin import db
from piper.voice import PiperVoice
from dotenv import load_dotenv
import noisereduce as nr
import scipy.io.wavfile as wavfile
import requests
import socket
import json
from vosk import Model, KaldiRecognizer
import queue
import warnings
import torch


# cred_obj = firebase_admin.credentials.Certificate('credentials/credentials.json')
# default_app = firebase_admin.initialize_app(cred_obj, {
#     'databaseURL':'https://scenescribe-d4be0-default-rtdb.asia-southeast1.firebasedatabase.app'
#     })



text = ""

conversation_history = [{
    "role": "system",
    "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful."}]

video_prompt = "Please explain what is happening in the video!"
endpoint_url = "https://6e65-119-158-64-26.ngrok-free.app/analyze_video/"


user_text = None
lock = threading.Lock()  # To ensure thread-safe updates to `user_text`

model = whisper.load_model("tiny")
# Recording settings
SAMPLE_RATE = 16000  # Whisper requires 16kHz
AUDIO_FILE = "recorded_audio.wav"  # Output file
FRAME_DURATION = 30  # Frame size in milliseconds


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.button_on = False

        # --- Thread-safe camera sharing ---
        self.frame_lock = threading.Lock()
        self.current_frame = None

        # --- Add a TTS lock to prevent concurrent speaker usage ---
        self.tts_lock = threading.Lock()

    def set_button_state(self, state: bool):
        with self.lock:
            self.button_on = state

    def get_button_state(self) -> bool:
        with self.lock:
            return self.button_on



class Utils:
    def __init__(self, vidStream=None, recognizer=None, whisper_model=None, openai_client=None, tts_model=None, tts_sample_rate=None, tts_speaker=None, shared_state: SharedState = None):
        self.latest_video_path = None
        # self.picam2 = picamera if picamera else Picamera2()
        self.recognizer = recognizer if recognizer else sr.Recognizer()
        self.whisper_model = whisper_model if whisper_model else whisper.load_model("tiny")
        self.openai = openai_client
        self.shared_state = shared_state if shared_state else SharedState()
        self.vidStream = vidStream if vidStream else cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.offline_tts_model = tts_model
        self.tts_sample_rate = tts_sample_rate
        self.tts_speaker = tts_speaker

    def encode_image(self,image_path):

        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P', 'LA'):
                img = img.convert('RGB')
        
            # Save image to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # base64 encode
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return image_base64
        

    def get_image(self):
        image_path = "/home/scenescribe/Pictures/IMG_1977.jpeg"

        ret = False
        with self.shared_state.frame_lock:
            if self.shared_state.current_frame is not None:
                frame = self.shared_state.current_frame.copy()
                ret = True
        
        if ret:
            # 4. Save the image
            cv2.imwrite(image_path, frame)
            print(f"Image successfully saved as {image_path}")
        else:
            print("Error: Could not read frame.")
        return image_path

    def record_video_from_camera(self, duration=5, fps=2, output_filename="video_smolvlm.avi",
                                  resolution=(640, 640), video_dir="/home/scenescribe/scenescribe/avis"):
        """
        Records video from a pre-initialized Picamera2 object.

        Args:
            picam2: Initialized Picamera2 object.
            duration (int): Duration in seconds to record.
            fps (int): Frames per second.
            output_filename (str): Name of the AVI file to save.
            resolution (tuple): Frame size (width, height).
            video_dir (str): Directory to save video.

        Returns:
            str: Full path to the saved video file.
        """

        # Ensure save directory exists
        os.makedirs(video_dir, exist_ok=True)

        # Prepare output path and writer
        filepath = os.path.join(video_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filepath, fourcc, fps, resolution)

        print(f"📹 Recording video to: {filepath}")

        start_time = time.time()
        while time.time() - start_time < duration:
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            time.sleep(1 / fps)

        # Cleanup
        out.release()
        print("✅ Recording finished.")
        self.latest_video_path = filepath
        return filepath

    def analyze_video_with_prompt(self,video_path = None, prompt_text=video_prompt, endpoint_url=endpoint_url):
        """
        Sends a video and prompt to an analysis API endpoint and returns the response.

        Args:
            video_path (str): Full path to the video file.
            prompt_text (str): Instruction or question for the model.
            endpoint_url (str): URL of the analysis API.

        Returns:
            dict: Response from the server containing status and text.
        """
        if video_path is None:
            video_path = self.latest_video_path
        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                data = {'prompt': prompt_text}

                response = requests.post(endpoint_url, files=files, data=data)

                return {
                    "status_code": response.status_code,
                    "response_text": response.text
                }

        except Exception as e:
            return {
                "status_code": None,
                "response_text": f"Error: {e}"
            }
    
    def wait_for_listen_command(self):
        try:
                print("Initializing microphone...")
                try:
                    self.record_with_softap_control(AUDIO_FILE, SAMPLE_RATE, FRAME_DURATION)
                    # Transcribe the recorded audio
                    start_time = time.time()
                    # result = model.transcribe(AUDIO_FILE)
                    audio_file= open(AUDIO_FILE, "rb")
                    transcription = self.openai.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                    end_time = time.time()
                    result= transcription.text

                    # Print the transcribed text and processing time
                    print(f"? Processing Time: {end_time - start_time:.2f} seconds")
                    print("?? Transcribed Text:", result)
                    return result

                except sr.UnknownValueError:
                    # Handle case where speech was not understood
                    print("Could not understand audio. Please try again.")
                except sr.RequestError as e:
                    # Handle API request issues
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    # Exit the loop if there's a serious issue
                except sr.WaitTimeoutError:
                    # Handle timeout waiting for speech
                    print("Listening timed out. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    def convert_and_play_speech(self,  text):
        # Load the voice model
        model = "models/tts/en_GB-northern_english_male-medium.onnx"
        voice = PiperVoice.load(model)
        
        # Generate speech and save to .wav file
        with wave.open("output.wav", "w") as wav_file:
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
            wav_file.setframerate(22050)  # 22.05 kHz sample rate (adjust if needed)
            voice.synthesize(text, wav_file)
        
        # Read the generated .wav file and play it through speakers
        wav_data = np.fromfile("output.wav", dtype=np.int16)
        sd.play(wav_data, 22050)  # Play the audio at the same sample rate as the .wav
        sd.wait()  # Wait until the audio finishes playing

    def openai_convert_and_play_speech(self, text):
        # --- NEW: Thread safe lock wrapper ---
        # 1. Generate audio with specific speed (lower = slower)
        # Default is 1.0. Try 0.8 or 0.9 for a calmer navigation pace.
        
        with self.shared_state.tts_lock:
            # # Call OpenAI TTS API
            # response = self.openai.audio.speech.create(
            #     model="tts-1",  
            #     voice="alloy",  
            #     input=text
            # )
            # # Save the response audio content to a .wav file
            # with open("output.wav", "wb") as f:
            #     f.write(response.content)
            
            # # Read the generated .wav file and play it through speakers
            # data, samplerate = sf.read("output.wav")
            # volume_boost = 1.5
            # data = data * volume_boost
            # data = data.clip(-1.0, 1.0)
            
            # sd.play(data, samplerate)  
            # sd.wait()
            
            ssml_text = f'<speak><prosody rate="slow">{text}</prosody></speak>'
            with torch.no_grad():
                audio_tensor = self.offline_tts_model.apply_tts(
                    ssml_text= ssml_text,
                    speaker=self.tts_speaker,
                    sample_rate=self.tts_sample_rate
                )

            # 2. Convert to numpy
            data = audio_tensor.cpu().numpy()

            # 3. Increase volume (Bumped to 2.0x for better clarity on the Jetson)
            volume_boost = 2.0
            data = data * volume_boost

            # 4. Prevent clipping/distortion
            data = np.clip(data, -1.0, 1.0)

            # 5. Optional: Save to file (in case you need it for logs)
            sf.write("output.wav", data, self.tts_sample_rate)

            # 6. Play
            sd.play(data, self.tts_sample_rate)
            sd.wait()

    def classify_input(self, sentence, loaded_model, loaded_vectorizer):
        sentence_transformed = loaded_vectorizer.transform([sentence])
        prediction = loaded_model.predict(sentence_transformed)
        print(f"The query '{sentence}' is classified as: {prediction[0]}")
        return prediction[0]


    def denoise_wav(self, file_path):
        print(f"🔧 Denoising audio: {file_path}")
        rate, data = wavfile.read(file_path)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1).astype(np.int16)
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        if reduced_noise.dtype != np.int16:
            reduced_noise = np.clip(reduced_noise, -32768, 32767).astype(np.int16)
        wavfile.write(file_path, rate, reduced_noise)
        print("✅ Noise reduction complete.")

    # --- Unified Recording Function ---
    def record_with_softap_control(self, filename="recorded_audio.wav", sample_rate=16000, frame_duration=30):
        print("Loading voice model...")
        model_path = "models/vosk-model-small-en-us-0.15"
        try:
            model = Model(model_path)
            rec = KaldiRecognizer(model, sample_rate)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            return

        print("🔁 Microphone active. Waiting for 'Hey Glasses' to start recording...")
        
        q = queue.Queue()
        audio_chunks = [] 
        self.is_recording = False 
        
        def callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            q.put(bytes(indata))
        
        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            blocksize=4000, 
            callback=callback
        ):
            
            while True:
                data = q.get()
                
                if self.is_recording:
                    audio_chunks.append(data)
                
                if rec.AcceptWaveform(data):
                    rec.Result()
                else:
                    partial_result = json.loads(rec.PartialResult())
                    partial_text = partial_result.get("partial", "").lower()
                    
                    # --- WAITING FOR WAKE WORD ---
                    if not self.is_recording:
                        if "hey glasses" in partial_text:
                            print("\n✅ 'Hey Glasses' detected! STARTING RECORDING.")
                            
                            # --- 🔊 PLAY BEEP ---
                            # Generate a 0.2 second sine wave at 1000Hz
                            t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
                            # Multiply by 10000 for a comfortable volume (max is 32767)
                            beep = (np.sin(1000 * t * 2 * np.pi) * 10000).astype(np.int16) 
                            try:
                                sd.play(beep, samplerate=sample_rate)
                            except Exception as e:
                                print(f"Could not play beep: {e}")
                            # ---------------------
                                
                            print("Speak your message, then say 'Stop Listening' to end.")
                            self.is_recording = True
                            rec.Reset() 
                            
                    # --- ACTIVELY RECORDING ---
                    else:
                        if "stop listening" in partial_text:
                            print("\n🛑 'Stop Listening' detected! STOPPING RECORDING.")
                            
                            # Optional: Play a lower pitched beep to indicate stopping
                            t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
                            stop_beep = (np.sin(500 * t * 2 * np.pi) * 10000).astype(np.int16)
                            try:
                                sd.play(stop_beep, samplerate=sample_rate)
                            except:
                                pass
                                
                            break 
        
        print("Recording hardware stopped.")
        
        if len(audio_chunks) > 0:
            combined_bytes = b''.join(audio_chunks)
            audio_data = np.frombuffer(combined_bytes, dtype=np.int16)
            
            if audio_data.size > 0:
                wavfile.write(filename, sample_rate, audio_data)
                print(f"Audio saved as: {filename}")
                print(f"Recording duration: {len(audio_data) / sample_rate:.2f} seconds")
                
                if hasattr(self, 'denoise_wav'):
                    self.denoise_wav(filename)
                print(f"✅ Finished processing: {filename}")
            else:
                print("Error: Audio data was empty after conversion.")
        else:
            print("No audio data was recorded between the wake words.")
        
def check_network_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if the network connection is available.
    Tries to connect to a public DNS server (Google).
    Returns True if network is available, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

ground_floor_hierarchical_tree = {
    "Department Entrance": {
        "In Front": {
            "Reception": "Central point connecting all branches",
            "Stairs": "Leads to the upper floor"
        },
        "Right Turn": {
            "Right corridor": {
                "CR MTS 01": "1st room on the right side",
                "Industrial Automation Lab": "1st room on the left side",
                "Robotics Lab": "2nd room on the left side",
                "HOD Corridor": "on the Straight at the end of the corridor",
                "Secondary Exit": "on the Right side on end of the corridor"
            }
        },
        "Left Turn": {
            "Left corridor": {
                "CAD/CAM Lab": "1st room on the right side",
                "Machine Vision Lab": "1st room on the left side",
                "Electronics Lab": "2nd room on the right side",
                "Washroom": "on the Left side on the end of the corridor",
                "Second stairs": "on the Left side on the end of the corridor"
            }
        }
    }
}
