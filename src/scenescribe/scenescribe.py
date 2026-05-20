#!/usr/bin/env python3
"""
SceneScribe - A visual assistance system for visually impaired individuals

This system uses computer vision, speech recognition, and AI to provide 
scene descriptions and navigation assistance to visually impaired users.
"""

import os
import sys
import time
import joblib
from openai import OpenAI
import speech_recognition as sr
import whisper
from dotenv import load_dotenv
import logging
import sys
import cv2
import time
# Import our modules
from ..lib.utils import SharedState, Utils, check_network_connection
from ..lib.agents import Agents
import cv2.aruco as aruco
import numpy as np
from ultralytics import YOLO
import networkx as nx
from collections import deque, Counter

from faster_whisper import WhisperModel

import threading
from ..yolo3d.detection_model import ObjectDetector
from ..yolo3d.depth_model import DepthEstimator
from ..yolo3d.haptic_controller import HapticController

import logging
import re
import torch

# Define the log file path
log_file = '/home/scenescribe/scenescribe_exp/error.log'

# Clear the log file if it exists
if os.path.exists(log_file):
    os.remove(log_file)

# Configure logging to capture errors
# logging.basicConfig(filename=log_file, level=logging.DEBUG)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # <-- prints to terminal
    ]
)


try:
    # Your script logic here
    logging.info("Starting the script")
    
    # Example script code that might raise an error
    # result = 1 / 0  # Example of an error (divide by zero)
    
except Exception as e:
    logging.error(f"Error occurred: {e}")
    raise

# # Define a signal handler to ignore SIGINT (Ctrl+C)
# def signal_handler(sig, frame):
#     logging.info("Ctrl+C was pressed, but the script will continue running.")
#     # Do nothing, just return so the script keeps running
#     pass

# # Attach the signal handler to SIGINT
# signal.signal(signal.SIGINT, signal_handler)

class SceneScribe:
    def __init__(self, shared_state: SharedState, language="English"):
        """
        Initialize the SceneScribe application.
        
        Args:
            language: Language for output (defaults to English)
        """
        logging.info(f"Initializing SceneScribe with {language} language...")
        self.sharedState = shared_state
        self.language = language

        self.navigation_active = False
        self.navigation_paused = False
        self.navigation_thread = None
        self.current_destination = None

        self.setup_environment()
        self.load_models()
        self.vidStream = self.initialize_camera_with_gst(device_id=0)

        self.camera_paused = False
        
        # Start the background obstacle avoidance
        self.oa_thread = threading.Thread(target=self.obstacle_avoidance_daemon, daemon=True)
        self.oa_thread.start()
        
        # Wait a split second to ensure the thread grabs the first frame
        time.sleep(1)
        
        # Initialize conversation history
        self.conversation_history = [{
            "role": "system",
            "content": "This is the chat history between the user and the assistant. Use the conversation below as context when generating responses. Be concise and helpful."
        }]
        
        # Initialize utility functions and agents
        self.utils = Utils(
            # picamera=self.picamera,
            recognizer=self.recognizer,
            whisper_model=self.whisper_model,
            openai_client=self.openai,
            shared_state = self.sharedState,
            vidStream=self.vidStream,
            tts_model=self.offline_tts_model,
            tts_sample_rate = self.tts_sample_rate,
            tts_speaker = self.tts_speaker
        )
        
        self.agents = Agents(
            openai_client=self.openai, 
            conversation_history=self.conversation_history,
            language=self.language
        )
        
        # API endpoint for video analysis
        self.endpoint_url = "https://ccdd-59-103-82-46.ngrok-free.app/analyze_video/"
        self.educational_mode = False
        
    def capture_high_res_image(self):
        """Temporarily switches camera to 1080p, grabs a frame, and reverts to 480p."""
        logging.info("Switching to high-res (1080p) for VLM capture...")
        self.camera_paused = True
        time.sleep(0.2) # Give the OA daemon a split second to finish its last read
        
        # Release the low-res fast stream
        if self.vidStream:
            self.vidStream.release()
            
        # Open high-res stream (will be slow FPS, but we only need 1 frame)
        high_res_stream = self.initialize_camera_with_gst(device_id=0, width=1920, height=1080)
        
        image_path = "/home/scenescribe/Pictures/high_res_scene.jpeg"
        if high_res_stream:
            ret, frame = high_res_stream.read()
            if ret:
                # Rotate exactly like your daemon does
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(image_path, frame)
                logging.info(f"High-res image successfully saved as {image_path}")
            else:
                logging.error("Failed to read high-res frame.")
            high_res_stream.release()
            
        # Restore the low-res 30fps stream for obstacle avoidance
        logging.info("Reverting to low-res (480p) for depth estimation...")
        self.vidStream = self.initialize_camera_with_gst(device_id=0, width=640, height=480)
        self.camera_paused = False
        
        return image_path


    def setup_environment(self):
        """Set up environment variables and suppress warnings"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment or set directly
        self.openai_key = os.getenv("OPENAI_API_KEY") 
        
        if not self.openai_key:
            logging.info("OpenAI API key is missing. Please set it in the environment variable or directly in the script.")
            sys.exit(1)
            
        # Suppress standard error for cleaner output
        sys.stderr = open(os.devnull, 'w')
    
    def load_models(self):
        logging.info("Loading models...")
        self.openai = OpenAI(
            api_key=self.openai_key
        )
        self.loaded_model = joblib.load("/home/scenescribe/scenescribe_exp/models/nb_classifier_3_classes_v2.pkl")
        self.loaded_vectorizer = joblib.load("/home/scenescribe/scenescribe_exp/models/vectorizer_3_classes_v2.pkl")
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper.load_model("tiny")

        logging.info("Loading YOLO Navigation Engine...")
        self.yolo_nav_model = YOLO('/home/scenescribe/scenescribe_exp/models/final.engine', task='classify')
        
        self.pos = {
            "Circuit Design Lab":               (6, 5),
            "Machine Vision Lab":               (12, -5),
            "CAD-CAM Lab":                      (18, 5),
            "Entrance":                         (24, -10),
            "Outside Circuit Design Lab":       (6, 0),
            "Outside Machine Vision Lab":       (12, 0),  
            "Outside CAD-CAM Lab":              (18, 0),
            "Reception":                        (24, 0),
            "Outside MTS-01":                   (30, 0),
            "Outside Robotics Lab":             (45, 0),
            "Outside Industrial Automation Lab":(36, 0), 
            "MTS-01":                           (30, -5),
            "Robotics Lab":                     (45, 5),
            "Industrial Automation Lab":        (36, 5),
        }

        self.aliases = {
            "reception": "Reception",
            "front desk": "Reception",
            "main desk": "Reception",
            "lobby": "Reception",
            "help desk": "Reception",
            "info desk": "Reception",
            "waiting area": "Reception",
            "entrance": "Entrance",
            "exit": "Entrance",
            "main door": "Entrance",
            "main gate": "Entrance",

            "classroom": "MTS-01",
            "mts-1": "MTS-01",
            "mts1": "MTS-01",
            "mts 1.": "MTS-01",
            "mts 1": "MTS-01",
            "mts-01": "MTS-01",
            "mts01": "MTS-01",
            "mts 01": "MTS-01",
            "lecture room 1": "MTS-01",
            "room 1": "MTS-01",
            "class 1": "MTS-01",

            "machine vision": "Machine Vision Lab",
            "machine visual lab": "Machine Vision Lab",
            "vision lab": "Machine Vision Lab",
            "mvl": "Machine Vision Lab",
            "computer vision lab": "Machine Vision Lab",
            "cv lab": "Machine Vision Lab",
            "mv lab": "Machine Vision Lab",
            "image processing lab": "Machine Vision Lab",

            "cad": "CAD-CAM Lab",
            "cam": "CAD-CAM Lab",
            "cad lab": "CAD-CAM Lab",
            "cam lab": "CAD-CAM Lab",
            "cad/cam": "CAD-CAM Lab",
            "cad cam": "CAD-CAM Lab",
            "cadcam": "CAD-CAM Lab",

            "circuit lab": "Circuit Design Lab",
            "circuit design lab": "Circuit Design Lab",
            "circuits lab": "Circuit Design Lab",
            "circuit design": "Circuit Design Lab",
            "cdl": "Circuit Design Lab",
            "electronics lab": "Circuit Design Lab",
            "electrical lab": "Circuit Design Lab",
            "pcb lab": "Circuit Design Lab",
            "circuitry lab": "Circuit Design Lab",
            "hardware lab": "Circuit Design Lab",

            "robotics lab": "Robotics Lab",
            "robot lab": "Robotics Lab",
            "robotics": "Robotics Lab",
            "robo lab": "Robotics Lab",
            "bot lab": "Robotics Lab",
            "rl": "Robotics Lab",

            "industrial automation": "Industrial Automation Lab",
            "automation lab": "Industrial Automation Lab",
            "automation": "Industrial Automation Lab",
            "ia lab": "Industrial Automation Lab",
            "ial": "Industrial Automation Lab",
            "industrial lab": "Industrial Automation Lab",
            "plc lab": "Industrial Automation Lab"
        }

        self.nav_graph = nx.Graph()
        edges = [
            ("Entrance", "Reception", 10),
            ("Reception", "Outside CAD-CAM Lab", 12),
            ("Outside Circuit Design Lab", "Circuit Design Lab", 5),
            ("Outside Machine Vision Lab", "Machine Vision Lab", 5),
            ("Outside CAD-CAM Lab", "CAD-CAM Lab", 5),
            ("Outside CAD-CAM Lab", "Outside Machine Vision Lab", 6),
            ("Outside Machine Vision Lab", "Outside Circuit Design Lab", 10),
            ("Reception", "Outside MTS-01", 6),
            ("Outside MTS-01", "Outside Industrial Automation Lab", 12),
            ("Outside MTS-01", "MTS-01", 5),
            ("Outside Industrial Automation Lab", "Outside Robotics Lab", 10),
            ("Outside Robotics Lab", "Robotics Lab", 5)
        ]
        self.nav_graph.add_weighted_edges_from(edges)

        logging.info("Loading Depth & OA Models...")
        self.detector = ObjectDetector(model_size='nano', model_path='/home/scenescribe/scenescribe_exp/models', conf_thres=0.3, device='cuda')
        self.depth_estimator = DepthEstimator(
            engine_path='/home/scenescribe/scenescribe_exp/models/depth_anything_v2_metric_hypersim_vits_fp16.engine', 
            device='cuda'
        )
        self.haptics = HapticController(vibration_time=0.5, duty_cycle=25)
        self.SAFE_DISTANCE = 1.8

        logging.info("Loading Offline TTS (Silero)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.offline_tts_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en',
            trust_repo=True
        )
        self.offline_tts_model.to(self.device)
        self.tts_sample_rate = 24000
        self.tts_speaker = 'en_11' 
        logging.info("Models loaded successfully")
    
    # def initialize_camera(self):
    #     """Initialize PiCamera2 for capturing images and video"""
    #     logging.info("Initializing Camera...")
        
    #     self.picamera = Picamera2()
    #     # camera_config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
    #     # picam2.configure(camera_config)
    #     self.picamera.start()
    #     time.sleep(0.1)
    #     # picam2.set_controls({"AfMode": 2})
    #     # picam2.set_controls({"AfTrigger": 0})
        
    #     logging.info("Camera Initialized")

    def navigate_with_turns(self, start, end, heading):
        curr_heading = heading
        try:
            path = nx.shortest_path(self.nav_graph, start, end, weight='weight')
        except nx.NetworkXNoPath:
            return ["Path not found."]

        instructions = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            p1, p2 = np.array(self.pos[u]), np.array(self.pos[v])
            
            target_vec = p2 - p1
            steps = self.nav_graph[u][v].get('weight', 1)

            if np.linalg.norm(target_vec) < 0.1:
                continue
                
            unit_target = target_vec / np.linalg.norm(target_vec)
            cross_prod = curr_heading[0] * unit_target[1] - curr_heading[1] * unit_target[0]
            dot_prod = np.dot(curr_heading, unit_target)

            if dot_prod > 0.9: turn = "go straight"
            elif dot_prod < -0.8: turn = "turn around"
            elif cross_prod > 0: turn = "turn left"
            else: turn = "turn right"

            # --- UPDATED: Conditional instructions for the final approach ---
            if v == end and v != "Reception":
                if end == "Entrance":
                    instructions.append(f"Go forward 5 steps, then {turn} and walk {steps} steps to {v}.")
                else:
                    instructions.append(f"Walk forward 3 steps, then {turn} and walk {steps} steps to {v}.")
            else:
                # Standard routing format for all other intermediate nodes
                instructions.append(f"At {u}, {turn}, walk {steps} steps to {v}.")
                
            curr_heading = unit_target

        return instructions
    

    def obstacle_avoidance_daemon(self):
        """Runs continuously in the background, updating haptics and the shared frame buffer."""
        logging.info("Started Obstacle Avoidance Background Thread.")
        
        while True:
            if getattr(self, 'camera_paused', False):
                time.sleep(0.1)
                continue
            try:
                
                ret, frame = self.vidStream.read()
                if not ret:
                    continue

                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # 1. Update the shared frame so other functions can use it
                with self.sharedState.frame_lock:
                    self.sharedState.current_frame = frame.copy()

                # 2. Run Inference
                annotated_frame, detections = self.detector.detect(frame, track=False)
                depth_map = self.depth_estimator.estimate_depth(frame)

                H, W = frame.shape[:2]
                third_w = W // 3
                sectors = {
                    "LEFT": {"start": 0, "end": third_w, "min_depth": float('inf')},
                    "CENTER": {"start": third_w, "end": 2 * third_w, "min_depth": float('inf')},
                    "RIGHT": {"start": 2 * third_w, "end": W, "min_depth": float('inf')}
                }

                # 3. Process Depth with Grid Chunks
                for detection in detections:
                    bbox, score, class_id, obj_id, obj_mask = detection
                    ix1, iy1, ix2, iy2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    chunk_size = 20
                    
                    # Iterate over bounding box in grid chunks
                    for y in range(iy1, iy2, chunk_size):
                        for x in range(ix1, ix2, chunk_size):
                            y_end = min(y + chunk_size, iy2)
                            x_end = min(x + chunk_size, ix2)
                            
                            chunk_mask = obj_mask[y:y_end, x:x_end]
                            valid_chunk_pixels = depth_map[y:y_end, x:x_end][chunk_mask == 1]
                            
                            # Dynamically calculate required pixels (handles edges smaller than 20x20)
                            min_required = (x_end - x) * (y_end - y) * 0.3
                            
                            if len(valid_chunk_pixels) > min_required:
                                chunk_avg = float(np.mean(valid_chunk_pixels))
                                
                                # Determine which sector THIS specific chunk belongs to
                                cx = x + ((x_end - x) // 2)
                                
                                if cx < third_w:
                                    sector_name = "LEFT"
                                elif cx < 2 * third_w:
                                    sector_name = "CENTER"
                                else:
                                    sector_name = "RIGHT"
                                    
                                # Update the sector's minimum depth if this chunk is closer
                                if chunk_avg < sectors[sector_name]["min_depth"]:
                                    sectors[sector_name]["min_depth"] = chunk_avg

                # 4. Decision Matrix
                center_dist = sectors["CENTER"]["min_depth"]
                left_dist = sectors["LEFT"]["min_depth"]
                right_dist = sectors["RIGHT"]["min_depth"]

                action = "CLEAR - MOVE FORWARD"
                if center_dist < self.SAFE_DISTANCE:
                    if left_dist > self.SAFE_DISTANCE and left_dist >= right_dist:
                        action = "OBSTACLE: TURN LEFT"
                    elif right_dist > self.SAFE_DISTANCE and right_dist > left_dist:
                        action = "OBSTACLE: TURN RIGHT"
                    else:
                        action = "BLOCKED: STOP / TURN AROUND"

                # 5. Fire Hardware
                self.haptics.process_command(action)
                
            except Exception as e:
                logging.error(f"OA Daemon Error: {e}")
                time.sleep(0.1) # Prevent aggressive looping on crash
    
    
    def initialize_camera_with_gst(self, device_id=0, width=640, height=480):
        """Initialise camera for capturing images and video"""
        logging.info(f"Initializing Camera at {width}x{height}...")

        # You commented out the GST pipeline, so we enforce resolution via V4L2 props
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if cap.isOpened():
            for _ in range(20):
                cap.read()
            logging.info("Camera Initialized")
            return cap
        else:
            logging.info("Camera failed to initialize")
            return None
    
    def process_scene_explanation(self, user_input):
        """
        Handle scene explanation queries by capturing an image and describing it.
        
        Args:
            user_input: User's query text
            
        Returns:
            str: Scene description
        """
        logging.info("Capturing image for scene explanation...")
        img_path = self.utils.get_image()
        base64_image = self.utils.encode_image(img_path)
        
        logging.info("Processing with Agent 1...")
        agent_1_output = self.agents.explanation_agent_1(base64_image, user_input)
        logging.info(f"Agent 1 Output: {agent_1_output}")
        
        # logging.info("Processing with Agent 2...")
        # agent_2_output = self.agents.explanation_agent_2(user_input, agent_1_output)
        # logging.info(f"Agent 2 Output: {agent_2_output}")
        
        return agent_1_output
    
    # def process_educational_explanation(self, user_input):
    #     """
    #     Handle educational explanation queries by capturing an image and describing it.
        
    #     Args:
    #         user_input: User's query text
            
    #     Returns:
    #         str: educational description
    #     """
    #     logging.info("Capturing image for educational explanation...")
    #     img_path = self.utils.get_image()
    #     base64_image = self.utils.encode_image(img_path)
        
    #     logging.info("Processing with Agent 1...")
    #     agent_1_output = self.agents.educational_agent_1(base64_image, user_input)
    #     logging.info(f"Agent 1 Output: {agent_1_output}")
        
    #     logging.info("Processing with Agent 2...")
    #     agent_2_output = self.agents.educational_agent_2(user_input, agent_1_output)
    #     logging.info(f"Agent 2 Output: {agent_2_output}")
        
    #     return agent_2_output

    # def process_local_navigation(self, user_input):
    #     """
    #     Handle navigation queries by capturing an image and providing directions.
        
    #     Args:
    #         user_input: User's query text
            
    #     Returns:
    #         str: Navigation instructions
    #     """
    #     logging.info("Capturing image for navigation...")
    #     img_path = self.utils.get_image()
    #     base64_image = self.utils.encode_image(img_path)
        
    #     logging.info("Processing with Navigation Agent 1...")
    #     agent_1_output = self.agents.navigation_agent_1(base64_image, user_input)
    #     logging.info(f"Navigation Agent 1 Output: {agent_1_output}")
        
    #     logging.info("Processing with Navigation Agent 2...")
    #     agent_2_output = self.agents.navigation_agent_2(user_input, agent_1_output)
    #     logging.info(f"Navigation Agent 2 Output: {agent_2_output}")
        
    #     return agent_2_output
    
    # --- NEW: Extracted Destination Parser ---
    def parse_destination(self, user_input):
        user_text = user_input.lower()
        for keyword, node_name in self.aliases.items():
            if keyword in user_text:
                return node_name
        return None

    # --- NEW: Start Threaded Navigation ---
    def start_navigation(self, user_input):
        logging.info("Starting real-time visual navigation thread...")
        destination = self.parse_destination(user_input)
        
        if not destination:
            return f"I'm sorry, I didn't recognize that destination. I heard you say: {user_input}"
            
        # Stop existing thread if running to prevent conflicting directions
        if self.navigation_active:
            self.navigation_active = False
            if self.navigation_thread and self.navigation_thread.is_alive():
                self.navigation_thread.join(timeout=2.0)
        
        self.utils.convert_and_play_speech(f"Navigating to {destination}. Please scan the area to find our starting location.")
        
        self.navigation_active = True
        self.navigation_paused = False
        self.current_destination = destination
        
        # Launch CV and Routing in the background
        self.navigation_thread = threading.Thread(target=self._navigation_worker, args=(destination,), daemon=True)
        self.navigation_thread.start()
        
        return "" # Suppress returning string since TTS is handled
        
    # --- NEW: Navigation Background Worker ---
    def _navigation_worker(self, destination):
        buffer_size = 25
        node_history = deque(maxlen=buffer_size)
        last_known_node = None
        visited_nodes = []
        current_instruction = "Scanning for location..."
        last_spoken_instruction = ""
        aruco_history = deque(maxlen=buffer_size)
        yolo_heading_history = deque(maxlen=buffer_size)
        stable_yolo_heading_class = None
        
        MARKER_SIZE = 0.1615
        DISTANCE_THRESHOLD = 2.8
        ARUCO_NODE_MAP = {
            0:"Reception", 4:"MTS-01", 8:"Machine Vision Lab", 12:"CAD-CAM Lab",
            14:"Circuit Design Lab", 16:"Robotics Lab", 18:"Industrial Automation Lab",
            20:"Outside CAD-CAM Lab", 22:"Outside Machine Vision Lab", 24:"Outside MTS-01",
            26:"Outside Circuit Design Lab", 42:"Outside Robotics Lab", 67:"Outside Industrial Automation Lab"
        }

        # Updated YOLO maps based on new class names
        YOLO_NODE_MAP = {
            "reception": "Reception",
            "Entrance": "Entrance",
            "entrance_forward": "Entrance",
            "entrance_left": "Entrance",
            "entrance_right": "Entrance",
            "cad-cam lab": "CAD-CAM Lab",
            "Circuit Design Lab": "Circuit Design Lab",
            "cdl": "Circuit Design Lab", 
            "Machine Vision Lab": "Machine Vision Lab",
            "MTS 1": "MTS-01",
            "Outside CAD-CAM Lab": "Outside CAD-CAM Lab",
            "Outside Circuit Design Lab": "Outside Circuit Design Lab",
            "outside industrial": "Outside Industrial Automation Lab",
            "Outside Machine Vision Lab": "Outside Machine Vision Lab",
            "Outside MTS 01": "Outside MTS-01",
            "Outside Robotics Lab": "Outside Robotics Lab",
            "Robotics Lab": "Robotics Lab"
        }

        # Vector mappings derived from self.pos coordinates
        YOLO_HEADING_MAP = {
            "entrance_forward": np.array([0, 1]),
            "entrance_left": np.array([-1, 0]),
            "entrance_right": np.array([1, 0]),
            "left_corridor_forward": np.array([-1, 0]),
            "left_corridor_reverse": np.array([1, 0]),
            "right_corridor_forward": np.array([1, 0]),
            "right_corridor_reverse": np.array([-1, 0])
        }

        orientation_map = {
            "CAD-CAM Lab": np.array([0, -1]),
            "Circuit Design Lab": np.array([0, -1]),
            "Machine Vision Lab": np.array([0, 1]), 
            "Reception": np.array([0, 1]),
            "Entrance": np.array([0, -1]),
            "MTS-01": np.array([0, 1]),
            "Robotics Lab": np.array([0, -1]),
        }

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        camera_matrix = np.array([[643.022, 0., 343.447], [0., 642.977, 243.191], [0., 0., 1.]], dtype=float)
        dist_coeffs = np.array([[-0.517, 0.120, -0.002, -0.004, 0.335]])
        global_heading = None

        # Replaces the `while True:` loop. Will cleanly exit if cancelled.
        while self.navigation_active:
            if self.navigation_paused:
                time.sleep(0.5)
                continue
                
            with self.sharedState.frame_lock:
                if self.sharedState.current_frame is None:
                    time.sleep(0.1) 
                    continue
                frame = self.sharedState.current_frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_current_node = "Unknown"
            closest_distance = float('inf')
            closest_marker_id = None
            yolo_detected_heading = None # Track visual heading per frame

            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    distance = np.linalg.norm(tvecs[i][0])
                    if distance < DISTANCE_THRESHOLD and distance < closest_distance:
                        if marker_id in ARUCO_NODE_MAP:
                            closest_distance = distance
                            closest_marker_id = marker_id

            # --- CORRECTION: Always run YOLO to capture the visual heading ---
            results = self.yolo_nav_model(frame, verbose=False)
            top1_index = results[0].probs.top1
            class_name = self.yolo_nav_model.names[top1_index]
            
            # --- NEW: INSTANT START + STABLE YOLO HEADING LOGIC ---
            # 1. Add current class to buffer
            if class_name in YOLO_HEADING_MAP:
                yolo_heading_history.append(class_name)
            else:
                yolo_heading_history.append("Unknown")

            # 2. Check for stability
            if stable_yolo_heading_class is None:
                # FIRST TIME: Lock immediately onto the first valid heading
                if class_name in YOLO_HEADING_MAP:
                    stable_yolo_heading_class = class_name
                    # Pre-fill the buffer with this class so it is instantly stable
                    yolo_heading_history.extend([class_name] * buffer_size)
            else:
                # SUBSEQUENT TIMES: Require buffer consistency to change heading
                if len(yolo_heading_history) == buffer_size:
                    most_common_yolo = Counter(yolo_heading_history).most_common(1)[0]
                    # Must be 100% consistent across the buffer to change directions
                    if most_common_yolo[1] == buffer_size and most_common_yolo[0] != "Unknown":
                        stable_yolo_heading_class = most_common_yolo[0]

            # 3. Map the stable class to the actual vector
            if stable_yolo_heading_class is not None:
                yolo_detected_heading = YOLO_HEADING_MAP[stable_yolo_heading_class]

            # # 2. Determine the raw_current_node
            # if closest_marker_id is not None:
            #     # Localize based on ArUco if present (e.g., when in a corridor)
            #     raw_current_node = ARUCO_NODE_MAP[closest_marker_id]
            # else:
            #     # Localize based on YOLO if no ArUco is found
            #     if "corridor" in class_name:
            #         raw_current_node = "Unknown"  # Corridors are only for direction
            #     else:
            #         raw_current_node = YOLO_NODE_MAP.get(class_name, "Unknown")

            # ==========================================
            # --- 2. SUPER-RESILIENT ARUCO COUNTING ---
            # Change back if no work
            aruco_history.append(closest_marker_id)
            stable_aruco_id = None
            valid_ids = [m_id for m_id in aruco_history if m_id is not None]
            
            if len(valid_ids) > 0:
                most_common_aruco = Counter(valid_ids).most_common(1)[0]
                # If we see it 3 times out of the last 25 frames, trust it
                if most_common_aruco[1] >= 3:
                    stable_aruco_id = most_common_aruco[0]

            if stable_aruco_id is not None:
                raw_current_node = ARUCO_NODE_MAP.get(stable_aruco_id, "Unknown")
            else:
                if "corridor" in class_name:
                    raw_current_node = "Unknown" 
                else:
                    raw_current_node = YOLO_NODE_MAP.get(class_name, "Unknown")
            # ===========================================

            node_history.append(raw_current_node)
            most_common = Counter(node_history).most_common(1)
            stable_node = most_common[0][0] if most_common else "Unknown"

            if stable_node != "Unknown" and stable_node in self.pos:
                if last_known_node is not None and stable_node != last_known_node:
                    if nx.shortest_path_length(self.nav_graph, last_known_node, stable_node) > 2:
                        stable_node = last_known_node

            if stable_node != "Unknown":
                if visited_nodes and visited_nodes[-1] != stable_node:
                    visited_nodes.append(stable_node)

            if stable_node != "Unknown" and stable_node in self.pos:
                
                # --- NEW FIX: Update heading continuously from stabilized YOLO ---
                if yolo_detected_heading is not None:
                    global_heading = yolo_detected_heading

                if stable_node != last_known_node:
                    if not visited_nodes: visited_nodes.append(stable_node)
                    
                    if last_known_node is not None:
                        for i in range(len(visited_nodes)-1):
                            if not visited_nodes[len(visited_nodes) - i-1].startswith("Outside "):
                                break
                        p1 = np.array(self.pos[last_known_node])
                        p2 = np.array(self.pos[stable_node])
                        move_vec = p2 - p1
                        
                        # Only override YOLO heading with physical movement vector if significant
                        if np.linalg.norm(move_vec) > 0.1:
                            global_heading = move_vec / np.linalg.norm(move_vec)
                    else:
                        if global_heading is None: 
                            global_heading = orientation_map.get(stable_node, np.array([1, 0]))

                    if stable_node == destination:
                        current_instruction = f"You have arrived at {destination}."
                    else:
                        route = self.navigate_with_turns(stable_node, destination, global_heading)
                        if route and route[0] != "Path not found.":
                            current_instruction = route[0]
                        else:
                            current_instruction = "Recalculating."
                    
                    last_known_node = stable_node

                else:
                    # --- NEW FIX: Recalculate route if heading changes while standing still ---
                    if stable_node != destination:
                        route = self.navigate_with_turns(stable_node, destination, global_heading)
                        if route and route[0] != "Path not found.":
                            current_instruction = route[0]

            if current_instruction != last_spoken_instruction and current_instruction != "Scanning for location...":
                logging.info(f"Speaking: {current_instruction}")
                self.utils.convert_and_play_speech(current_instruction)
                last_spoken_instruction = current_instruction
                
                if stable_node == destination:
                    self.navigation_active = False # Kill the thread cleanly
                    break
        
    # def process_video_analysis(self, user_input):
    #     """
    #     Handle detailed scene analysis using video recording.
        
    #     Args:
    #         user_input: User's query text
            
    #     Returns:
    #         str: Detailed scene description based on video
    #     """
    #     logging.info("Using Video Analysis...")
    #     output = self.agents.activity_detection(
    #         user_input, 
    #         self.utils.record_video_from_camera,
    #         self.utils.analyze_video_with_prompt,
    #         self.endpoint_url
    #     )
    #     return output
    def cleanup(self):
        """Safely release all hardware and resources."""
        logging.info("Running cleanup tasks...")
        
        # Clean up haptics
        if hasattr(self, 'haptics'):
            try:
                # Assuming your HapticController has a cleanup method or you can send a "STOP" command
                self.haptics.process_command("BLOCKED: STOP / TURN AROUND") 
                # self.haptics.cleanup() 
            except Exception as e:
                logging.error(f"Error cleaning up haptics: {e}")

        # Release the camera
        if hasattr(self, 'vidStream') and self.vidStream is not None:
            self.vidStream.release()
            
        cv2.destroyAllWindows()
        logging.info("Cleanup complete.")
    
    def run(self):
        """
        Main application loop that listens for commands and processes them.
        """
        logging.info("Welcome to SceneScribe! Ready to listen for commands.")
        try:
            while True:
                try:
                    logging.info("Waiting for voice command...")
                    user_input = self.utils.wait_for_listen_command()
                    
                    if not user_input:
                        logging.info("No input detected, trying again...")
                        continue
                        
                    if user_input:
                        user_input = re.sub(r'(?i)[^\w\s]*\bstop\s+listening\b[^\w\s]*', '', user_input)
                        user_input = user_input.strip()

                    logging.info(f"Command received: {user_input}")
                    start_time = time.time()
                    
                    if user_input.lower() in ["exit,","exit.", "quit.", "goodbye."]:
                        logging.info("Exiting SceneScribe. Goodbye!")
                        self.navigation_active = False # Clean up background thread
                        break
                    
                    if self.educational_mode:
                        print("Using Educational Mode")
                    else:
                        if not check_network_connection():
                            output = "Sorry not connected to internet"
                            self.utils.convert_and_play_speech(output)
                        else:
                            classification = self.utils.classify_input(
                                user_input,
                                self.loaded_model,
                                self.loaded_vectorizer
                            )
                            
                            # --- INTERRUPT LOGIC HUB ---
                            if classification == "Scene Explanation":
                                was_navigating = False
                                
                                # 1. Pause active navigation
                                if self.navigation_active and not self.navigation_paused:
                                    logging.info("Pausing navigation thread for scene explanation...")
                                    self.navigation_paused = True
                                    was_navigating = True
                                    self.utils.convert_and_play_speech("Navigation paused. Let me check the scene.")
                                
                                # 2. Answer the user's question
                                output = self.process_scene_explanation(user_input)
                                if output:
                                    print(f"Output: {output}")
                                    self.utils.convert_and_play_speech(output)
                                    
                                # 3. Resume navigation
                                if was_navigating:
                                    logging.info("Resuming background navigation thread.")
                                    self.utils.convert_and_play_speech("Resuming navigation.")
                                    self.navigation_paused = False
                            else:
                                # 4. Starting a brand new navigation intent
                                output = self.start_navigation(user_input)
                                if output:
                                    print(f"Output: {output}")
                                    self.utils.convert_and_play_speech(output)
                        
                    end_time = time.time()
                    print(f"Time Taken: {end_time - start_time}")
                except KeyboardInterrupt:
                    logging.info("\nProgram interrupted by user. Exiting...")
                    self.navigation_active = False
                    break
                except Exception as e:
                    logging.info(f"Error in main loop: {e}")
                    error_msg = f"I'm sorry, I encountered an error: {str(e)}"
                    try:
                        self.utils.convert_and_play_speech(error_msg)
                    except:
                        pass
        finally:
            self.cleanup()


# def main():
#     """
#     Entry point for the SceneScribe application.
#     """
#     # Use language from command line arg if provided, otherwise default to English
#     language = sys.argv[1] if len(sys.argv) > 1 else "English"
    
#     # Create and run SceneScribe instance
#     app = SceneScribe(language=language)
#     app.run()

