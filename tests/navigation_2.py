#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import networkx as nx
from collections import deque, Counter
from ultralytics import YOLO
import time

class VideoNavTester:
    def __init__(self, video_path, destination, yolo_model_path):
        self.video_path = video_path
        self.destination = destination
        
        print("Loading YOLO Model...")
        self.yolo_model = YOLO(yolo_model_path, task='classify')

        self.setup_graph()
        self.setup_maps()
        
    def setup_graph(self):
        # 1. Define Positions
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

        # 2. Build Navigation Graph
        self.nav_graph = nx.Graph()
        edges = [
            ("Entrance", "Reception", 10),
            ("Reception", "Outside CAD-CAM Lab", 6),
            ("Outside Circuit Design Lab", "Circuit Design Lab", 5),
            ("Outside Machine Vision Lab", "Machine Vision Lab", 5),
            ("Outside CAD-CAM Lab", "CAD-CAM Lab", 5),
            ("Outside CAD-CAM Lab", "Outside Machine Vision Lab", 6),
            ("Outside Machine Vision Lab", "Outside Circuit Design Lab", 6),
            ("Reception", "Outside MTS-01", 6),
            ("Outside MTS-01", "Outside Industrial Automation Lab", 6),
            ("Outside MTS-01", "MTS-01", 5),
            ("Outside Industrial Automation Lab", "Outside Robotics Lab", 10),
            ("Outside Robotics Lab", "Robotics Lab", 5)
        ]
        self.nav_graph.add_weighted_edges_from(edges)

    def setup_maps(self):
        self.ARUCO_NODE_MAP = {
            0:"Reception", 4:"MTS-01", 8:"Machine Vision Lab", 12:"CAD-CAM Lab",
            14:"Circuit Design Lab", 16:"Robotics Lab", 18:"Industrial Automation Lab",
            20:"Outside CAD-CAM Lab", 22:"Outside Machine Vision Lab", 24:"Outside MTS-01",
            26:"Outside Circuit Design Lab", 42:"Outside Robotics Lab", 67:"Outside Industrial Automation Lab"
        }

        self.YOLO_NODE_MAP = {
            "reception": "Reception", "Entrance": "Entrance", "entrance_forward": "Entrance",
            "entrance_left": "Entrance", "entrance_right": "Entrance", "cad-cam lab": "CAD-CAM Lab",
            "Circuit Design Lab": "Circuit Design Lab", "Machine Vision Lab": "Machine Vision Lab",
            "MTS 1": "MTS-01", "Outside CAD-CAM Lab": "Outside CAD-CAM Lab",
            "Outside Circuit Design Lab": "Outside Circuit Design Lab",
            "outside industrial": "Outside Industrial Automation Lab",
            "Outside Machine Vision Lab": "Outside Machine Vision Lab",
            "Outside MTS 01": "Outside MTS-01", "Outside Robotics Lab": "Outside Robotics Lab",
            "Robotics Lab": "Robotics Lab"
        }

        self.YOLO_HEADING_MAP = {
            "entrance_forward": np.array([0, 1]), "entrance_left": np.array([-1, 0]),
            "entrance_right": np.array([1, 0]), 
            # "left_corridor_forward": np.array([-1, 0]),
            # "left_corridor_reverse": np.array([1, 0]), "right_corridor_forward": np.array([1, 0]),
            # "right_corridor_reverse": np.array([-1, 0])
        }

        self.orientation_map = {
            "CAD-CAM Lab": np.array([0, -1]), "Circuit Design Lab": np.array([0, -1]),
            "Machine Vision Lab": np.array([0, 1]), "Reception": np.array([0, 1]),
            "Entrance": np.array([0, -1]), "MTS-01": np.array([0, 1]),
            "Robotics Lab": np.array([0, -1]),
        }

        # ArUco parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create() if hasattr(aruco, 'DetectorParameters_create') else aruco.DetectorParameters()
        self.camera_matrix = np.array([[643.022, 0., 343.447], [0., 642.977, 243.191], [0., 0., 1.]], dtype=float)
        self.dist_coeffs = np.array([[-0.517, 0.120, -0.002, -0.004, 0.335]])
        self.MARKER_SIZE = 0.1615
        self.DISTANCE_THRESHOLD = 2.8

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

            if v == end and v != "Reception":
                if end == "Entrance":
                    instructions.append(f"Go forward 5 steps, then {turn} and walk {steps} steps to {v}.")
                else:
                    instructions.append(f"Walk forward 3 steps, then {turn} and walk {steps} steps to {v}.")
            else:
                instructions.append(f"At {u}, {turn}, walk {steps} steps to {v}.")
                
            curr_heading = unit_target

        return instructions

    def draw_overlays(self, frame, current_instruction, stable_node, yolo_class, closest_marker_id):
        # Create a dark semi-transparent rectangle at the top for better text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw Status Texts
        cv2.putText(frame, f"Destination: {self.destination}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Current Node: {stable_node}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Instruction: {current_instruction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display raw sensor info
        sensor_info = f"YOLO: {yolo_class} | ArUco ID: {closest_marker_id if closest_marker_id is not None else 'None'}"
        cv2.putText(frame, sensor_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return

        buffer_size = 30
        node_history = deque(maxlen=buffer_size)
        last_known_node = None
        visited_nodes = []
        current_instruction = "Scanning for location..."
        global_heading = None

        yolo_heading_history = deque(maxlen=buffer_size)
        stable_yolo_heading_class = None

        print("\nStarting Video Navigation Loop...")
        print("Press 'q' in the video window to quit.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # Simulate 90-deg rotation if your camera originally required it
            # Uncomment if the test video also needs to be rotated
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_current_node = "Unknown"
            closest_distance = float('inf')
            closest_marker_id = None
            yolo_detected_heading = None

            # --- ARUCO DETECTION ---
            corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None:
                # Draw markers on the frame
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Handling for different OpenCV versions
                if hasattr(aruco, 'estimatePoseSingleMarkers'):
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.MARKER_SIZE, self.camera_matrix, self.dist_coeffs)
                else:
                    # Fallback for newer OpenCV versions where estimatePoseSingleMarkers is removed
                    obj_points = np.array([[-self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
                                           [self.MARKER_SIZE/2, self.MARKER_SIZE/2, 0],
                                           [self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
                                           [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]])
                    tvecs = []
                    for corner in corners:
                        success, _, tvec = cv2.solvePnP(obj_points, corner, self.camera_matrix, self.dist_coeffs)
                        if success: tvecs.append([tvec])

                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    distance = np.linalg.norm(tvecs[i][0])
                    if distance < self.DISTANCE_THRESHOLD and distance < closest_distance:
                        if marker_id in self.ARUCO_NODE_MAP:
                            closest_distance = distance
                            closest_marker_id = marker_id

            # --- YOLO DETECTION ---
            results = self.yolo_model(frame, verbose=False)
            top1_index = results[0].probs.top1
            class_name = self.yolo_model.names[top1_index]

            # 1. Add current class to buffer if it's a heading class
            if class_name in self.YOLO_HEADING_MAP:
                yolo_heading_history.append(class_name)
            else:
                yolo_heading_history.append("Unknown")

            # 2. Check for stability across the buffer
            if len(yolo_heading_history) == buffer_size:
                most_common_yolo = Counter(yolo_heading_history).most_common(1)[0]
                if most_common_yolo[1] == buffer_size and most_common_yolo[0] != "Unknown":
                    stable_yolo_heading_class = most_common_yolo[0]

            # 3. Map the stable class to the actual vector
            if stable_yolo_heading_class is not None:
                yolo_detected_heading = self.YOLO_HEADING_MAP[stable_yolo_heading_class]

            # --- LOCALIZATION LOGIC ---
            if closest_marker_id is not None:
                raw_current_node = self.ARUCO_NODE_MAP[closest_marker_id]
            else:
                if "corridor" in class_name:
                    raw_current_node = "Unknown"
                else:
                    raw_current_node = self.YOLO_NODE_MAP.get(class_name, "Unknown")

            node_history.append(raw_current_node)
            most_common = Counter(node_history).most_common(1)
            stable_node = most_common[0][0] if most_common else "Unknown"

            # Node Jumping Protection
            if stable_node != "Unknown" and stable_node in self.pos:
                if last_known_node is not None and stable_node != last_known_node:
                    if nx.shortest_path_length(self.nav_graph, last_known_node, stable_node) > 2:
                        stable_node = last_known_node

            if stable_node != "Unknown" and stable_node in self.pos:
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
                        
                        if np.linalg.norm(move_vec) > 0.1:
                            global_heading = move_vec / np.linalg.norm(move_vec)
                        elif yolo_detected_heading is not None:
                            global_heading = yolo_detected_heading
                    else:
                        if yolo_detected_heading is not None:
                            global_heading = yolo_detected_heading
                        else:
                            global_heading = self.orientation_map.get(stable_node, np.array([1, 0]))

                    if stable_node == self.destination:
                        current_instruction = f"You have arrived at {self.destination}."

                        # self.draw_overlays(frame, current_instruction, stable_node, class_name, closest_marker_id)
                        # cv2.imshow("Navigation Test", frame)
                        # print(f"\nSUCCESS: {current_instruction}")
                        # print("Killing navigation thread...")
                        
                        # Wait 3 seconds so you can see the final instruction on the video
                        cv2.waitKey(3000) 
                        
                        # Break out of the while loop to end the script
                        break
                    else:
                        route = self.navigate_with_turns(stable_node, self.destination, global_heading)
                        if route and route[0] != "Path not found.":
                            current_instruction = route[0]
                        else:
                            current_instruction = "Recalculating."
                    
                    last_known_node = stable_node
                else:
                    # --- Recalculate route if heading changes while standing still ---
                    if stable_node != self.destination:
                        route = self.navigate_with_turns(stable_node, self.destination, global_heading)
                        if route and route[0] != "Path not found.":
                            current_instruction = route[0]

            # Draw everything onto the frame
            self.draw_overlays(frame, current_instruction, stable_node, class_name, closest_marker_id)

            # Display the result
            cv2.imshow("Navigation Test", frame)

            # Control playback speed (Wait ~30ms for ~30 FPS). Press 'q' to quit.
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Quit command received.")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # ==========================================
    # SET YOUR TEST VARIABLES HERE
    # ==========================================
    VIDEO_FILE = "/home/scenescribe/dataset/test_entrace_to_cdl/test_entrace_to_cdl_1777116907.mp4" 
    YOLO_MODEL = "/home/scenescribe/scenescribe_exp/models/final.engine"
    TARGET_DESTINATION = "Circuit Design Lab"
    
    tester = VideoNavTester(
        video_path=VIDEO_FILE,
        destination=TARGET_DESTINATION,
        yolo_model_path=YOLO_MODEL
    )
    
    tester.run()