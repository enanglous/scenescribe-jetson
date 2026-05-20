import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    """
    Instance Segmentation using YOLOv11-seg from Ultralytics
    """
    def __init__(self, model_size='small', model_path='/home/scenescribe/scenescribe_exp/models', conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        self.model_path = model_path
        print(f"Using device: {self.device} for instance segmentation")
        
        # Map model size to SEGMENTATION model names (-seg)
        model_map = {
            'nano': 'yolo26n-seg.engine',
            'small': 'yolo26s-seg.engine',
            'medium': 'yolo26m-seg.engine',
            'large': 'yolo26l-seg.engine',
            'extra': 'yolo26x-seg.engine'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # Load model
        try:
            self.model = YOLO(f"{model_path}/{model_name}")
            print(f"Loaded YOLOv11 Segmentation {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = YOLO(model_name)
        
        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 100
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        detections = []
        annotated_image = image.copy()
        orig_h, orig_w = image.shape[:2]
        
        if track:
            results = self.model.track(image, verbose=False, device=self.device, persist=True)
        else:
            results = self.model.predict(image, verbose=False, device=self.device)
        
        if track:
            # Clean up old trajectories
            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None 
                              for bbox in predictions.boxes if bbox.id is not None]:
                    del self.tracking_trajectories[id_]
                    
        for predictions in results:
            if predictions is None or predictions.boxes is None:
                continue
            
            scores = predictions.boxes.conf
            classes = predictions.boxes.cls
            bbox_coords = predictions.boxes.xyxy
            
            # Extract tracking IDs if tracking is enabled
            if track and hasattr(predictions.boxes, 'id') and predictions.boxes.id is not None:
                ids = predictions.boxes.id
            else:
                ids = [None] * len(scores)
                
            # Process each detection AND its corresponding mask
            for i, (score, class_id, bbox_coord, id_) in enumerate(zip(scores, classes, bbox_coords, ids)):
                xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                
                # --- NEW: MASK PROCESSING ---
                # Create a blank binary mask for this specific object
                obj_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                
                if predictions.masks is not None:
                    # Get the polygon coordinates for this mask
                    polygon = predictions.masks.xy[i]
                    if len(polygon) > 0:
                        # Fill the polygon on our blank mask
                        cv2.fillPoly(obj_mask, [np.int32(polygon)], 1)
                        
                        # (Optional) Draw a semi-transparent colored mask on the annotated image
                        color = (0, 255, 0) # Green mask
                        colored_mask = np.zeros_like(annotated_image)
                        colored_mask[obj_mask == 1] = color
                        cv2.addWeighted(annotated_image, 1.0, colored_mask, 0.4, 0, annotated_image)
                
                # Add to detections list (Now returns 5 items!)
                detections.append([
                    [xmin, ymin, xmax, ymax],  # bbox
                    float(score),              # confidence score
                    int(class_id),             # class id
                    int(id_) if id_ is not None else None, # object id
                    obj_mask                   # The binary segmentation mask!
                ])
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 225), 2)
                
                # Add label
                id_text = f"ID:{int(id_)} " if id_ is not None else ""
                label = f"{id_text}{predictions.names[int(class_id)]} {float(score):.2f}"
                cv2.putText(annotated_image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Tracking lines
                if id_ is not None:
                    centroid_x = (xmin + xmax) / 2
                    centroid_y = (ymin + ymax) / 2
                    if int(id_) not in self.tracking_trajectories:
                        self.tracking_trajectories[int(id_)] = deque(maxlen=10)
                    self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
            
            # Draw trajectories
            if track:
                for id_, trajectory in self.tracking_trajectories.items():
                    for i in range(1, len(trajectory)):
                        thickness = int(2 * (i / len(trajectory)) + 1)
                        cv2.line(annotated_image, 
                                (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                                (int(trajectory[i][0]), int(trajectory[i][1])), 
                                (255, 255, 255), thickness)
                                
        return annotated_image, detections
    
    def get_class_names(self):
        return self.model.names
