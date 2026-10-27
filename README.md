import os
import cv2
import yaml
import json
import numpy as np
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

class TrafficAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model = YOLO(os.getenv("DETECTION_MODEL_PATH", "yolo11n.pt"))
        self.speed_limit = float(os.getenv("SPEED_LIMIT_KMH", "80"))
        self.webhook_url = os.getenv("GIS_ALERT_WEBHOOK_URL")
        
        # Build out spatial transform homography matrix from camera specs
        src_pts = np.float32(self.config['calibration']['pixel_points'])
        dst_pts = np.float32(self.config['calibration']['real_world_meters'])
        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Vehicle track storage dict
        self.vehicle_tracks = {}

    def calculate_speed(self, track_id, current_centroid, fps=30):
        if track_id not in self.vehicle_tracks:
            self.vehicle_tracks[track_id] = {"centroids": [current_centroid], "timestamps": [0.0]}
            return 0.0
            
        track = self.vehicle_tracks[track_id]
        track["centroids"].append(current_centroid)
        
        if len(track["centroids"]) < 2:
            return 0.0
            
        # Extract last two tracking elements to parse instantaneous speed vector
        p1 = np.array([[[track["centroids"][-2][0], track["centroids"][-2][1]]]], dtype=np.float32)
        p2 = np.array([[[track["centroids"][-1][0], track["centroids"][-1][1]]]], dtype=np.float32)
        
        # Transform positions into true meter space metrics using matrix arrays
        m1 = cv2.perspectiveTransform(p1, self.H)[0][0]
        m2 = cv2.perspectiveTransform(p2, self.H)[0][0]
        
        distance_meters = np.linalg.norm(m2 - m1)
        speed_mps = distance_meters * fps
        speed_kmh = speed_mps * 3.6
        return round(speed_kmh, 2)

    def validate_direction(self, track_id, current_centroid):
        # Entry-Exit validator baseline logic framework mapping
        track = self.vehicle_tracks.get(track_id)
        if not track or len(track["centroids"]) < 5:
            return "UNKNOWN"
            
        start_y = track["centroids"][0][1]
        current_y = current_centroid[1]
        
        # If moving from bottom screen to top screen against typical roadway geometry flow
        if current_y < start_y - 50: 
            return "WRONG_WAY"
        return "CORRECT"

    def dispatch_spatial_alert(self, violation_type, vehicle_class, speed):
        payload = {
            "camera_id": self.config["camera_id"],
            "violation_type": violation_type,
            "vehicle_class": vehicle_class,
            "velocity_kmh": speed,
            "anchor_location": {
                "lat": self.config["geographic_anchor"]["latitude"],
                "lng": self.config["geographic_anchor"]["longitude"]
            }
        }
        
        print(f"🚨 ALERT DISPATCHED: {violation_type} | Speed: {speed} km/h")
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json=payload, timeout=2)
            except requests.exceptions.RequestException:
                pass # Suppress network failures in pipeline simulation modes

    def process_frame(self, frame):
        # Run inference using standard tracking engines
        results = self.model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7]) # Cars, bikes, buses, trucks
        
        if not results[0].boxes.id:
            return frame
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        names = self.model.names
        
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x1, y1, x2, y2 = box
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            vehicle_class = names[cls_id]
            
            speed = self.calculate_speed(track_id, centroid)
            direction = self.validate_direction(track_id, centroid)
            
            # Action triggers for pipeline alerts execution paths
            if direction == "WRONG_WAY":
                self.dispatch_spatial_alert("WRONG_WAY", vehicle_class, speed)
            elif speed > self.speed_limit:
                self.dispatch_spatial_alert("SPEEDING", vehicle_class, speed)
                
        return frame

if __name__ == "__main__":
    # Standard initialization runner routine block execution sequence
    analyzer = TrafficAnalyzer("configs/camera_config.yaml")
    print("Edge-AI Analytics Core Started. Processing telemetry pipeline streams...")
