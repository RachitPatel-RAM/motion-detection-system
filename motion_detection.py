import cv2
import numpy as np
import time
import datetime
import os
from collections import deque

class MotionDetector:
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            "threshold_sensitivity": 25,
            "min_contour_area": 150,
            "video_duration": 5,
            "pre_event_seconds": 2,
            "debounce_seconds": 5,
            "resolution": (1280, 720),
            "slow_fps": 10.0,
            "normal_fps": 20.0,
            "speed_threshold": 100,
            "min_object_width": 100,
            "stability_frames": 3,
            "snapshots_dir": 'snapshots',
            "videos_dir": 'videos',
            "roi": None  # Region of interest (x, y, width, height)
        }
        
        # Override with user config if provided
        if config:
            self.config.update(config)
            
        # Create output directories
        os.makedirs(self.config["snapshots_dir"], exist_ok=True)
        os.makedirs(self.config["videos_dir"], exist_ok=True)
        
        # Initialize video capture
        self.cap = None
        self.frame_buffer = None
        self.video_writer = None
        self.recording = False
        self.record_start_time = 0
        self.last_capture_time = 0
        self.stable_detections = 0
        self.prev_area = 0
        
        # Callbacks
        self.on_motion_detected = None
        
    def start_capture(self, camera_index=0):
        """Initialize and start the video capture"""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["resolution"][1])
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        
        # Initialize frame buffer for pre-event footage
        buffer_size = int(self.cap.get(cv2.CAP_PROP_FPS) * self.config["pre_event_seconds"])
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Get initial frame
        ret, frame1 = self.cap.read()
        if not ret:
            raise Exception("Could not read frame")
            
        self.gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        self.gray1 = cv2.GaussianBlur(self.gray1, (21, 21), 0)
        
        return True
        
    def process_stream_frame(self, frame_bytes):
        """Process a single frame from a byte stream"""
        try:
            # Decode frame
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # Initialize on first frame
            if self.gray1 is None:
                self.frame_width = frame.shape[1]
                self.frame_height = frame.shape[0]
                self.gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.gray1 = cv2.GaussianBlur(self.gray1, (21, 21), 0)
                buffer_size = int(self.config["normal_fps"] * self.config["pre_event_seconds"])
                self.frame_buffer = deque(maxlen=buffer_size)

            # Add frame to buffer
            self.frame_buffer.append(frame.copy())

            # Apply real-time sharpening for better quality
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(frame, -1, kernel)

            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

            # Frame differencing
            delta = cv2.absdiff(self.gray1, gray2)
            thresh = cv2.threshold(delta, self.config["threshold_sensitivity"], 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

        except Exception as e:
            print(f"Error processing stream frame: {e}")

        # Apply ROI mask if defined
        if self.config["roi"]:
            mask = np.zeros_like(thresh)
            x, y, w, h = self.config["roi"]
            mask[y:y+h, x:x+w] = 1
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize motion detection variables
        motion_detected = False
        current_area = 0
        best_contour = None
        motion_data = {}

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config["min_contour_area"]:
                x, y, w, h = cv2.boundingRect(contour)
                current_area += area
                
                # Track largest valid object
                if w > self.config["min_object_width"] and (best_contour is None or area > cv2.contourArea(best_contour)):
                    best_contour = contour
                    motion_detected = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect speedy motion
        area_change = abs(current_area - self.prev_area)
        is_speedy = area_change > self.config["speed_threshold"]
        fps = self.config["slow_fps"] if is_speedy else self.config["normal_fps"]
        self.prev_area = current_area
        
        # Update previous frame
        self.gray1 = gray2
        
        # If motion detected and debounce time passed
        event_triggered = False
        if motion_detected and (time.time() - self.last_capture_time) > self.config["debounce_seconds"]:
            # Get object measurements from best contour
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # Prepare motion data
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            motion_data = {
                "timestamp": timestamp,
                "object_size": f"{w}x{h}px",
                "is_speedy": is_speedy,
                "area": current_area,
                "area_change": area_change,
                "bounding_box": (x, y, w, h)
            }
            
            # Take high-quality snapshot
            snapshot_filename = os.path.join(self.config["snapshots_dir"], f"snapshot_{int(time.time())}_{'speedy' if is_speedy else 'normal'}.jpg")
            cv2.imwrite(snapshot_filename, sharpened, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            motion_data["snapshot_path"] = snapshot_filename
            
            # Start recording with pre-event buffer
            video_filename = os.path.join(self.config["videos_dir"], f"motion_{int(time.time())}_{'speedy' if is_speedy else 'normal'}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (self.frame_width, self.frame_height))
            
            # Write pre-event frames to video
            for buffered_frame in self.frame_buffer:
                self.video_writer.write(buffered_frame)
                
            motion_data["video_path"] = video_filename
            self.recording = True
            self.record_start_time = time.time()
            self.last_capture_time = time.time()
            event_triggered = True
            
            # Call the callback if registered
            if self.on_motion_detected:
                self.on_motion_detected(motion_data)

        # Continue recording if in progress
        if self.recording:
            self.video_writer.write(frame)
            
            if time.time() - self.record_start_time > self.config["video_duration"]:
                self.video_writer.release()
                self.recording = False
                
        return frame, motion_detected, motion_data if event_triggered else None
        
    def stop(self):
        """Stop the video capture and release resources"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        cv2.destroyAllWindows()
        
    def get_frame_jpeg(self):
        """Get the current frame as JPEG for streaming"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Process for display
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
