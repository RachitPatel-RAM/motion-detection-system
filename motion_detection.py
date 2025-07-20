import cv2
import numpy as np
import time
import datetime
import os
import csv
import pygame
from collections import deque

# Configuration
THRESHOLD_SENSITIVITY = 25  # Increased for fewer false positives
MIN_CONTOUR_AREA = 150      # Filter small movements
VIDEO_DURATION = 5
MASK_REGION = None
SIREN_SOUND = 'siren.mp3'
PRE_EVENT_SECONDS = 2       # Store 2 seconds of pre-event footage
DEBOUNCE_SECONDS = 5        # Minimum time between captures
RESOLUTION = (1280, 720)    # 720p resolution
SLOW_FPS = 10.0             # Slower FPS for speedy motions
NORMAL_FPS = 20.0           # Normal FPS
SPEED_THRESHOLD = 100       # Pixel area change to detect 'speedy' motion
MIN_OBJECT_WIDTH = 100      # Minimum valid object size
STABILITY_FRAMES = 3        # Require consecutive detections

# Output directories
SNAPSHOTS_DIR = 'snapshots'
VIDEOS_DIR = 'videos'
LOG_FILE = 'motion_log.csv'

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Initialize pygame for sound
pygame.mixer.init()
siren_sound = pygame.mixer.Sound(SIREN_SOUND)

# Initialize logging
def init_log():
    with open(LOG_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Event'])

# Log event
def log_event(event):
    with open(LOG_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), event])

# Main function
def main():
    init_log()
    
    # Initialize video capture with higher resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Video writer for recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    recording = False
    record_start_time = 0
    
    # Frame buffer for pre-event footage
    frame_buffer = deque(maxlen=int(cap.get(cv2.CAP_PROP_FPS) * PRE_EVENT_SECONDS))
    last_capture_time = 0
    stable_detections = 0
    best_contour = None

    # Initial frame for differencing
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

    prev_area = 0  # To track area change for speed detection

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
            
        # Add frame to buffer
        frame_buffer.append(frame2.copy())
        
        # Apply real-time sharpening for better quality
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(frame2, -1, kernel)
        
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # Frame differencing
        delta = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(delta, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Apply mask if defined
        if MASK_REGION:
            mask = np.zeros_like(thresh)
            x, y, w, h = MASK_REGION
            mask[y:y+h, x:x+w] = 1
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize motion detection variables
        motion_detected = False
        current_area = 0
        best_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                current_area += area
                
                # Track largest valid object
                if w > MIN_OBJECT_WIDTH and (best_contour is None or area > cv2.contourArea(best_contour)):
                    best_contour = contour
                    motion_detected = True
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect speedy motion
        area_change = abs(current_area - prev_area)
        is_speedy = area_change > SPEED_THRESHOLD
        fps = SLOW_FPS if is_speedy else NORMAL_FPS
        prev_area = current_area

        # Process motion if detected and debounce time passed
        if motion_detected and (time.time() - last_capture_time) > DEBOUNCE_SECONDS:
            # Play siren sound
            siren_sound.play()
            
            # Get object measurements from best contour
            x, y, w, h = cv2.boundingRect(best_contour)
            object_size = f"{w}x{h}px"
            
            # Start recording with pre-event buffer
            video_filename = os.path.join(VIDEOS_DIR, f"motion_{int(time.time())}_{'speedy' if is_speedy else 'normal'}.mp4")
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
            
            # Write pre-event frames to video
            for buffered_frame in frame_buffer:
                video_writer.write(buffered_frame)
                
            # Take high-quality snapshot
            snapshot_filename = os.path.join(SNAPSHOTS_DIR, f"snapshot_{int(time.time())}_{'speedy' if is_speedy else 'normal'}.jpg")
            cv2.imwrite(snapshot_filename, sharpened, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            log_event(f"Motion detected - Size: {object_size}, Snapshot: {snapshot_filename}, Video: {video_filename}, Speedy: {is_speedy}")
            
            recording = True
            record_start_time = time.time()
            last_capture_time = time.time()

        # Continue recording if in progress
        if recording:
            video_writer.write(frame2)
            
            if time.time() - record_start_time > VIDEO_DURATION:
                video_writer.release()
                recording = False

        # Update previous frame
        gray1 = gray2

        # Display the frame
        cv2.imshow('Motion Detection', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if recording:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()