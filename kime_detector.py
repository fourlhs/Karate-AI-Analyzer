import cv2
import mediapipe as mp
import numpy as np
import time

"""
Kime Detector (Standalone Module)
---------------------------------
Focuses exclusively on detecting 'Kime' (Snap) in Karate techniques.
Uses velocity and acceleration thresholds to identify sudden stops.
"""

# --- CONFIGURATION ---
# Thresholds calibrated for high-intensity training
# Speed must exceed this value (pixels/sec) before the stop
MIN_SPEED_FOR_KIME = 1000.0   
# Acceleration must be lower than this (negative value) to count as a snap
MIN_ACCEL_FOR_KIME = -9000.0     

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# State variables
prev_time = 0
prev_wrist = np.array([0.0, 0.0, 0.0]) 
prev_speed = 0
acceleration = 0

# Feedback triggers
kime_trigger_time = 0
feedback_text = ""
feedback_color = (0, 0, 255)

# Initialize Camera
cap = cv2.VideoCapture(0)
print("[INFO] Kime Detector Module Started.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror and convert image
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get 3D Coordinates of Right Wrist
        # Z-axis is weighted to emphasize forward motion (depth)
        current_wrist = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * w 
        ])
        
        curr_time = time.time()
        dt = curr_time - prev_time
        
        if dt > 0:
            # Calculate Velocity (3D Euclidean Distance / Time)
            distance = np.linalg.norm(current_wrist - prev_wrist)
            raw_speed = distance / dt 
            
            # Strong Smoothing (Low-Pass Filter)
            # 70% previous value, 30% new value to reduce camera jitter
            current_speed = 0.7 * prev_speed + 0.3 * raw_speed
            
            # Noise Gate: Ignore minor movements
            if current_speed < 50: 
                current_speed = 0

            # Calculate Acceleration
            acceleration = (current_speed - prev_speed) / dt

            # --- KIME LOGIC ---
            # Trigger if high speed is followed by sudden deceleration
            if prev_speed > MIN_SPEED_FOR_KIME and acceleration < MIN_ACCEL_FOR_KIME:
                kime_trigger_time = curr_time
                feedback_text = "KIME!"
                feedback_color = (0, 255, 0) # Green

            # Update state
            prev_speed = current_speed
            prev_wrist = current_wrist
            prev_time = curr_time

        # --- VISUALIZATION ---
        
        # Display Kime Feedback
        if curr_time - kime_trigger_time < 0.5:
            cv2.putText(image, feedback_text, (50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, feedback_color, 4, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 10)

        # Debug Data Panel (Top Left)
        cv2.rectangle(image, (0,0), (320, 100), (0,0,0), -1)
        
        # Speed Indicator
        speed_color = (0, 255, 0) if current_speed > MIN_SPEED_FOR_KIME else (255, 255, 255)
        cv2.putText(image, f"Speed: {int(current_speed)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        
        # Acceleration Indicator
        accel_text = f"Accel: {int(acceleration)}"
        accel_color = (0, 255, 255) # Yellow
        if acceleration < MIN_ACCEL_FOR_KIME: 
            accel_color = (0, 0, 255) # Red (Triggered)
        
        cv2.putText(image, accel_text, (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, accel_color, 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Kime Detector (Standalone)', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()