import cv2
import mediapipe as mp
import numpy as np
import time

"""
Speed Tracker Module
--------------------
Calculates the speed of the right wrist in real-time.
- Bar turns RED when speed >= 850.
- Scale adjusted for high-speed punches (0-1200 range).
"""

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# State variables
prev_time = 0
prev_wrist = np.array([0.0, 0.0, 0.0]) 

punch_speed = 0
max_punch_speed = 0
last_punch_time = 0

# Threshold for color change
SPEED_THRESHOLD = 850

# Initialize Camera
cap = cv2.VideoCapture(0)
print(f"[INFO] Speed Tracker Started. High Speed Threshold: {SPEED_THRESHOLD}")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror view
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 1. Get 3D Coordinates
        current_wrist = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z 
        ])
        
        # 2. Calculate Time Delta
        curr_time = time.time()
        dt = curr_time - prev_time
        
        # 3. Calculate Speed
        distance = np.linalg.norm(current_wrist - prev_wrist)
        
        if dt > 0:
            instant_speed = (distance / dt) * 100 
            
            if instant_speed < 2:
                instant_speed = 0
            
            punch_speed = 0.7 * punch_speed + 0.3 * instant_speed
            
            if punch_speed > max_punch_speed:
                max_punch_speed = punch_speed
                last_punch_time = curr_time

        if curr_time - last_punch_time > 2.0:
            max_punch_speed = 0

        prev_time = curr_time
        prev_wrist = current_wrist

        # --- VISUALIZATION ---
        
        # Background of Speed Bar
        cv2.rectangle(image, (50, 150), (80, 400), (0, 0, 0), 2)
        
        # Dynamic Color Logic
        # Default: Yellow (0, 255, 255)
        # Over 850: Red (0, 0, 255)
        bar_color = (0, 255, 255) 
        if punch_speed >= SPEED_THRESHOLD:
            bar_color = (0, 0, 255)

        # Bar Height Scaling
        bar_height = int(np.clip(punch_speed * 0.3, 0, 250))
        
        # Draw Fill
        cv2.rectangle(image, (50, 400 - bar_height), (80, 400), bar_color, -1)
        
        # Add a Marker line for the 850 threshold
        threshold_y = 400 - int(SPEED_THRESHOLD * 0.3)
        if 150 < threshold_y < 400:
            cv2.line(image, (45, threshold_y), (85, threshold_y), (0, 0, 255), 2)

        # Max Speed Text
        cv2.putText(image, f"MAX: {int(max_punch_speed)}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3) 
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Karate Speed Tracker', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()