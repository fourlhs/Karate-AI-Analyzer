import cv2
import mediapipe as mp
import numpy as np
import time

# --- CONFIGURATION ---
# Tuning parameters for movement detection
MIN_SPEED_FOR_KIME = 1000.0   
MIN_ACCEL_FOR_KIME = -9000.0
STANCE_LOW_THRESH = 140  # Maximum angle for a bent knee
STANCE_HIGH_THRESH = 160 # Minimum angle for a straight leg

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# State variables for Kime detection
prev_time = 0
prev_wrist = np.array([0.0, 0.0, 0.0]) 
prev_speed = 0
kime_trigger_time = 0
kime_msg = ""

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    b is the vertex of the angle.
    Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('Karate Dojo AI', cv2.WINDOW_NORMAL) 

print("[INFO] AI Dojo Started. System Ready.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror view for self-correction
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform Pose Estimation
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape
    
    # UI Overlay Background
    cv2.rectangle(image, (0, h-80), (w, h), (0, 0, 0), -1)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- 1. STANCE ANALYSIS (Zenkutsu-dachi) ---
        # Extract Left Leg Coordinates
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Extract Right Leg Coordinates
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate Knee Angles
        angle_l = calculate_angle(l_hip, l_knee, l_ankle)
        angle_r = calculate_angle(r_hip, r_knee, r_ankle)
        
        stance_status = "NEUTRAL"
        stance_color = (200, 200, 200)

        # Stance Classification Logic
        if angle_l < STANCE_LOW_THRESH and angle_r > STANCE_HIGH_THRESH:
            stance_status = "L ZENKUTSU"
            stance_color = (0, 255, 0)
        elif angle_r < STANCE_LOW_THRESH and angle_l > STANCE_HIGH_THRESH:
            stance_status = "R ZENKUTSU"
            stance_color = (0, 255, 0)
        elif angle_r < STANCE_LOW_THRESH and angle_l < STANCE_LOW_THRESH:
             stance_status = "TOO LOW"
             stance_color = (0, 0, 255)

        # --- 2. KIME & SPEED ANALYSIS ---
        # Get 3D coordinates for depth perception
        current_wrist = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * w 
        ])
        
        curr_time = time.time()
        dt = curr_time - prev_time
        
        current_speed = 0 
        if dt > 0:
            # Calculate Euclidean distance in 3D space
            distance = np.linalg.norm(current_wrist - prev_wrist)
            raw_speed = distance / dt 
            
            # Apply Low-Pass Filter (Smoothing) to reduce jitter
            current_speed = 0.7 * prev_speed + 0.3 * raw_speed 
            
            # Noise Gate
            if current_speed < 50: 
                current_speed = 0

            # Calculate Acceleration (Delta Speed / Time)
            acceleration = (current_speed - prev_speed) / dt

            # Kime Trigger: High speed followed by sudden negative acceleration (stop)
            if prev_speed > MIN_SPEED_FOR_KIME and acceleration < MIN_ACCEL_FOR_KIME:
                kime_trigger_time = curr_time
                kime_msg = "KIME!"

            # Update state for next frame
            prev_speed = current_speed
            prev_wrist = current_wrist
            prev_time = curr_time

        # --- 3. VISUALIZATION ---
        
        # Stance Status
        cv2.putText(image, f"STANCE: {stance_status}", (20, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)
        
        # Speed Meter
        cv2.putText(image, f"SPEED: {int(current_speed)}", (w-250, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Kime Feedback (remains on screen for 0.5s)
        if curr_time - kime_trigger_time < 0.5:
            cv2.putText(image, kime_msg, (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 15)

        # Draw Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Karate Dojo AI', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()