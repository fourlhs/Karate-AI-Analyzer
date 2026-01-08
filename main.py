import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

"""
AI Karate Analyzer - Main Dojo Module
-------------------------------------
A comprehensive biomechanics analysis system fusing three distinct metrics:
1. Static Geometry (Zenkutsu-dachi Stance Analysis)
2. Kinematics (Speed & Acceleration/Kime Detection)
3. Stability (Center of Mass Trajectory & Vertical Oscillation)

Target: WKF Objective Scoring & Training Assistance.
"""

# --- CONFIGURATION & TUNING ---

# 1. Kime (Snap) Thresholds
# Minimum velocity (px/s) required before impact
MIN_SPEED_FOR_KIME = 1100.0   
# Maximum negative acceleration (deceleration) to register a 'lock'
MIN_ACCEL_FOR_KIME = -9500.0

# 2. Stance Geometry (Zenkutsu-dachi)
STANCE_LOW_THRESH = 140  # Max angle for front knee (too high)
STANCE_HIGH_THRESH = 155 # Min angle for back leg (must be straight)

# 3. Stability Analysis
# Buffer size for visual trajectory trail
TRAJECTORY_BUFFER_SIZE = 30 
# Pixel threshold for vertical bouncing (instability)
VERTICAL_OSCILLATION_LIMIT = 40

# --- SYSTEM INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- STATE VARIABLES ---

# Kime/Speed State
prev_time = 0
prev_wrist = np.array([0.0, 0.0, 0.0]) 
prev_speed = 0
kime_trigger_time = 0
kime_msg = ""

# Stability State (FIFO Queue for trajectory)
com_history = deque(maxlen=TRAJECTORY_BUFFER_SIZE)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """
    Computes the angle between three points (a, b, c) in 2D space.
    Vertex is point b. Returns degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Karate Dojo AI', cv2.WINDOW_NORMAL) 

print("[INFO] System Initialized. Sensors Active.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror view for intuitive self-correction
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process Pose Estimation
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape
    
    # UI Dashboard Background (Bottom bar)
    cv2.rectangle(image, (0, h-80), (w, h), (0, 0, 0), -1)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # =============================================================
        # MODULE 1: STABILITY TRACKER (Center of Mass)
        # =============================================================
        
        # Extract Torso Anchors (Shoulders & Hips) in Pixels
        l_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
        r_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
        l_hip_px = np.array([landmarks[23].x * w, landmarks[23].y * h])
        r_hip_px = np.array([landmarks[24].x * w, landmarks[24].y * h])
        
        # Compute Geometric Centers
        shoulder_center = (l_shoulder + r_shoulder) / 2
        hip_center = (l_hip_px + r_hip_px) / 2
        
        # Compute Center of Mass (CoM) Approximation
        com = (shoulder_center + hip_center) / 2
        com_history.appendleft(com)

        # Visualizing Trajectory
        for i in range(1, len(com_history)):
            if com_history[i - 1] is None or com_history[i] is None: continue
            
            # Dynamic thickness gradient (Head is thick, tail is thin)
            thickness = int(np.sqrt(TRAJECTORY_BUFFER_SIZE / float(i + 1)) * 2.5)
            pt1 = (int(com_history[i-1][0]), int(com_history[i-1][1]))
            pt2 = (int(com_history[i][0]), int(com_history[i][1]))
            cv2.line(image, pt1, pt2, (0, 255, 255), thickness)

        # Visualizing CoM Point
        cv2.circle(image, (int(com[0]), int(com[1])), 8, (0, 0, 255), -1)

        # Analyzing Vertical Oscillation (Bouncing)
        stability_status = "STABLE"
        stability_color = (0, 255, 0)
        
        if len(com_history) > 10:
            y_coords = [p[1] for p in com_history]
            vertical_shift = max(y_coords) - min(y_coords)
            
            if vertical_shift > VERTICAL_OSCILLATION_LIMIT:
                stability_status = "UNSTABLE"
                stability_color = (0, 0, 255)
            
            # Display Stability Metrics (Top Left)
            cv2.putText(image, f"CORE: {stability_status}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, stability_color, 2)

        # =============================================================
        # MODULE 2: STANCE ANALYSIS (Zenkutsu-dachi)
        # =============================================================
        
        # Landmarks for Angles (Normalized coordinates are fine for angles)
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        l_ankle = [landmarks[27].x, landmarks[27].y]
        
        r_hip = [landmarks[24].x, landmarks[24].y]
        r_knee = [landmarks[26].x, landmarks[26].y]
        r_ankle = [landmarks[28].x, landmarks[28].y]
        
        angle_l = calculate_angle(l_hip, l_knee, l_ankle)
        angle_r = calculate_angle(r_hip, r_knee, r_ankle)
        
        stance_status = "NEUTRAL"
        stance_color = (200, 200, 200)

        # Classification Logic
        if angle_l < STANCE_LOW_THRESH and angle_r > STANCE_HIGH_THRESH:
            stance_status = "L ZENKUTSU"
            stance_color = (0, 255, 0)
        elif angle_r < STANCE_LOW_THRESH and angle_l > STANCE_HIGH_THRESH:
            stance_status = "R ZENKUTSU"
            stance_color = (0, 255, 0)
        elif angle_r < STANCE_LOW_THRESH and angle_l < STANCE_LOW_THRESH:
             stance_status = "TOO LOW"
             stance_color = (0, 0, 255)

        # Display Stance (Bottom Left)
        cv2.putText(image, f"STANCE: {stance_status}", (20, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)

        # =============================================================
        # MODULE 3: KINEMATICS (Speed & Kime)
        # =============================================================
        
        # 3D Wrist Coordinates (Z-axis is critical for punch depth)
        current_wrist = np.array([
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * w 
        ])
        
        curr_time = time.time()
        dt = curr_time - prev_time
        
        current_speed = 0 
        if dt > 0:
            # Euclidean Distance in 3D
            distance = np.linalg.norm(current_wrist - prev_wrist)
            raw_speed = distance / dt 
            
            # Low-Pass Filter (Smoothing)
            current_speed = 0.7 * prev_speed + 0.3 * raw_speed 
            if current_speed < 50: current_speed = 0 # Noise gate

            # Acceleration Calculation
            acceleration = (current_speed - prev_speed) / dt

            # Kime Trigger Logic
            if prev_speed > MIN_SPEED_FOR_KIME and acceleration < MIN_ACCEL_FOR_KIME:
                kime_trigger_time = curr_time
                kime_msg = "KIME!"

            prev_speed = current_speed
            prev_wrist = current_wrist
            prev_time = curr_time

        # Display Speed (Bottom Right)
        cv2.putText(image, f"SPEED: {int(current_speed)}", (w-250, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Kime Feedback Flash (Center)
        if curr_time - kime_trigger_time < 0.4:
            cv2.putText(image, kime_msg, (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (w, h), (0, 255, 0), 15)

        # =============================================================
        # RENDER SKELETON
        # =============================================================
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Karate Dojo AI', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()