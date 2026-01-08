import cv2
import mediapipe as mp
import numpy as np
from collections import deque

"""
Stability Tracker Module
------------------------
Analyzes the athlete's Center of Mass (CoM) in real-time.
- Calculates the geometric center of the torso.
- Visualizes the movement trajectory to detect unnecessary bouncing.
- Measures vertical oscillation (Z-axis stability) during movement.
"""

# --- CONFIGURATION ---
# Number of frames to keep in the trajectory history (visual trail)
TRAJECTORY_BUFFER_SIZE = 30 

# Threshold (pixels) for vertical instability
# If the CoM moves up/down more than this, it flags as "Unstable"
VERTICAL_OSCILLATION_LIMIT = 40

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Data structure to store past CoM points (FIFO queue)
com_history = deque(maxlen=TRAJECTORY_BUFFER_SIZE)

# Initialize Camera
cap = cv2.VideoCapture(0)
print("[INFO] Stability Tracker Module Started.")

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
        
        # Extract Torso Landmarks
        # 11: Left Shoulder, 12: Right Shoulder
        # 23: Left Hip, 24: Right Hip
        l_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
        r_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
        l_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
        r_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])
        
        # 1. Calculate Shoulder Midpoint
        shoulder_center = (l_shoulder + r_shoulder) / 2
        
        # 2. Calculate Hip Midpoint (Pelvis center)
        hip_center = (l_hip + r_hip) / 2
        
        # 3. Calculate Approximate Center of Mass (CoM)
        # We average the shoulder and hip centers to find the core center
        com = (shoulder_center + hip_center) / 2
        
        # Add current CoM to history buffer
        com_history.appendleft(com)

        # --- VISUALIZATION ---
        
        # Draw CoM Marker
        cv2.circle(image, (int(com[0]), int(com[1])), 10, (0, 0, 255), -1) # Red Dot
        cv2.circle(image, (int(com[0]), int(com[1])), 15, (0, 255, 255), 2) # Yellow Ring
        
        # Draw Trajectory Trail
        for i in range(1, len(com_history)):
            if com_history[i - 1] is None or com_history[i] is None:
                continue
            
            # Dynamic thickness: thicker at the head, thinner at the tail
            thickness = int(np.sqrt(TRAJECTORY_BUFFER_SIZE / float(i + 1)) * 2.5)
            
            pt1 = (int(com_history[i-1][0]), int(com_history[i-1][1]))
            pt2 = (int(com_history[i][0]), int(com_history[i][1]))
            
            cv2.line(image, pt1, pt2, (0, 255, 255), thickness)

        # Stability Metrics (Vertical Oscillation)
        if len(com_history) > 10:
            # Extract Y-coordinates from history
            y_coords = [p[1] for p in com_history]
            vertical_shift = max(y_coords) - min(y_coords)
            
            status_text = "STABLE"
            status_color = (0, 255, 0) # Green
            
            if vertical_shift > VERTICAL_OSCILLATION_LIMIT:
                status_text = "UNSTABLE (BOUNCING)"
                status_color = (0, 0, 255) # Red
            
            # Display Status
            cv2.rectangle(image, (0,0), (350, 100), (0,0,0), -1)
            cv2.putText(image, f"Stability: {status_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.putText(image, f"Vertical Shift: {int(vertical_shift)} px", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Center of Mass (CoM) Tracker', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()