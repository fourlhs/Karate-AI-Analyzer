import cv2
import mediapipe as mp
import numpy as np

"""
Pose Legs Module
----------------
Analyzes lower body stance (Zenkutsu-dachi).
Calculates knee angles to determine if the stance is correct (deep enough).
"""

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculates angle between three points (a, b, c).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

cap = cv2.VideoCapture(0)
print("[INFO] Zenkutsu-dachi Analyzer Started.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Mirror view
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- LEFT LEG ---
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # --- RIGHT LEG ---
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate Angles
        angle_l_knee = calculate_angle(l_hip, l_knee, l_ankle)
        angle_r_knee = calculate_angle(r_hip, r_knee, r_ankle)

        # Stance Logic (Thresholds)
        # Front leg bent: 90-140 deg, Back leg straight: >160 deg
        status = "NEUTRAL"
        color = (255, 255, 255) # White

        if angle_l_knee < 140 and angle_r_knee > 160:
            status = "LEFT ZENKUTSU"
            color = (0, 255, 0) # Green
            
        elif angle_r_knee < 140 and angle_l_knee > 160:
            status = "RIGHT ZENKUTSU"
            color = (0, 255, 0) # Green
            
        elif angle_r_knee < 140 and angle_l_knee < 140:
            status = "TOO LOW / ERROR"
            color = (0, 0, 255) # Red

        # --- VISUALIZATION ---
        h, w, _ = image.shape
        
        # Status Box
        cv2.rectangle(image, (0,0), (350, 70), (0,0,0), -1)
        cv2.putText(image, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Knee Angles
        cv2.putText(image, str(int(angle_l_knee)), tuple(np.multiply(l_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, str(int(angle_r_knee)), tuple(np.multiply(r_knee, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Karate Analyzer - Legs', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()