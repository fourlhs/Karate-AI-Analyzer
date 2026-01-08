import cv2
import mediapipe as mp
import numpy as np

"""
Pose Hands Module
-----------------
Visualizes upper body mechanics and calculates elbow angles.
Useful for checking guard position (Kamae) and arm extension.
"""

# --- INITIALIZATION ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    b is the vertex. Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Initialize Camera
cap = cv2.VideoCapture(0)
print("[INFO] Hand Analysis Module Started.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Mirror view
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process Pose
    results = pose.process(image_rgb)
    
    h, w, _ = image.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # --- LEFT ARM ---
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        angle_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        # --- RIGHT ARM ---
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        angle_r = calculate_angle(r_shoulder, r_elbow, r_wrist)

        # --- VISUALIZATION ---
        
        # Display Left Arm Angle (White)
        cv2.putText(image, str(int(angle_l)), 
                    tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                           
        # Display Right Arm Angle (Blue)
        cv2.putText(image, str(int(angle_r)), 
                    tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Draw Skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    cv2.imshow('Karate Analyzer - Hands', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()