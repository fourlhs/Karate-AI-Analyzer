import numpy as np
import time
from collections import deque
import mediapipe as mp # Χρειαζόμαστε τα enums για ευκολία

class MechanicsAnalyzer:
    """
    The Core Physics Engine.
    Handles Speed, Kime, Stability AND Stance logic.
    """
    
    def __init__(self):
        # --- CONFIGURATION ---
        self.MIN_SPEED_FOR_KIME = 1100.0   
        self.MIN_ACCEL_FOR_KIME = -9500.0
        self.TRAJECTORY_BUFFER = 30
        self.VERTICAL_LIMIT = 40
        
        # Stance Thresholds (Zenkutsu)
        self.STANCE_LOW = 140  # Max angle for bent knee
        self.STANCE_HIGH = 160 # Min angle for straight leg

        # --- STATE ---
        self.prev_time = 0
        self.prev_wrist = np.array([0.0, 0.0, 0.0]) 
        self.prev_speed = 0
        self.com_history = deque(maxlen=self.TRAJECTORY_BUFFER)

    def calculate_angle(self, a, b, c):
        """Returns angle in degrees at vertex b."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def track_speed_kime(self, current_wrist_3d):
        """Calculates punch velocity and Kime."""
        curr_time = time.time()
        dt = curr_time - self.prev_time
        current_speed = 0
        is_kime = False
        
        if dt > 0:
            distance = np.linalg.norm(current_wrist_3d - self.prev_wrist)
            raw_speed = distance / dt 
            current_speed = 0.7 * self.prev_speed + 0.3 * raw_speed 
            if current_speed < 50: current_speed = 0
            acceleration = (current_speed - self.prev_speed) / dt
            if self.prev_speed > self.MIN_SPEED_FOR_KIME and acceleration < self.MIN_ACCEL_FOR_KIME:
                is_kime = True
            self.prev_speed = current_speed
            self.prev_wrist = current_wrist_3d
            self.prev_time = curr_time
            
        return current_speed, is_kime

    def track_stability(self, landmarks, w, h):
        """Tracks Center of Mass trajectory."""
        l_sh = np.array([landmarks[11].x * w, landmarks[11].y * h])
        r_sh = np.array([landmarks[12].x * w, landmarks[12].y * h])
        l_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
        r_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])
        
        com = ((l_sh + r_sh)/2 + (l_hip + r_hip)/2) / 2
        self.com_history.appendleft(com)
        
        status = "STABLE"
        color = (0, 255, 0)
        if len(self.com_history) > 10:
            y_coords = [p[1] for p in self.com_history]
            if (max(y_coords) - min(y_coords)) > self.VERTICAL_LIMIT:
                status = "UNSTABLE"
                color = (0, 0, 255)
        return int(com[0]), int(com[1]), status, color, self.com_history

    def track_stance(self, landmarks):
        """
        Analyzes legs for Zenkutsu-dachi.
        Returns: (status_text, color, left_knee_angle, right_knee_angle)
        """
        # Extract points (Normalized coordinates are fine for angles)
        # Left Leg: Hip(23), Knee(25), Ankle(27)
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        l_ankle = [landmarks[27].x, landmarks[27].y]
        
        # Right Leg: Hip(24), Knee(26), Ankle(28)
        r_hip = [landmarks[24].x, landmarks[24].y]
        r_knee = [landmarks[26].x, landmarks[26].y]
        r_ankle = [landmarks[28].x, landmarks[28].y]
        
        angle_l = self.calculate_angle(l_hip, l_knee, l_ankle)
        angle_r = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        status = "NEUTRAL"
        color = (200, 200, 200)

        # Logic: One leg bent (<140), One leg straight (>160)
        if angle_l < self.STANCE_LOW and angle_r > self.STANCE_HIGH:
            status = "L ZENKUTSU"
            color = (0, 255, 0)
        elif angle_r < self.STANCE_LOW and angle_l > self.STANCE_HIGH:
            status = "R ZENKUTSU"
            color = (0, 255, 0)
        elif angle_r < self.STANCE_LOW and angle_l < self.STANCE_LOW:
             status = "TOO LOW"
             color = (0, 0, 255)

        return status, color, int(angle_l), int(angle_r)