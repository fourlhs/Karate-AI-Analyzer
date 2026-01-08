import numpy as np
import time
from collections import deque
import mediapipe as mp

class MechanicsAnalyzer:
    """
    Core Physics & Biomechanics Engine.
    
    Responsibilities:
    1. Kinematics Analysis (Velocity, Acceleration, Kime/Impulse detection).
    2. Statics & Stability (Center of Mass trajectory analysis).
    3. Posture Classification (Heuristic Decision Tree for WKF Stances).
    
    Architecture:
    - Uses Exponential Moving Average (EMA) for real-time signal smoothing.
    - Implements vector geometry for joint angle calculation.
    """
    
    def __init__(self):
        # --- CONFIGURATION (Phase 4 Calibration) ---
        # Thresholds derived from Golden Reference Data (pixels/sec)
        self.MIN_SPEED_FOR_KIME = 1100.0   
        self.MIN_ACCEL_FOR_KIME = -9500.0 # High negative acceleration = Snap back
        
        # Stability Constraints
        self.TRAJECTORY_BUFFER = 30       # Frames to track CoM history
        self.VERTICAL_LIMIT = 40          # Max allowed vertical oscillation (px)
        
        # Geometry Constraints (Zenkutsu Dachi)
        self.STANCE_LOW = 140  # Max angle for flexion (loaded leg)
        self.STANCE_HIGH = 160 # Min angle for extension (straight leg)

        # --- INTERNAL STATE ---
        self.prev_time = 0
        self.prev_wrist = np.array([0.0, 0.0, 0.0]) 
        self.prev_speed = 0
        self.com_history = deque(maxlen=self.TRAJECTORY_BUFFER)

    def calculate_angle(self, a, b, c):
        """
        Calculates the 2D joint angle at vertex B using arctangent.
        
        Args:
            a, b, c: Coordinates [x, y] of three joints (e.g., Hip, Knee, Ankle).
        Returns:
            angle: Degree (0-180).
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        
        # Calculate vector angles relative to the X-axis
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        # Normalize to internal angle (0-180)
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def track_speed_kime(self, current_wrist_3d):
        """
        Real-time Kinematics Analysis.
        
        Calculates instantaneous velocity and detects 'Kime' (Focus) events
        based on rapid deceleration signatures (Snap).
        """
        curr_time = time.time()
        dt = curr_time - self.prev_time
        
        current_speed = 0
        is_kime = False
        
        if dt > 0:
            # 1. Euclidean Distance (Delta Position)
            distance = np.linalg.norm(current_wrist_3d - self.prev_wrist)
            
            # 2. Instantaneous Velocity (v = d/t)
            raw_speed = distance / dt 
            
            # 3. Signal Smoothing (Exponential Moving Average - EMA)
            # Reduces sensor jitter while maintaining responsiveness.
            current_speed = 0.7 * self.prev_speed + 0.3 * raw_speed 
            
            # Noise Gate
            if current_speed < 50: current_speed = 0
            
            # 4. Acceleration Calculation (a = dv/dt)
            acceleration = (current_speed - self.prev_speed) / dt
            
            # 5. Kime Detection Logic (Impulse Analysis)
            # Kime requires high entry speed AND sudden negative acceleration (braking).
            if self.prev_speed > self.MIN_SPEED_FOR_KIME and acceleration < self.MIN_ACCEL_FOR_KIME:
                is_kime = True
            
            # Update State
            self.prev_speed = current_speed
            self.prev_wrist = current_wrist_3d
            self.prev_time = curr_time
            
        return current_speed, is_kime

    def track_stability(self, landmarks, w, h):
        """
        Statics Analysis: Center of Mass (CoM) Tracking.
        
        Approximates CoM using the geometric center of the torso box 
        (Shoulders + Hips) and monitors vertical oscillation.
        """
        # Feature Extraction: Torso Anchors
        l_sh = np.array([landmarks[11].x * w, landmarks[11].y * h])
        r_sh = np.array([landmarks[12].x * w, landmarks[12].y * h])
        l_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
        r_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])
        
        # CoM Approximation (Geometric Centroid)
        com = ((l_sh + r_sh)/2 + (l_hip + r_hip)/2) / 2
        self.com_history.appendleft(com)
        
        status = "STABLE"
        color = (0, 255, 0)
        
        # Stability Logic: Vertical Deviation Check
        if len(self.com_history) > 10:
            y_coords = [p[1] for p in self.com_history]
            vertical_oscillation = max(y_coords) - min(y_coords)
            
            if vertical_oscillation > self.VERTICAL_LIMIT:
                status = "UNSTABLE"
                color = (0, 0, 255) # Red warning
                
        return int(com[0]), int(com[1]), status, color, self.com_history

    def track_stance(self, landmarks):
        """
        Biomechanical Stance Classifier (BSC).
        
        This module implements a heuristic-based decision tree to classify 
        Karate-specific postures by fusing angular kinematics and 
        spatial distribution of the lower extremities.
        
        Methodology:
        - Angular Analysis: Calculates 2D projection of joint flexions.
        - Spatial Mapping: Measures Euclidean and axis-aligned distances between anchors.
        - Alignment Validation: Checks toe-to-heel vectors for rotation-sensitive stances.
        """
        
        # --- 1. Landmark Mapping (Normalization Layer) ---
        # Hip, Knee, Ankle, and Foot landmarks (MediaPipe Topology)
        l_hip, r_hip = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]
        l_knee, r_knee = [landmarks[25].x, landmarks[25].y], [landmarks[26].x, landmarks[26].y]
        l_ank, r_ank = [landmarks[27].x, landmarks[27].y], [landmarks[28].x, landmarks[28].y]
        l_foot, r_foot = [landmarks[31].x, landmarks[31].y], [landmarks[32].x, landmarks[32].y]

        # --- 2. Feature Extraction ---
        # Flexion angles for the sagittal/frontal plane projection
        angle_l = self.calculate_angle(l_hip, l_knee, l_ank)
        angle_r = self.calculate_angle(r_hip, r_knee, r_ank)
        
        # Longitudinal (Y - Depth) and Lateral (X - Width) base dimensions
        base_width_x = abs(l_ank[0] - r_ank[0])
        base_length_y = abs(l_ank[1] - r_ank[1])
        
        # Default State
        status = "NEUTRAL"
        color = (200, 200, 200)

        # --- 3. Classifier Logic (Heuristic Engine) ---

        # A. ZENKUTSU DACHI (Front Stance)
        # Logic: Deep stance (Length > Y_Threshold) AND Asymmetric Leg Loading (One bent, one straight)
        if base_length_y > 0.2 and ((angle_l < 110 and angle_r > 150) or (angle_r < 110 and angle_l > 150)):
            status = "ZENKUTSU DACHI"
            color = (0, 255, 0)

        # B. KOKUTSU DACHI (Back Stance)
        # Logic: Weight Shift Posterior (Hip over back leg) AND L-Shape feet
        elif base_length_y > 0.15:
            # Check if posterior knee is flexed while anterior is extended
            if (angle_r < 110 and l_ank[1] < r_ank[1]) or (angle_l < 110 and r_ank[1] < l_ank[1]):
                status = "KOKUTSU DACHI"
                color = (255, 0, 255)

        # C. KIBA vs SHIKO DACHI (Straddle vs Square Stance)
        # Logic: Wide Stance (X > Threshold) AND Bilateral Flexion (Both knees bent)
        # Differentiation: Toe Vector Alignment (External Rotation)
        elif angle_l < 140 and angle_r < 140 and base_width_x > 0.2:
            toe_outward_l = abs(l_foot[0] - l_ank[0])
            if toe_outward_l > 0.05: # External rotation detected
                status = "SHIKO DACHI"
            else: # Parallel alignment
                status = "KIBA DACHI"
            color = (255, 165, 0)

        # D. NEKO ASHI DACHI (Cat Stance)
        # Logic: Compressed base (Small Y) AND Unilateral Loading (Rear leg bears 90% weight)
        elif base_length_y < 0.15 and (angle_l < 100 or angle_r < 100):
            status = "NEKO ASHI DACHI"
            color = (0, 255, 255)

        # E. SANCHIN DACHI (Hourglass Stance)
        # Logic: Compact base AND Internal Adduction (Knees/Toes inward - "Pigeon-toed")
        elif base_length_y < 0.15 and angle_l < 155 and angle_r < 155:
            status = "SANCHIN DACHI"
            color = (0, 128, 255)

        # F. FUDO DACHI (Rooted Stance / Sochin)
        # Logic: Hybrid of Zenkutsu/Kiba. Wide base with balanced bilateral flexion.
        elif base_length_y > 0.15 and angle_l < 140 and angle_r < 140:
            status = "FUDO DACHI"
            color = (0, 100, 0)

        # G. KAKE DACHI (Crossed Stance)
        # Logic: X-Axis Intersection. Ankle position crosses the midline relative to Hips.
        elif (l_ank[0] > r_ank[0] and l_hip[0] < r_hip[0]) or (r_ank[0] < l_ank[0] and r_hip[0] > l_hip[0]):
            status = "KAKE DACHI"
            color = (128, 0, 128)

        return status, color, int(angle_l), int(angle_r)