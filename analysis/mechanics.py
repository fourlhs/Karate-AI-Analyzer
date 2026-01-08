import numpy as np
import time
from collections import deque
import sys
from typing import Tuple, List, Any

# --- ARCHITECTURE: HIGH-PERFORMANCE ENGINE LOADING ---
# We attempt to import the compiled C++ module ('karate_core').
# Design Pattern: "Graceful Degradation"
# - If C++ is found: System runs in "Real-Time Mode" (Phase 6).
# - If C++ is missing: System falls back to "Legacy Mode" (Python/NumPy).

USING_CPP = False
try:
    sys.path.append('.') 
    import karate_core
    USING_CPP = True
    print("\n[SYSTEM] 🚀 HIGH-PERFORMANCE C++ ENGINE LOADED")
    print("[SYSTEM] Physics & Biomechanics running on bare metal.\n")
except ImportError as e:
    print(f"\n[SYSTEM] ⚠️ C++ Engine not found ({e}). Running in LEGACY Python mode.\n")

class MechanicsAnalyzer:
    """
    Hybrid Physics Engine Wrapper.
    
    Design Pattern: Proxy / Adapter
    -------------------------------
    Acts as the interface between the high-level UI (Python/MediaPipe) 
    and the low-level calculation engine (C++ or Python).
    
    Responsibilities:
    1. Data Marshaling: Converts MediaPipe landmarks to C++ vectors.
    2. Execution: Calls the appropriate backend logic.
    3. State Management: Maintains visual history for UI rendering.
    """
    
    def __init__(self):
        # --- CONFIGURATION (Golden Reference Standards) ---
        # Thresholds derived from WKF Champion Data
        self.MIN_SPEED_FOR_KIME = 1100.0   
        self.MIN_ACCEL_FOR_KIME = -9500.0 # Snap-back threshold
        self.VERTICAL_LIMIT = 40          # Stability tolerance (pixels)
        self.STABILITY_BUFFER_SIZE = 30   # History length for CoM tracking

        # --- INTERNAL STATE (Visualization) ---
        # We maintain a deque in Python strictly for UI rendering purposes (drawing the tail).
        # The C++ engine maintains its own separate deque for mathematical analysis.
        self.viz_com_history = deque(maxlen=self.STABILITY_BUFFER_SIZE)

        # --- BACKEND INITIALIZATION ---
        if USING_CPP:
            # Initialize the C++ Object (Heap allocation via PyBind11)
            self.cpp_engine = karate_core.KarateEngine()
        else:
            # Initialize Legacy Python State
            print("[INFO] Initializing Legacy Python Mechanics...")
            self.prev_time = 0
            self.prev_wrist = np.array([0.0, 0.0, 0.0]) 
            self.prev_speed = 0
            # Python logic needs its own deque for stability calculations
            self.py_com_history = deque(maxlen=self.STABILITY_BUFFER_SIZE)

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """
        Computes the 2D joint angle at vertex B.
        Used for: Knee Flexion, Elbow Snap, Hip Rotation.
        """
        if USING_CPP:
            # [FAST PATH] C++ handles vector math instantly
            return self.cpp_engine.calculate_angle(a, b, c)
        else:
            # [SLOW PATH] Python/NumPy Implementation
            a, b, c = np.array(a), np.array(b), np.array(c)
            # Calculate angle using arctan2 (robust against quadrant issues)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0: angle = 360 - angle
            return angle

    def track_speed_kime(self, current_wrist_3d: np.ndarray) -> Tuple[float, bool]:
        """
        Kinematics Analysis: Velocity & Impact Detection.
        
        Math:
        v = dx/dt (Velocity)
        a = dv/dt (Acceleration)
        Kime = (v > Threshold) AND (a < Negative_Threshold)
        
        Returns: (speed_px_s, is_kime_bool)
        """
        x, y, z = current_wrist_3d

        if USING_CPP:
            # [FAST PATH] Direct call to C++
            # Passes individual coordinates to avoid overhead of converting full numpy array
            return self.cpp_engine.track_speed_kime(x, y, z, time.time())
        
        else:
            # [SLOW PATH] Legacy Python Logic
            curr_time = time.time()
            if self.prev_time == 0:
                self.prev_time = curr_time
                self.prev_wrist = current_wrist_3d
                return 0.0, False

            dt = curr_time - self.prev_time
            if dt <= 0.001: return 0.0, False # Prevent division by zero

            # Euclidean distance
            dist = np.linalg.norm(current_wrist_3d - self.prev_wrist)
            raw_speed = dist / dt 
            
            # EMA Smoothing
            current_speed = 0.7 * self.prev_speed + 0.3 * raw_speed 
            if current_speed < 50: current_speed = 0
            
            # Acceleration
            accel = (current_speed - self.prev_speed) / dt
            
            is_kime = False
            if self.prev_speed > self.MIN_SPEED_FOR_KIME and accel < self.MIN_ACCEL_FOR_KIME:
                is_kime = True
            
            self.prev_speed = current_speed
            self.prev_wrist = current_wrist_3d
            self.prev_time = curr_time
            return current_speed, is_kime

    def track_stability(self, landmarks, w, h) -> Tuple[int, int, str, Tuple[int, int, int], deque]:
        """
        Statics Analysis: Center of Mass (CoM).
        
        Calculates the geometric center of the torso and monitors vertical oscillation.
        Returns: (x, y, status_string, color_tuple, history_deque_for_ui)
        """
        

        # Feature Extraction (Needed for both engines)
        # We manually extract coordinates to pass simple floats/lists to C++
        l_sh = [landmarks[11].x * w, landmarks[11].y * h]
        r_sh = [landmarks[12].x * w, landmarks[12].y * h]
        l_hip = [landmarks[23].x * w, landmarks[23].y * h]
        r_hip = [landmarks[24].x * w, landmarks[24].y * h]

        if USING_CPP:
            # [FAST PATH] C++ Logic
            # C++ calculates CoM and determines stability status internally
            cx, cy, status, color = self.cpp_engine.track_stability(l_sh, r_sh, l_hip, r_hip)
            
            # We append to Python deque ONLY for visualization (UI drawing)
            self.viz_com_history.appendleft((cx, cy))
            return cx, cy, status, color, self.viz_com_history
            
        else:
            # [SLOW PATH] Python Logic
            l_sh_np = np.array(l_sh)
            r_sh_np = np.array(r_sh)
            l_hip_np = np.array(l_hip)
            r_hip_np = np.array(r_hip)
            
            com = ((l_sh_np + r_sh_np)/2 + (l_hip_np + r_hip_np)/2) / 2
            
            # Use Python-side logic deque
            self.py_com_history.appendleft(com)
            self.viz_com_history.appendleft((int(com[0]), int(com[1]))) # Sync viz
            
            status = "STABLE"
            color = (0, 255, 0)
            
            if len(self.py_com_history) > 10:
                y_coords = [p[1] for p in self.py_com_history]
                if max(y_coords) - min(y_coords) > self.VERTICAL_LIMIT:
                    status = "UNSTABLE"
                    color = (0, 0, 255)
            
            return int(com[0]), int(com[1]), status, color, self.viz_com_history

    def track_stance(self, landmarks) -> Tuple[str, Tuple[int, int, int], int, int]:
        """
        Biomechanical Classifier: Detects WKF Stances.
        
        Uses a Heuristic Decision Tree to classify postures based on:
        1. Base Dimensions (Width/Length)
        2. Joint Angles (Knee Flexion)
        3. Alignment (Toe vectors)
        
        Returns: (Stance Name, Color, Left Angle, Right Angle)
        """
        # Feature Extraction: Normalize inputs for C++
        # We pass raw normalized [0-1] coordinates
        l_hip = [landmarks[23].x, landmarks[23].y]
        r_hip = [landmarks[24].x, landmarks[24].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        r_knee = [landmarks[26].x, landmarks[26].y]
        l_ank = [landmarks[27].x, landmarks[27].y]
        r_ank = [landmarks[28].x, landmarks[28].y]
        l_foot = [landmarks[31].x, landmarks[31].y]
        r_foot = [landmarks[32].x, landmarks[32].y]

        if USING_CPP:
            # [FAST PATH] C++ Execution
            # Runs the full Decision Tree in compiled machine code
            return self.cpp_engine.track_stance(
                l_hip, r_hip, l_knee, r_knee, l_ank, r_ank, l_foot, r_foot
            )
        
        else:
            # [SLOW PATH] Legacy Python Fallback
            # Replicates the exact logic of the C++ engine
            angle_l = self.calculate_angle(l_hip, l_knee, l_ank)
            angle_r = self.calculate_angle(r_hip, r_knee, r_ank)
            
            base_width_x = abs(l_ank[0] - r_ank[0])
            base_length_y = abs(l_ank[1] - r_ank[1])
            
            status = "NEUTRAL"
            color = (200, 200, 200)

            # A. ZENKUTSU DACHI
            if base_length_y > 0.2 and ((angle_l < 110 and angle_r > 150) or (angle_r < 110 and angle_l > 150)):
                status = "ZENKUTSU DACHI"; color = (0, 255, 0)
            
            # B. KOKUTSU DACHI
            elif base_length_y > 0.15 and ((angle_r < 110 and l_ank[1] < r_ank[1]) or (angle_l < 110 and r_ank[1] < l_ank[1])):
                status = "KOKUTSU DACHI"; color = (255, 0, 255)
            
            # C. KIBA vs SHIKO DACHI
            elif angle_l < 140 and angle_r < 140 and base_width_x > 0.2:
                toe_outward_l = abs(l_foot[0] - l_ank[0])
                if toe_outward_l > 0.05:
                    status = "SHIKO DACHI"
                else:
                    status = "KIBA DACHI"
                color = (255, 165, 0)

            # D. NEKO ASHI DACHI
            elif base_length_y < 0.15 and (angle_l < 100 or angle_r < 100):
                status = "NEKO ASHI DACHI"; color = (0, 255, 255)

            # E. SANCHIN DACHI
            elif base_length_y < 0.15 and angle_l < 155 and angle_r < 155:
                status = "SANCHIN DACHI"; color = (0, 128, 255)
            
            # F. FUDO DACHI
            elif base_length_y > 0.15 and angle_l < 140 and angle_r < 140:
                status = "FUDO DACHI"; color = (0, 100, 0)

            # G. KAKE DACHI
            elif (l_ank[0] > r_ank[0] and l_hip[0] < r_hip[0]) or (r_ank[0] < l_ank[0] and r_hip[0] > l_hip[0]):
                status = "KAKE DACHI"; color = (128, 0, 128)
            
            return status, color, int(angle_l), int(angle_r)

    def reset(self):
        """Resets both engines for a new session."""
        self.viz_com_history.clear()
        if USING_CPP:
            self.cpp_engine.reset()
        else:
            self.prev_speed = 0
            self.prev_time = 0
            self.prev_wrist = np.array([0.0, 0.0, 0.0])
            self.py_com_history.clear()