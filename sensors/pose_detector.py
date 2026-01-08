import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    """
    Sensor Abstraction Layer.
    Currently wraps Google MediaPipe.
    Ready for Multi-Camera/YOLO upgrades.
    """
    
    def __init__(self, static_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.results = None

    def find_pose(self, image, draw=True):
        """Processes the frame and finds the skeleton."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                image, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
        return image

    def get_landmarks(self):
        """Returns raw landmarks list or None."""
        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks.landmark
        return None

    def get_3d_coordinates(self, image, landmark_index):
        """
        Helper to get (x, y, z) in pixels.
        Critical for Physics Engine (Phase 2).
        """
        if not self.results or not self.results.pose_landmarks:
            return np.array([0, 0, 0])

        h, w, _ = image.shape
        lm = self.results.pose_landmarks.landmark[landmark_index]
        return np.array([lm.x * w, lm.y * h, lm.z * w])