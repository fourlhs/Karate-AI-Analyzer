import cv2
import threading
import time
import numpy as np
from typing import Tuple, Optional, Any

class CameraStream:
    """
    High-Performance Threaded Image Capture.
    
    Architecture Pattern: Producer-Consumer
    ---------------------------------------
    OpenCV's 'read()' is a blocking I/O operation. If run on the main thread, 
    it throttles the entire application logic (Physics + UI) to the camera's FPS.
    
    This class moves the I/O bottleneck to a background daemon thread, 
    allowing the main application to run at maximum speed (e.g., for Physics calculations).
    """

    def __init__(self, src: Any = 0, name: str = "Cam-A"):
        """
        Initializes the video stream and starts the acquisition thread.
        
        Args:
            src: Source index (int for webcam) or path (str for video file).
            name: Label for debugging/logging.
        """
        self.name = name
        self.src = src
        
        # Initialize the hardware connection
        self.stream = cv2.VideoCapture(src)
        
        # Validation: Ensure hardware is accessible
        if not self.stream.isOpened():
            print(f"[ERROR] {self.name}: Could not open video source {src}.")
            self.stopped = True
        else:
            self.stopped = False
            
        # Buffer: Stores the most recent frame
        # We only care about the *latest* data for real-time analysis.
        self.grabbed, self.frame = self.stream.read()
        
        # Telemetry: FPS monitoring
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Concurrency: Start background thread if source is valid
        if not self.stopped:
            self.t = threading.Thread(target=self.update, args=())
            self.t.daemon = True # Thread dies when main program exits
            self.t.start()
            print(f"[INFO] {self.name}: Thread started on source {src}.")

    def update(self):
        """
        The 'Daemon' Loop.
        Constantly grabs frames from the hardware buffer as fast as possible.
        Updates the shared 'self.frame' variable.
        """
        while not self.stopped:
            if not self.stream.isOpened():
                continue
            
            # Blocking call (I/O bound) - happens in background
            grabbed, frame = self.stream.read()
            
            if grabbed:
                # Critical Section: Update shared state
                # (Atomic assignment in Python, so explicit Lock is often skipped for single-frame buffers,
                # but technically a Lock would be safer in strictly typed languages).
                self.grabbed = grabbed
                self.frame = frame
                
                # Update FPS Metric
                self.frame_count += 1
                if time.time() - self.start_time > 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.start_time = time.time()
            else:
                # End of stream (Video file finished or Camera disconnected)
                self.stopped = True

    def read(self) -> Optional[np.ndarray]:
        """Returns the latest available frame (Non-blocking)."""
        return self.frame

    def stop(self):
        """Clean shutdown of hardware resources."""
        self.stopped = True
        if self.t.is_alive():
            self.t.join(timeout=1.0)
        self.stream.release()

class StereoManager:
    """
    Multi-Camera Fusion Controller (Phase 7 Core).
    
    Responsibilities:
    1. Hardware Abstraction: Manages multiple CameraStream instances.
    2. Synchronization: Attempts to align frames from different sources.
    3. Resilience: Handles cases where a secondary camera is missing (Graceful Degradation).
    """
    
    def __init__(self, source_a: Any = 0, source_b: Any = None):
        print(f"\n[STEREO] 📡 Initializing Dual-Camera System (Phase 7)...")
        
        # Primary Sensor (e.g., Laptop Webcam)
        self.cam_a = CameraStream(source_a, "Primary-Cam")
        
        # Secondary Sensor (e.g., USB Webcam or External Feed)
        # If None, the system runs in Mono-Vision mode (Phase 1-6 behavior).
        self.cam_b = None
        if source_b is not None:
            self.cam_b = CameraStream(source_b, "Secondary-Cam")
            print(f"[STEREO] ✅ Dual-View Mode Active.")
        else:
            print(f"[STEREO] ⚠️ Single-View Mode (Secondary camera not configured).")

        # Allow sensors to warm up (Auto-Exposure/White Balance adjustment)
        time.sleep(1.0) 
        print("[STEREO] System Ready for Acquisition.\n")

    def get_dual_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves synchronized frames from available sensors.
        
        Returns:
            Tuple: (Frame_Primary, Frame_Secondary)
            * Frame_Secondary will be None if not available.
        """
        frame_a = self.cam_a.read()
        frame_b = None
        
        if self.cam_b:
            frame_b = self.cam_b.read()
            
        return frame_a, frame_b

    def get_fps(self) -> Tuple[int, int]:
        """Returns telemetry for UI."""
        fps_a = self.cam_a.fps
        fps_b = self.cam_b.fps if self.cam_b else 0
        return fps_a, fps_b

    def stop(self):
        """Shuts down the entire sensor array."""
        print("[STEREO] Stopping sensors...")
        self.cam_a.stop()
        if self.cam_b:
            self.cam_b.stop()