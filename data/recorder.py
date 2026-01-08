import csv
import time
import os
import numpy as np

class DataRecorder:
    """
    The 'Black Box' of the System.
    Records every metric and skeletal frame for Post-Analysis & Machine Learning.
    
    Target:
    - Create 'Golden Reference' datasets from champions.
    - Train ML models using these CSVs.
    """
    
    def __init__(self, output_folder="sessions"):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        self.buffer = []
        self.is_recording = False
        self.start_time = 0
        self.filename = ""

    def start(self):
        """Starts a new recording session."""
        self.buffer = []
        self.is_recording = True
        self.start_time = time.time()
        # Create a unique filename based on timestamp
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        self.filename = f"{self.output_folder}/karate_session_{timestamp_str}.csv"
        print(f"[REC] Recording started: {self.filename}")

    def log(self, metrics):
        """
        Adds a frame of data to the buffer.
        :param metrics: Dictionary containing all values (Speed, Angles, CoM, etc.)
        """
        if not self.is_recording:
            return
            
        # Add relative timestamp (time since start)
        metrics['timestamp'] = time.time() - self.start_time
        self.buffer.append(metrics)

    def stop(self):
        """Stops recording and triggers the save process."""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        return self.save_session()

    def save_session(self):
        """
        Flushes buffer to disk and returns the filename for post-analysis.
        Required for Phase 5 AI Judge integration.
        """
        if not self.buffer:
            print("[WARN] Buffer empty. Nothing to save.")
            return None

        keys = self.buffer[0].keys()
        
        try:
            with open(self.filename, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.buffer)
            
            print(f"[REC] Data persisted to: {self.filename} ({len(self.buffer)} frames)")
            return self.filename
            
        except Exception as e:
            print(f"[ERROR] Critical I/O Failure during CSV serialization: {e}")
            return None