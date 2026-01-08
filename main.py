import cv2
import time
import sys
import pandas as pd
import numpy as np
from visualize_data import analyze_session

# --- CUSTOM MODULES (Architecture Layers) ---
from sensors.pose_detector import PoseDetector      # Perception Layer
from analysis.mechanics import MechanicsAnalyzer    # Logic Layer
from data.recorder import DataRecorder              # Storage Layer
from ui.ai_judge import WKF_Judge                      # Evaluation Layer (Phase 5)
import config                                       # Global Configuration

def main():
    """
    Main Execution Pipeline for AI Karate Coach.
    Orchestrates Sensor Data -> Physics Engine -> Recorder -> AI Judge.
    """
    
    # --- 1. SYSTEM INITIALIZATION ---
    print("[INIT] Initializing Computer Vision Subsystems...")
    cap = cv2.VideoCapture(0)

    # video_path = "assets/videos/golden_kata_2.mp4"
    # cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[CRITICAL] Camera sensor not found. Aborting.")
        sys.exit()

    # Instantiate core components
    detector = PoseDetector()       # MediaPipe Wrapper
    analyzer = MechanicsAnalyzer()  # Physics & Geometry Engine
    recorder = DataRecorder()       # Telemetry Logger
    judge = WKF_Judge()             # WKF Rule Engine (Phase 5)

    # State Variables
    kime_timer = 0
    kime_msg = ""
    
    print(f"[READY] System Online. WKF Thresholds loaded: Excellent > {config.WKF_SPEED_EXCELLENT} px/s")
    print("[INSTRUCT] Press 'r' to toggle Recording | Press 'q' to Quit & Score.")

    # --- 2. REAL-TIME PROCESSING LOOP ---
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # Mirror view for user interaction (UX)
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # A. SENSING (Perception)
        image = detector.find_pose(image)
        landmarks = detector.get_landmarks()
        
        # Default empty metrics container
        metrics = {} 
        
        if landmarks:
            # B. PHYSICS ANALYSIS (Logic)
            # 1. Kinematics (Speed & Acceleration)
            wrist_3d = detector.get_3d_coordinates(image, 16) # Right Wrist
            speed, is_kime = analyzer.track_speed_kime(wrist_3d)
            
            # 2. Stability (Center of Mass Trajectory)
            com_x, com_y, stab_status, stab_color, history = analyzer.track_stability(landmarks, w, h)
            
            # 3. Geometry (Stance Classification)
            stance_status, stance_color, ang_l, ang_r = analyzer.track_stance(landmarks)
            
            # Kime Event Trigger (Visual Feedback)
            if is_kime:
                kime_timer = time.time()
                kime_msg = "KIME DETECTED!"

            # C. DATA PACKET PREPARATION
            # Constructing the telemetry frame for storage
            metrics = {
                "speed": speed,
                "is_kime": 1 if is_kime else 0,
                "com_x": com_x,
                "com_y": com_y,
                "stance_status": stance_status,
                "knee_angle_l": ang_l,
                "knee_angle_r": ang_r,
                "stability_status": stab_status
                # Future: Add raw landmarks for Dynamic Time Warping (DTW)
            }
            
            # D. RECORDING (Storage)
            if recorder.is_recording:
                recorder.log(metrics)
                
                # Visual Indicator (UI)
                cv2.circle(image, (w-30, 30), 10, (0, 0, 255), -1) # Recording LED
                cv2.putText(image, "REC", (w-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # E. AUGMENTED REALITY (Visualization)
            # Dashboard: Speed
            cv2.putText(image, f"VELOCITY: {int(speed)} px/s", (w-350, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Dashboard: Kime Message (Temporary Overlay)
            if time.time() - kime_timer < 0.5:
                cv2.putText(image, kime_msg, (w//2 - 150, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Dashboard: Stability Core
            cv2.circle(image, (com_x, com_y), 8, (0, 0, 255), -1)
            cv2.putText(image, f"CORE: {stab_status}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 2)
            
            # Dashboard: Stance Classification
            cv2.putText(image, f"STANCE: {stance_status}", (20, h-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)

            # Biometrics: Knee Angles
            l_knee_pos = detector.get_3d_coordinates(image, 25)
            r_knee_pos = detector.get_3d_coordinates(image, 26)
            cv2.putText(image, f"{int(ang_l)}", (int(l_knee_pos[0]), int(l_knee_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"{int(ang_r)}", (int(r_knee_pos[0]), int(r_knee_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show Viewport
        cv2.imshow('Karate AI Coach (Phase 5)', image)
        
        # --- 3. INPUT HANDLING ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            # Graceful Shutdown
            if recorder.is_recording: recorder.stop()
            break
        elif key == ord('r'):
            # Toggle Recording State
            if recorder.is_recording:
                recorder.stop()
                print("[INFO] Recording Stopped. Data buffered.")
            else:
                recorder.start()
                print("[INFO] Recording Started. Capturing telemetry...")

    # --- 4. POST-PROCESSING & WKF SCORING (The AI Judge) ---
    print("\n" + "="*60)
    print("          WKF EVALUATION PROTOCOL")
    print("="*60)
    
    cap.release()
    cv2.destroyAllWindows()

    # Retrieve the last session file
    session_file = recorder.save_session()
    
    if session_file:
        try:
            print(f"[PROCESS] Loading telemetry from: {session_file}")
            df = pd.read_csv(session_file)
            
            if len(df) < 10:
                print("[WARNING] Insufficient data for valid scoring (<10 frames).")
            else:
                print("[PROCESS] Aggregating performance metrics...")
                
                # Metric Aggregation (Heuristic Logic)
                # 1. Find Max Speed
                max_speed_session = df['speed'].max()
                
                # 2. Check for Kime (Boolean OR across session)
                kime_achieved = 1 in df['is_kime'].values
                
                # 3. Check Stability (Did we ever go UNSTABLE?)
                stability_fault = "UNSTABLE" in df['stability_status'].values
                final_stability = "UNSTABLE" if stability_fault else "STABLE"
                
                # 4. Average Zenkutsu Angle (Only consider frames where user attempted Zenkutsu)
                # This prevents "NEUTRAL" standing frames from ruining the average.
                zenkutsu_frames = df[df['stance_status'].str.contains('ZENKUTSU', na=False)]
                avg_zenkutsu = 180 # Default
                if not zenkutsu_frames.empty:
                    # We take the bent knee (minimum angle of the two legs)
                    avg_zenkutsu = zenkutsu_frames[['knee_angle_l', 'knee_angle_r']].min(axis=1).mean()
                
                # Compile Metrics for the Judge
                session_summary = {
                    'max_speed': max_speed_session,
                    'kime_detected': kime_achieved,
                    'stability_status': final_stability,
                    'zenkutsu_angle': avg_zenkutsu
                }
                
                # CALL THE JUDGE
                verdict = judge.evaluate_performance(session_summary, user_df=df)    

                # REPORT GENERATION
                print("\n" + "-"*40)
                print(f" OFFICIAL WKF SCORE: {verdict['score']} / 10.0")
                print(f" RANKING LEVEL     : {verdict['rank']}")
                print("-" * 40)
                print(" TECHNICAL FEEDBACK:")
                for note in verdict['feedback']:
                    print(f" > {note}")
                print("-" * 40)
                analyze_session(session_file)
                
        except Exception as e:
            print(f"[ERROR] Failed to analyze session data: {e}")
    else:
        print("[INFO] No recording data found. Practice session ended without scoring.")

if __name__ == "__main__":
    main()