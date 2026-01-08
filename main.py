import cv2
import time
import sys
import pandas as pd
import numpy as np
from visualize_data import analyze_session
from sensors.stereo import StereoManager 

# --- CUSTOM MODULES (Architecture Layers) ---
from sensors.pose_detector import PoseDetector      # Perception Layer
from analysis.mechanics import MechanicsAnalyzer    # Logic Layer (Hybrid C++/Python)
from data.recorder import DataRecorder              # Storage Layer
from ui.ai_judge import WKF_Judge                   # Evaluation Layer (Phase 5)
import config                                       # Global Configuration

def main():
    """
    Main Execution Pipeline for AI Karate Coach (Phase 7 - Sensor Fusion).
    Architecture: Dual-Stream Perception with Late Fusion Strategy.
    """

    # --- 1. SYSTEM INITIALIZATION ---
    print("\n[INIT] 🚀 Booting AI Karate Coach (Phase 7 Engine)...")
    print("[INIT] Initializing Sensor Array...")
    
    try:
        # source_a=0 (Laptop Webcam), source_b=1 (Mobile via Iriun/DroidCam)
        # Αν δεν βρει τη 2η κάμερα, το σύστημα θα τρέξει σε Single Mode αυτόματα.
        camera_system = StereoManager(source_a=0, source_b=1)
    except Exception as e:
        print(f"[CRITICAL ERROR] Camera Init Failed: {e}")
        return

    # --- DUAL AI ARCHITECTURE ---
    print("[INIT] Loading Neural Networks & Physics Engines...")
    
    # 1. Perception Layer (Independent Detectors to prevent state corruption)
    detector_main = PoseDetector() 
    detector_side = PoseDetector()

    # 2. Logic Layer (Independent Physics Engines for correct velocity calculation)
    # analyzer_main: Υπολογίζει ταχύτητα/ευστάθεια από μπροστά.
    # analyzer_side: Υπολογίζει ταχύτητα από το πλάι (προφίλ).
    analyzer_main = MechanicsAnalyzer()
    analyzer_side = MechanicsAnalyzer()

    # 3. Storage & Evaluation
    recorder = DataRecorder()
    judge = WKF_Judge()

    # State Variables
    kime_timer = 0
    kime_msg = ""
    
    print(f"[READY] System Online. Fusion Logic: ACTIVE.")
    print(f"[READY] WKF Speed Threshold: >{config.WKF_SPEED_EXCELLENT} px/s")
    print("[INSTRUCT] Press 'r' to toggle Recording | Press 'q' to Quit & Score.\n")

    # --- 2. REAL-TIME PROCESSING LOOP ---
    while True:
        # A. SENSING (Synchronized Capture)
        frame_a, frame_b = camera_system.get_dual_frames()
        
        # Αν χαθεί η κύρια κάμερα, τερματίζουμε
        if frame_a is None:
            print("[WARNING] Main Sensor Signal Lost. Shutting down.")
            break

        # Mirror Main View (Standard UX)
        image_main = cv2.flip(frame_a, 1)
        h_main, w_main, _ = image_main.shape
        
        # --- B. PERCEPTION & PHYSICS (CORE 1: FRONT) ---
        image_main = detector_main.find_pose(image_main)
        landmarks_main = detector_main.get_landmarks()
        
        # Init Variables
        speed_main = 0
        is_kime_main = False
        com_x, com_y = 0, 0
        stab_status, stab_color = "STABLE", (0, 255, 0)
        stance_status, stance_color = "NEUTRAL", (200, 200, 200)
        ang_l, ang_r = 0, 0

        if landmarks_main:
            # 1. Kinematics (Front)
            wrist_3d_main = detector_main.get_3d_coordinates(image_main, 16)
            speed_main, is_kime_main = analyzer_main.track_speed_kime(wrist_3d_main)
            
            # 2. Stability (Front is best for CoM)
            com_x, com_y, stab_status, stab_color, _ = analyzer_main.track_stability(landmarks_main, w_main, h_main)
            
            # 3. Stance (Front is best for Angles)
            stance_status, stance_color, ang_l, ang_r = analyzer_main.track_stance(landmarks_main)

        # --- C. PERCEPTION & PHYSICS (CORE 2: SIDE) ---
        speed_side = 0
        is_kime_side = False
        image_side_disp = None

        if frame_b is not None:
            # Mirror Side View (Optional, but keeps consistency)
            image_side = cv2.flip(frame_b, 1)
            
            # Detect Pose (Full Resolution for Physics)
            image_side = detector_side.find_pose(image_side) 
            landmarks_side = detector_side.get_landmarks()
            
            if landmarks_side:
                # 1. Kinematics (Side) - Εδώ πιάνουμε την κίνηση βάθους (Z-axis motion)
                wrist_3d_side = detector_side.get_3d_coordinates(image_side, 16)
                speed_side, is_kime_side = analyzer_side.track_speed_kime(wrist_3d_side)
            
            # Create Resize Display Image (Optimization)
            image_side_disp = cv2.resize(image_side, (640, 480))

        # --- D. SENSOR FUSION (DECISION LAYER) ---
        # Logic: "Winner Takes All" -> The camera with the best view of the velocity vector wins.
        final_speed = max(speed_main, speed_side)
        
        # Logic: Kime is a global event. If ANY sensor detects snap, it counts.
        is_kime = is_kime_main or is_kime_side

        if is_kime:
            kime_timer = time.time()
            kime_msg = "KIME DETECTED!"

        # --- E. VISUALIZATION (GUI) ---
        
        # 1. MAIN WINDOW (Augmented Reality)
        # Velocity Display (Fused)
        color_speed = (0, 255, 255) if final_speed == speed_side else (255, 255, 255) # Yellow if Side Cam won
        cv2.putText(image_main, f"FUSION SPEED: {int(final_speed)} px/s", (w_main-400, h_main-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_speed, 3)
        
        # Debug Data (Sensor Breakdown)
        cv2.putText(image_main, f"[CAM A: {int(speed_main)} | CAM B: {int(speed_side)}]", (w_main-380, h_main-70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Kime Flash
        if time.time() - kime_timer < 0.5:
            cv2.putText(image_main, kime_msg, (w_main//2 - 150, h_main//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if landmarks_main:
            # Stability Core
            cv2.circle(image_main, (com_x, com_y), 8, (0, 0, 255), -1)
            cv2.putText(image_main, f"CORE: {stab_status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stab_color, 2)
            
            # Stance Info
            cv2.putText(image_main, f"STANCE: {stance_status}", (20, h_main-30), cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)
            
            # Biometrics (Knees)
            l_pos = detector_main.get_3d_coordinates(image_main, 25)
            r_pos = detector_main.get_3d_coordinates(image_main, 26)
            cv2.putText(image_main, f"{int(ang_l)}", (int(l_pos[0]), int(l_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(image_main, f"{int(ang_r)}", (int(r_pos[0]), int(r_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 2. SIDE WINDOW (Secondary View)
        if image_side_disp is not None:
            # Draw local metrics just for verification
            cv2.putText(image_side_disp, f"SIDE VELOCITY: {int(speed_side)} px/s", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image_side_disp, "SIDE VIEW (Analysis Active)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Karate AI - Secondary (Side)', image_side_disp)

        # Show Main
        cv2.imshow('Karate AI - Primary (Front)', image_main)

        # --- F. RECORDING LOGIC ---
        metrics = {
            "speed": final_speed,
            "is_kime": 1 if is_kime else 0,
            "com_x": com_x, "com_y": com_y,
            "stance_status": stance_status,
            "knee_angle_l": ang_l, "knee_angle_r": ang_r,
            "stability_status": stab_status
        }
        
        if recorder.is_recording:
            recorder.log(metrics)
            # Rec Indicator
            cv2.circle(image_main, (w_main-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(image_main, "REC", (w_main-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- G. INPUT HANDLING ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if recorder.is_recording: recorder.stop()
            break
        elif key == ord('r'):
            if recorder.is_recording:
                recorder.stop()
                print("[INFO] Recording Stopped. Data buffered.")
            else:
                recorder.start()
                print("[INFO] Recording Started. Capturing telemetry...")

    # --- 3. CLEANUP & SCORING PROTOCOL ---
    print("\n[SYSTEM] Shutting down sensors...")
    camera_system.stop()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("          WKF EVALUATION PROTOCOL (AI JUDGE)")
    print("="*60)
    
    # Save & Load Data
    session_file = recorder.save_session()
    
    if session_file:
        try:
            print(f"[PROCESS] Loading telemetry from: {session_file}")
            df = pd.read_csv(session_file)
            
            if len(df) < 10:
                print("[WARNING] Insufficient data for valid scoring (<10 frames).")
            else:
                print("[PROCESS] Aggregating performance metrics...")
                
                # Metric Aggregation
                max_speed_session = df['speed'].max()
                kime_achieved = 1 in df['is_kime'].values
                
                # Stability Check
                stability_fault = "UNSTABLE" in df['stability_status'].values
                final_stability = "UNSTABLE" if stability_fault else "STABLE"
                
                # Zenkutsu Analysis
                zenkutsu_frames = df[df['stance_status'].str.contains('ZENKUTSU', na=False)]
                avg_zenkutsu = 180 
                if not zenkutsu_frames.empty:
                    avg_zenkutsu = zenkutsu_frames[['knee_angle_l', 'knee_angle_r']].min(axis=1).mean()
                
                # Summary Construction
                session_summary = {
                    'max_speed': max_speed_session,
                    'kime_detected': kime_achieved,
                    'stability_status': final_stability,
                    'zenkutsu_angle': avg_zenkutsu
                }
                
                # CALL THE JUDGE
                verdict = judge.evaluate_performance(session_summary, user_df=df)    

                # PRINT REPORT
                print("\n" + "-"*40)
                print(f" OFFICIAL WKF SCORE: {verdict['score']} / 10.0")
                print(f" RANKING LEVEL     : {verdict['rank']}")
                print("-" * 40)
                print(" TECHNICAL FEEDBACK:")
                for note in verdict['feedback']:
                    print(f" > {note}")
                print("-" * 40)
                
                # SHOW GRAPHS
                analyze_session(session_file)
                
        except Exception as e:
            print(f"[ERROR] Failed to analyze session data: {e}")
    else:
        print("[INFO] No recording data found. Practice session ended without scoring.")

if __name__ == "__main__":
    main()