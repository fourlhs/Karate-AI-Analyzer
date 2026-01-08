import cv2
import time
from sensors.pose_detector import PoseDetector
from analysis.mechanics import MechanicsAnalyzer
from data.recorder import DataRecorder

def main():
    # --- SETUP ---
    cap = cv2.VideoCapture(0)
    
    detector = PoseDetector()       
    analyzer = MechanicsAnalyzer()  
    recorder = DataRecorder()
    
    kime_timer = 0
    kime_msg = ""
    
    print("Press 'r' to START/STOP Recording.")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # 1. SENSE
        image = detector.find_pose(image)
        landmarks = detector.get_landmarks()
        
        # Default metrics (in case no landmarks found)
        metrics = {} 
        
        if landmarks:
            # --- ANALYSIS ---
            wrist_3d = detector.get_3d_coordinates(image, 16)
            speed, is_kime = analyzer.track_speed_kime(wrist_3d)
            com_x, com_y, stab_status, stab_color, history = analyzer.track_stability(landmarks, w, h)
            stance_status, stance_color, ang_l, ang_r = analyzer.track_stance(landmarks)
            
            # Kime Visuals
            if is_kime:
                kime_timer = time.time()
                kime_msg = "KIME! PERFECT!"

            # --- PREPARE DATA FOR RECORDING ---
            metrics = {
                "speed": speed,
                "is_kime": 1 if is_kime else 0,
                "com_x": com_x,
                "com_y": com_y,
                "stance_status": stance_status,
                "knee_angle_l": ang_l,
                "knee_angle_r": ang_r,
                "stability_status": stab_status
                # In the future, we will add raw landmarks here for DTW
            }
            
            # --- VISUALIZATION ---
            # Recording Indicator (Blimp)
            if recorder.is_recording:
                cv2.circle(image, (w-30, 30), 10, (0, 0, 255), -1) # Red Dot
                cv2.putText(image, "REC", (w-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                recorder.log(metrics) # <-- LOG DATA

            # Drawing (Standard UI)
            cv2.putText(image, f"SPEED: {int(speed)}", (w-250, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if time.time() - kime_timer < 0.5:
                cv2.putText(image, kime_msg, (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            cv2.circle(image, (com_x, com_y), 8, (0,0,255), -1)
            cv2.putText(image, f"CORE: {stab_status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)
            cv2.putText(image, f"STANCE: {stance_status}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)

            l_knee_pos = detector.get_3d_coordinates(image, 25)
            r_knee_pos = detector.get_3d_coordinates(image, 26)
            cv2.putText(image, str(ang_l), (int(l_knee_pos[0]), int(l_knee_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(image, str(ang_r), (int(r_knee_pos[0]), int(r_knee_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Karate AI (Phase 4 - Data)', image)
        
        # --- CONTROLS ---
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            if recorder.is_recording: recorder.stop() # Safe exit
            break
        elif key == ord('r'):
            if recorder.is_recording:
                recorder.stop()
            else:
                recorder.start()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()