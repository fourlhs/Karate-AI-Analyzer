import cv2
import time
from sensors.pose_detector import PoseDetector
from analysis.mechanics import MechanicsAnalyzer

def main():
    # --- SETUP (Phase 3 Architecture) ---
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()       
    analyzer = MechanicsAnalyzer()  
    
    kime_timer = 0
    kime_msg = ""
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # 1. SENSE
        image = detector.find_pose(image)
        landmarks = detector.get_landmarks()
        
        if landmarks:
            # --- ARM ANALYSIS (Speed & Kime) ---
            wrist_3d = detector.get_3d_coordinates(image, 16)
            speed, is_kime = analyzer.track_speed_kime(wrist_3d)
            if is_kime:
                kime_timer = time.time()
                kime_msg = "KIME! PERFECT!"

            # --- CORE ANALYSIS (Stability) ---
            com_x, com_y, stab_status, stab_color, history = analyzer.track_stability(landmarks, w, h)

            # --- LEG ANALYSIS (Stance) ---
            stance_status, stance_color, ang_l, ang_r = analyzer.track_stance(landmarks)
            
            # Find Knee Coordinates for visualization (Index 25=Left, 26=Right)
            l_knee_pos = detector.get_3d_coordinates(image, 25)
            r_knee_pos = detector.get_3d_coordinates(image, 26)


            # ================= VISUALIZATION UI =================
            
            # 1. Speed & Kime
            cv2.putText(image, f"SPEED: {int(speed)}", (w-250, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if time.time() - kime_timer < 0.5:
                cv2.putText(image, kime_msg, (w//2-150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

            # 2. Stability Trajectory
            cv2.circle(image, (com_x, com_y), 8, (0,0,255), -1)
            cv2.putText(image, f"CORE: {stab_status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stab_color, 2)
            for i in range(1, len(history)):
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                cv2.line(image, pt1, pt2, (0, 255, 255), 2)

            # 3. Stance Status & Angles
            cv2.putText(image, f"STANCE: {stance_status}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, stance_color, 2)
            
            # Draw angles on knees
            cv2.putText(image, str(ang_l), (int(l_knee_pos[0]), int(l_knee_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(image, str(ang_r), (int(r_knee_pos[0]), int(r_knee_pos[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow('Karate AI (Full Body)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()