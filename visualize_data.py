import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

"""
Offline Data Visualizer (Science Module)
----------------------------------------
Reads CSV telemetry from 'data/recorder.py' and generates 
scientific plots for biomechanical analysis.

Purpose:
- Validate the quality of 'Golden Reference' data.
- Mimic Signal Processing analysis (Velocity vs Time).
"""

def analyze_session(csv_path):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return

    print(f"[INFO] Analyzing Session: {csv_path}")
    print(f"[INFO] Total Frames: {len(df)}")
    print(f"[INFO] Max Speed: {df['speed'].max():.2f}")

    # 2. Setup Plots (Subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # --- PLOT 1: KINEMATICS (Speed) ---
    ax1.plot(df['timestamp'], df['speed'], label='Wrist Speed', color='blue', linewidth=2)
    ax1.set_title('Kinematics: Velocity Profile')
    ax1.set_ylabel('Speed (px/s)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight Kime points (Impacts)
    kime_points = df[df['is_kime'] == 1]
    ax1.scatter(kime_points['timestamp'], kime_points['speed'], color='red', s=100, label='Kime Impact', zorder=5)
    ax1.legend()

    # --- PLOT 2: STABILITY (Center of Mass) ---
    # We plot Vertical Oscillation (CoM Y-axis)
    # In computer vision, Y increases downwards, so we might invert for intuitive graph
    ax2.plot(df['timestamp'], df['com_y'], label='Vertical CoM', color='green')
    ax2.set_title('Stability: Vertical Oscillation')
    ax2.set_ylabel('Vertical Position (px)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight Unstable moments
    # Assuming 'stability_status' is a string, we map it to colors if needed
    # (Simplified for now)

    # --- PLOT 3: STANCE GEOMETRY (Knees) ---
    ax3.plot(df['timestamp'], df['knee_angle_l'], label='Left Knee', color='orange', linestyle='--')
    ax3.plot(df['timestamp'], df['knee_angle_r'], label='Right Knee', color='purple', linestyle='--')
    
    # Draw Threshold Lines
    ax3.axhline(y=140, color='red', linestyle=':', alpha=0.5, label='Max Bent Threshold (140°)')
    ax3.axhline(y=160, color='green', linestyle=':', alpha=0.5, label='Min Straight Threshold (160°)')
    
    ax3.set_title('Geometry: Zenkutsu-dachi Angles')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.legend(loc='lower right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 3. Show Dashboard
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Usage: python visualize_data.py sessions/karate_session_XXXX.csv
    # If no argument, find the latest file automatically
    
    target_file = ""
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Find latest csv in sessions/
        session_dir = "sessions"
        if not os.path.exists(session_dir):
            print("[ERROR] No 'sessions' folder found. Record some data first!")
            sys.exit()
            
        files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('.csv')]
        if not files:
            print("[ERROR] No CSV files found.")
            sys.exit()
            
        target_file = max(files, key=os.path.getctime) # Get newest file

    analyze_session(target_file)