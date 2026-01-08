import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches

def analyze_session(csv_path):
    """
    Biomechanical Analytics Module (Post-Processing).
    
    Generates a multi-dimensional analysis dashboard for Karate performance.
    
    Metrics Analyzed:
    1. Kinematics: Linear Velocity profile (Wrist).
    2. Dynamics: Acceleration/Deceleration profile (Force generation F=ma).
    3. Stability: Center of Mass (CoM) vertical oscillation.
    4. Technique: Joint angle constraints verification (Stance Geometry).
    """
    
    # --- 1. DATA INGESTION & VALIDATION ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Telemetry file not found: {csv_path}")
        return

    # Check for sufficient sample size for signal processing
    if len(df) < 10:
        print("[WARNING] Insufficient data points for spectral analysis.")
        return

    print(f"[ANALYTICS] Processing telemetry: {csv_path}")

    # --- 2. DIGITAL SIGNAL PROCESSING (DSP) LAYER ---
    # We apply Savitzky-Golay filtering to denoise sensor data.
    # Rationale: Unlike moving averages, SG preserves the amplitude and width of 
    # signal transients (peaks), which is critical for detecting 'Kime' (impact).
    
    # A. Velocity Smoothing
    try:
        df['speed_smooth'] = savgol_filter(df['speed'], window_length=13, polyorder=3)
    except:
        df['speed_smooth'] = df['speed'] # Fallback to raw data on failure

    # B. Acceleration Calculation (Newtonian Mechanics)
    # Derivation: a = dv/dt. We use numerical differentiation (gradient) of the velocity vector.
    # Significance: Acceleration is directly proportional to Force (F = m * a).
    df['acceleration'] = np.gradient(df['speed_smooth'])
    
    # Smooth Acceleration signal (Derivative amplifies noise, so smoothing is mandatory)
    try:
        df['accel_smooth'] = savgol_filter(df['acceleration'], window_length=13, polyorder=3)
    except:
        df['accel_smooth'] = df['acceleration']

    # C. Joint Angle Smoothing
    if 'knee_angle_l' in df.columns:
        try:
            df['knee_angle_l_smooth'] = savgol_filter(df['knee_angle_l'], window_length=15, polyorder=3)
        except:
            df['knee_angle_l_smooth'] = df['knee_angle_l']

    # --- 3. DATA VISUALIZATION DASHBOARD ---
    # Layout: 4 Vertical Tracks for time-aligned comparison of metrics.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    plt.subplots_adjust(hspace=0.3) 

    frames = df.index

    # --- TRACK 1: KINEMATICS (VELOCITY) ---
    # Displays the scalar magnitude of velocity over time.
    ax1.plot(frames, df['speed_smooth'], color='#007acc', linewidth=2, label='Velocity')
    ax1.set_title('1. KINEMATICS: Velocity Profile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speed (px/s)')
    ax1.grid(True, alpha=0.3)
    
    # Event Overlay: Kime Detection Points
    # Correlates the physics-based detection with the visual timeline.
    kime_pts = df[df['is_kime'] == 1]
    if not kime_pts.empty:
        ax1.scatter(kime_pts.index, kime_pts['speed_smooth'], color='red', s=80, zorder=5, label='Kime Impact')
    ax1.legend(loc='upper right')

    # --- TRACK 2: DYNAMICS (ACCELERATION / FORCE) ---
    # Visualizes the "Explosion" vs "Control" phases.
    # Positive peak = Initial Impulse (Power generation).
    # Negative peak = Impact/Snap back (Deceleration phase).
    ax2.plot(frames, df['accel_smooth'], color='#e67e22', linewidth=2, label='Acceleration')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--') # Equilibrium line
    
    ax2.set_title('2. FORCE GENERATION: Acceleration (Proxy for F=ma)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accel (px/s²)')
    
    # Semantic coloring for Power vs Braking phases
    ax2.fill_between(frames, df['accel_smooth'], 0, where=(df['accel_smooth'] > 0), color='#e67e22', alpha=0.3, label='Power Gen')
    ax2.fill_between(frames, df['accel_smooth'], 0, where=(df['accel_smooth'] < 0), color='red', alpha=0.1, label='Braking/Impact')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # --- TRACK 3: STABILITY (CENTER OF MASS) ---
    # Analyzes vertical displacement efficiency. Minimizing Y-oscillation is crucial for energy transfer.
    ax3.plot(frames, df['com_y'], color='green', label='Vertical CoM')
    
    # Fault Detection Overlay
    if 'stability_status' in df.columns:
        unstable = df['stability_status'] == 'UNSTABLE'
        ax3.fill_between(frames, ax3.get_ylim()[0], ax3.get_ylim()[1], where=unstable, 
                         color='red', alpha=0.2, label='Balance Loss')

    ax3.set_title('3. STABILITY: Vertical Oscillation Analysis', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Height (px)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # --- TRACK 4: TECHNIQUE (STANCE GEOMETRY) ---
    # Verifies biomechanical alignment against WKF standards.
    if 'knee_angle_l_smooth' in df.columns:
        ax4.plot(frames, df['knee_angle_l_smooth'], color='purple', linewidth=2, label='Left Knee Angle')
        
        # Tolerance Band Visualization (Golden Reference: 90°-110°)
        ax4.axhspan(90, 110, color='green', alpha=0.15, label='Optimal Biomechanical Range')
        ax4.axhline(90, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(110, color='green', linestyle='--', alpha=0.5)

        # Logic Verification: Visualize AI Classification state
        if 'stance_status' in df.columns:
            is_zenkutsu = df['stance_status'].str.contains('ZENKUTSU', na=False)
            ax4.fill_between(frames, 0, 180, where=is_zenkutsu, color='blue', alpha=0.05, label='AI Detected Valid Stance')

    ax4.set_title('4. TECHNIQUE: Stance Geometry & Compliance', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Angle (°)')
    ax4.set_ylim(50, 180) 
    ax4.set_xlabel('Temporal Domain (Frames)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right')

    # --- 4. RENDER OUTPUT ---
    plt.suptitle(f"WKF BIOMECHANICAL REPORT\nDataset: {os.path.basename(csv_path)}", fontsize=14)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_session(sys.argv[1])
    else:
        print("Usage: python visualize_data.py <csv_file>")