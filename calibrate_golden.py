import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
FILE_1 = 'sessions/golden_reference_data_1.csv'
FILE_2 = 'sessions/golden_reference_data_2.csv'

def process_file(filepath, label):
    """Loads, cleans, and extracts metrics from a CSV."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {filepath}")
        return None

    # 1. Signal Smoothing (Savitzky-Golay)
    # Window length 13, Polyorder 3 is standard for biomechanics
    if len(df) > 15:
        df['speed_smooth'] = savgol_filter(df['speed'], window_length=13, polyorder=3)
        # Smooth knee angles if they exist
        if 'knee_angle_l' in df.columns:
            df['knee_angle_l_smooth'] = savgol_filter(df['knee_angle_l'], window_length=13, polyorder=3)
    else:
        df['speed_smooth'] = df['speed'] # Not enough data to smooth

    # 2. Extract Metrics (Feature Extraction)
    metrics = {
        'label': label,
        'max_speed': df['speed_smooth'].max(),
        'avg_speed': df['speed_smooth'].mean(),
        'kime_count': df['is_kime'].sum(),
        'dataframe': df
    }
    
    # Extract Stance Metrics (Average Knee Angle during Zenkutsu)
    # Filters rows where the logic detected "ZENKUTSU"
    if 'status' in df.columns:
        zenkutsu_frames = df[df['status'].str.contains('ZENKUTSU', na=False)]
        if not zenkutsu_frames.empty:
            # We take the angle of the bent leg (assuming logic stores it in knee_angle_l/r)
            # This is a simplification; ideally we check which leg is bent.
            # Taking the MIN angle usually represents the bent knee in Zenkutsu.
            avg_bent_knee = zenkutsu_frames[['knee_angle_l_smooth', 'knee_angle_r_smooth']].min(axis=1).mean()
            metrics['zenkutsu_angle'] = avg_bent_knee
        else:
            metrics['zenkutsu_angle'] = None
    
    return metrics

# --- MAIN EXECUTION ---
data1 = process_file(FILE_1, "Golden Ref 1")
data2 = process_file(FILE_2, "Golden Ref 2")

if data1 and data2:
    # --- 1. VISUAL COMPARISON ---
    plt.figure(figsize=(12, 6))
    
    # Plot Speed Profiles
    plt.plot(data1['dataframe']['speed_smooth'], label=f"Ref 1 (Max: {data1['max_speed']:.0f})", color='blue', alpha=0.8)
    plt.plot(data2['dataframe']['speed_smooth'], label=f"Ref 2 (Max: {data2['max_speed']:.0f})", color='cyan', linestyle='--')
    
    plt.title("Golden Reference Consistency Check: Velocity Profiles")
    plt.xlabel("Frame")
    plt.ylabel("Speed (px/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 2. CALCULATE FINAL STANDARDS ---
    # We average the max speeds to get the robust 100% mark
    final_max_speed = (data1['max_speed'] + data2['max_speed']) / 2
    
    # Average Zenkutsu Angle (if detected in both)
    final_zenkutsu = "N/A"
    if data1['zenkutsu_angle'] and data2['zenkutsu_angle']:
        avg_zenkutsu = (data1['zenkutsu_angle'] + data2['zenkutsu_angle']) / 2
        final_zenkutsu = f"{avg_zenkutsu:.1f}"
    elif data1['zenkutsu_angle']:
        final_zenkutsu = f"{data1['zenkutsu_angle']:.1f}"

    # --- 3. OUTPUT FOR CONFIG.PY ---
    print("\n" + "="*40)
    print("   GENERATE CONFIGURATION (Copy this!)")
    print("="*40)
    print(f"# Derived from {FILE_1} and {FILE_2}")
    print(f"GOLDEN_MAX_SPEED = {final_max_speed:.1f}  # The 10.0 Speed Score")
    print(f"WKF_EXCELLENT_THRESHOLD = {final_max_speed * 0.9:.1f} # 9.0+ Score range")
    print(f"GOLDEN_ZENKUTSU_ANGLE = {final_zenkutsu} # Target degrees for bent knee")
    print("="*40 + "\n")

else:
    print("Could not process both files.")