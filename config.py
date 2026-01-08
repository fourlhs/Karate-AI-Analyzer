# WKF Rules 2026 - Calibration Constants
# Generated from Golden Reference Analysis (Stefanos/Georgina Data)

# --- SPEED METRICS (Derived from Max Peaks: 4257 & 3996) ---
GOLDEN_MAX_SPEED = 4126.0  # Το "10.0" στην ταχύτητα
WKF_SPEED_EXCELLENT = 3713.0  # 90% του Golden (Score: 9.0 - 9.9) 
WKF_SPEED_VERY_GOOD = 3300.0  # 80% του Golden (Score: 8.0 - 8.9) 
WKF_SPEED_GOOD = 2888.0       # 70% του Golden (Score: 7.0 - 7.9) 

# --- STANCE METRICS (Angles in degrees) ---
# Zenkutsu Dachi Logic 
STANCE_ZENKUTSU_KNEE_MIN = 90
STANCE_ZENKUTSU_KNEE_MAX = 110
STANCE_PENALTY_FACTOR = 0.3
GOLDEN_ZENKUTSU_ANGLE = 98.5

# --- SCORING RULES ---
BASE_SCORE = 10.0
MIN_SCORE = 5.0  # Lowest score for accepted performance [cite: 298]
DISQUALIFICATION_SCORE = 0.0  # [cite: 299]