import config
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
from typing import Dict, List, Optional

class WKF_Judge:
    """
    Phase 5.5: The Intelligent Judge (Hybrid Evaluation Engine).
    
    Combines Deterministic Rules (WKF Regulations) with Stochastic ML Analysis (DTW).
    
    Architectural Layers:
    1. Rule Engine: Checks absolute thresholds (Speed > X, Angle in [A, B]).
    2. ML Engine: Performs Dynamic Time Warping (DTW) to evaluate the 'Rhythm' 
       and 'Shape' of the motion curve against a Golden Reference.
    
    Goal: Eliminate subjectivity by quantifying 'Kata Flow' and 'Technical Consistency'.
    """
    
    def __init__(self):
        self.base_score = 10.0
        self.feedback_log = []
        
        # --- ML INIT: Load Golden Reference Model ---
        # Pre-loads the 'Perfect' speed curve into memory for fast comparison.
        self.golden_speed_curve = self._load_golden_curve()

    def _load_golden_curve(self) -> Optional[np.ndarray]:
        """
        Ingests the Golden Reference Dataset for Machine Learning comparison.
        Returns: Numpy array of the velocity profile.
        """
        # We use Ref 1 as the primary baseline for 'Shape Matching'
        path = 'data/references/golden_reference_data_1.csv'
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Fill NaNs to prevent DTW crash. 
                # We extract the 'speed' feature vector.
                print(f"[AI JUDGE] ML Model loaded: {path}")
                return df['speed'].fillna(0).to_numpy()
            else:
                print(f"[AI JUDGE WARNING] Baseline data missing at {path}. DTW will be disabled.")
                return None
        except Exception as e:
            print(f"[AI JUDGE ERROR] Failed to initialize ML core: {e}")
            return None

    def evaluate_performance(self, user_metrics: Dict, user_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Executes the Scoring Pipeline.
        
        Args:
            user_metrics: Scalar values (Max Speed, Stance Angle) for Rule Checks.
            user_df: Full time-series telemetry for ML Curve Matching.
            
        Returns:
            Dict containing Final Score, Rank, and Granular Feedback.
        """
        current_score = self.base_score
        self.feedback_log = [] # Reset per session

        # =========================================================
        # LAYER 1: DETERMINISTIC RULE ENGINE (WKF Criteria 5.4)
        # =========================================================
        
        # --- CRITERION A: ATHLETIC PERFORMANCE (Speed/Power) ---
        user_speed = user_metrics.get('max_speed', 0)
        
        if user_speed >= config.WKF_SPEED_EXCELLENT:
             self.feedback_log.append("[INFO] Kime/Speed: Excellent (Elite Level).")
        elif user_speed >= config.WKF_SPEED_VERY_GOOD:
            penalty = 0.3
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Very Good. Push for {config.WKF_SPEED_EXCELLENT:.0f} px/s.")
        elif user_speed >= config.WKF_SPEED_GOOD:
            penalty = 0.7
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Good. Lacks explosive finish.")
        else:
            penalty = 1.5
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Insufficient for WKF standards.")

        # --- CRITERION B: TECHNICAL FORM (Stances) ---
        # Logic: Compare user knee angle vs Golden Reference (98.5 deg)
        user_knee = user_metrics.get('zenkutsu_angle', 180)
        
        # Tolerance: +/- 15 degrees is acceptable variance
        if abs(user_knee - config.GOLDEN_ZENKUTSU_ANGLE) > 15:
            penalty = 0.5
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Stance Error: Knee angle {user_knee:.1f}°. Target: {config.GOLDEN_ZENKUTSU_ANGLE}°.")

        # --- CRITERION C: STABILITY (Balance) ---
        if user_metrics.get('stability_status') == "UNSTABLE":
            penalty = 0.3
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Stability: Core balance loss detected.")

        # =========================================================
        # LAYER 2: MACHINE LEARNING ENGINE (Dynamic Time Warping)
        # =========================================================
        # Algorithms: FastDTW (O(N) complexity approximation)
        # Purpose: Compares the 'Flow' and 'Rhythm' regardless of total duration.
        
        if user_df is not None and self.golden_speed_curve is not None:
            # 1. Extract Feature Vector
            user_curve = user_df['speed'].fillna(0).to_numpy()
            
            # 2. Run DTW
            # Returns 'distance' (dissimilarity cost) and 'path' (warp path)
            distance, path = fastdtw(self.golden_speed_curve, user_curve)   
                     
            # 3. Normalize Score
            # Raw distance depends on length. We normalize by path length to get 'Average Error per Frame'.
            similarity_cost = distance / len(path)
            
            # 4. ML Evaluation Logic
            # Thresholds are empirical. Lower is better.
            # < 40: Excellent Match | 40-70: Good | > 70: Different Technique
            if similarity_cost > 70.0:
                ml_penalty = 0.5
                current_score -= ml_penalty
                self.feedback_log.append(f"[-{ml_penalty}] AI Analysis: Motion curve differs significantly from Pro (DTW Cost: {similarity_cost:.1f}).")
            elif similarity_cost > 40.0:
                self.feedback_log.append(f"[INFO] AI Analysis: Good rhythm match, minor timing variances (DTW Cost: {similarity_cost:.1f}).")
            else:
                self.feedback_log.append(f"[AI MATCH] PERFECT FLOW! Your motion curve matches the Champion (DTW Cost: {similarity_cost:.1f}).")

        # =========================================================
        # FINAL VERDICT GENERATION
        # =========================================================
        # Clamp score to official WKF range [5.0 - 10.0]
        final_score = max(5.0, round(current_score, 1))
        
        return {
            "score": final_score,
            "feedback": self.feedback_log,
            "rank": self._get_rank_label(final_score)
        }

    def _get_rank_label(self, score):
        """Maps numerical score to qualitative WKF label."""
        if score >= 9.0: return "EXCELLENT"
        if score >= 8.0: return "VERY GOOD"
        if score >= 7.0: return "GOOD"
        if score >= 6.0: return "ACCEPTABLE"
        return "INSUFFICIENT"