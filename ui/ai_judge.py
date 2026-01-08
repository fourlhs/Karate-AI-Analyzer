import config
from typing import Dict

class WKF_Judge:
    """
    The AI Judge
    Implements WKF Competition Rules 2026 (Article 5: Evaluation).
    Scoring Scale: 5.0 - 10.0 
    """
    
    def __init__(self):
        self.base_score = 10.0 # Perfect performance start
        self.feedback_log = []

    def evaluate_performance(self, user_metrics: Dict) -> Dict:
        """
        Calculates the final score based on Technical Performance (Speed, Stance)
        and Athletic Performance (Balance, Kime).
        """
        current_score = self.base_score
        self.feedback_log = [] # Reset feedback

        # --- CRITERION 1: STANCES (WKF Point 1) --- [cite: 349]
        user_knee_angle = user_metrics.get('zenkutsu_angle', 180) # Default straight if not detected
        
        if not (config.STANCE_ZENKUTSU_KNEE_MIN <= user_knee_angle <= config.STANCE_ZENKUTSU_KNEE_MAX):
            penalty = 0.5
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Stance Error: Knee too high/low ({user_knee_angle:.1f}°). Aim for {config.GOLDEN_ZENKUTSU_ANGLE}°.")

        # --- CRITERION 9: SPEED (WKF Point 9) --- [cite: 349]
        user_speed = user_metrics.get('max_speed', 0)
        
        if user_speed >= config.WKF_SPEED_EXCELLENT:
            self.feedback_log.append("[INFO] Speed: Excellent (9.0-9.9 range)")
            # No deduction
        elif user_speed >= config.WKF_SPEED_VERY_GOOD:
            penalty = 0.3
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Very Good. Push for {config.WKF_SPEED_EXCELLENT:.0f} px/s.")
        elif user_speed >= config.WKF_SPEED_GOOD:
            penalty = 0.7
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Good. Needs more explosive power.")
        else:
            penalty = 1.5
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Speed: Insufficient. Far below Golden Reference.")

        # --- CRITERION 6: KIME (Focus) --- [cite: 349]
        if not user_metrics.get('kime_detected', False):
            penalty = 0.5
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] No Kime Detected: Sharp stop missing at end of technique.")

        # --- CRITERION 10: BALANCE --- [cite: 349]
        # WKF Rule 5.8: "Minor loss of balance" is a foul [cite: 357]
        if user_metrics.get('stability_status') == "UNSTABLE":
            penalty = 0.2
            current_score -= penalty
            self.feedback_log.append(f"[-{penalty}] Stability: Minor loss of balance detected.")

        # --- FINAL SCORE CALCULATION ---
        # WKF Rule 5.4.1: Lowest accepted score is 5.0 
        final_score = max(5.0, round(current_score, 1))
        
        return {
            "score": final_score,
            "feedback": self.feedback_log,
            "rank": self._get_rank_label(final_score)
        }

    def _get_rank_label(self, score):
        # Based on WKF Guidelines 5.5.3 
        if score >= 9.0: return "EXCELLENT"      # [cite: 336]
        if score >= 8.0: return "VERY GOOD"      # [cite: 338]
        if score >= 7.0: return "GOOD"           # [cite: 340]
        if score >= 6.0: return "ACCEPTABLE"     # [cite: 342]
        return "INSUFFICIENT"                    # [cite: 344]