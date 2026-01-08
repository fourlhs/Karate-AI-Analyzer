#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Essential for converting std::vector/tuple to Python lists
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <numeric>
#include <algorithm>
#include <tuple>

namespace py = pybind11;

// --- ENGINEERING CONSTANTS ---
// High-precision PI for vector geometry calculations
const double PI = 3.14159265358979323846;

// --- DATA STRUCTURES ---
struct Point3D {
    double x, y, z;
};

struct Point2D {
    double x, y;
};

/**
 * @brief Core Physics & Biomechanics Engine (High-Performance C++ Implementation).
 * * This class replaces the Python 'MechanicsAnalyzer'.
 * It handles Kinematics (Speed/Kime), Statics (Center of Mass), and 
 * Morphology (Stance Classification) in real-time (<1ms latency).
 */
class KarateEngine {
private:
    // --- INTERNAL STATE (Kinematics) ---
    Point3D prev_wrist;
    double prev_time;
    double prev_speed;
    
    // --- INTERNAL STATE (Stability) ---
    // Circular buffer for Center of Mass (CoM) Y-coordinates.
    // Equivalent to Python's deque(maxlen=30).
    std::deque<double> com_history;
    const size_t STABILITY_BUFFER_SIZE = 30;
    const double VERTICAL_LIMIT = 40.0;

    // --- CONFIGURATION (Calibration) ---
    const double MIN_SPEED_FOR_KIME = 1100.0;
    const double MIN_ACCEL_FOR_KIME = -9500.0;

public:
    KarateEngine() {
        reset();
    }

    // =========================================================================
    // MODULE 1: VECTOR GEOMETRY
    // =========================================================================

    /**
     * @brief Calculates the 2D inner angle between three points (A-B-C).
     * Used for biomechanical joint analysis (e.g., Knee flexion).
     */
    double calculate_angle(std::vector<double> a, std::vector<double> b, std::vector<double> c) {
        // Math: atan2(y, x) computes the angle relative to X-axis.
        // We take the difference between the two vector angles.
        double ang = atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]);
        
        double angle_deg = std::abs(ang * 180.0 / PI);
        
        // Normalize to inner angle [0, 180]
        if (angle_deg > 180.0) angle_deg = 360.0 - angle_deg;
        return angle_deg;
    }

    // =========================================================================
    // MODULE 2: KINEMATICS (SPEED & KIME)
    // =========================================================================

    /**
     * @brief Tracks velocity and detects Kime (Impulse).
     * Returns: (Current Speed, Is Kime Detected)
     */
    std::pair<double, bool> track_speed_kime(double x, double y, double z, double timestamp) {
        double dt = timestamp - prev_time;
        double current_speed = 0.0;
        bool is_kime = false;

        // Protection against first frame or zero-time delta
        if (dt > 0.001 && prev_time > 0) { 
            double dx = x - prev_wrist.x;
            double dy = y - prev_wrist.y;
            double dz = z - prev_wrist.z;
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

            double raw_speed = distance / dt;

            // Exponential Moving Average (EMA) for noise reduction
            // S_t = alpha * Y_t + (1-alpha) * S_t-1
            current_speed = 0.7 * prev_speed + 0.3 * raw_speed;
            
            // Noise Gate
            if (current_speed < 50.0) current_speed = 0.0;

            // Acceleration: a = dv/dt
            double acceleration = (current_speed - prev_speed) / dt;

            // Kime Logic: High Speed + Sudden Braking
            if (prev_speed > MIN_SPEED_FOR_KIME && acceleration < MIN_ACCEL_FOR_KIME) {
                is_kime = true;
            }
        }

        // State Update
        prev_wrist = {x, y, z};
        prev_speed = current_speed;
        prev_time = timestamp;

        return {current_speed, is_kime};
    }

    // =========================================================================
    // MODULE 3: STATICS (STABILITY & CoM)
    // =========================================================================

    /**
     * @brief Analyzes vertical oscillation of the Center of Mass.
     * * Args: Coordinates for L/R Shoulders and L/R Hips.
     * Returns: (CoM_X, CoM_Y, Status String, Color Tuple)
     */
    std::tuple<int, int, std::string, std::tuple<int, int, int>> track_stability(
        std::vector<double> l_sh, std::vector<double> r_sh,
        std::vector<double> l_hip, std::vector<double> r_hip
    ) {
        // 1. Calculate Geometric Centroid of the Torso
        double com_x = ((l_sh[0] + r_sh[0]) / 2.0 + (l_hip[0] + r_hip[0]) / 2.0) / 2.0;
        double com_y = ((l_sh[1] + r_sh[1]) / 2.0 + (l_hip[1] + r_hip[1]) / 2.0) / 2.0;

        // 2. Update History Buffer
        if (com_history.size() >= STABILITY_BUFFER_SIZE) {
            com_history.pop_back();
        }
        com_history.push_front(com_y);

        // 3. Evaluate Stability
        std::string status = "STABLE";
        std::tuple<int, int, int> color = std::make_tuple(0, 255, 0); // Green

        if (com_history.size() > 10) {
            // Find Min/Max in deque (Linear scan is fast for size 30)
            double min_y = com_history[0];
            double max_y = com_history[0];
            for (double val : com_history) {
                if (val < min_y) min_y = val;
                if (val > max_y) max_y = val;
            }

            double oscillation = max_y - min_y;
            if (oscillation > VERTICAL_LIMIT) {
                status = "UNSTABLE";
                color = std::make_tuple(0, 0, 255); // Red
            }
        }

        return std::make_tuple((int)com_x, (int)com_y, status, color);
    }

    // =========================================================================
    // MODULE 4: MORPHOLOGY (STANCE CLASSIFIER)
    // =========================================================================

    /**
     * @brief Heuristic Decision Tree for Stance Classification.
     * * Args: Normalized coordinates [x,y] for hips, knees, ankles, feet.
     * Returns: (Stance Name, Color, Angle Left, Angle Right)
     */
    std::tuple<std::string, std::tuple<int, int, int>, int, int> track_stance(
        std::vector<double> l_hip, std::vector<double> r_hip,
        std::vector<double> l_knee, std::vector<double> r_knee,
        std::vector<double> l_ank, std::vector<double> r_ank,
        std::vector<double> l_foot, std::vector<double> r_foot
    ) {
        // 1. Feature Extraction: Joint Angles
        double angle_l = calculate_angle(l_hip, l_knee, l_ank);
        double angle_r = calculate_angle(r_hip, r_knee, r_ank);

        // 2. Feature Extraction: Spatial Dimensions
        double base_width_x = std::abs(l_ank[0] - r_ank[0]);
        double base_length_y = std::abs(l_ank[1] - r_ank[1]);

        // 3. Decision Tree
        std::string status = "NEUTRAL";
        std::tuple<int, int, int> color = std::make_tuple(200, 200, 200); // Gray

        // A. ZENKUTSU DACHI (Front Stance)
        // Deep stance, asymmetric loading
        if (base_length_y > 0.2 && ((angle_l < 110 && angle_r > 150) || (angle_r < 110 && angle_l > 150))) {
            status = "ZENKUTSU DACHI";
            color = std::make_tuple(0, 255, 0); // Green
        }
        // B. KOKUTSU DACHI (Back Stance)
        // Weight back, L-shape feet approx.
        else if (base_length_y > 0.15 && 
                ((angle_r < 110 && l_ank[1] < r_ank[1]) || (angle_l < 110 && r_ank[1] < l_ank[1]))) {
            status = "KOKUTSU DACHI";
            color = std::make_tuple(255, 0, 255); // Magenta
        }
        // C. KIBA / SHIKO DACHI
        // Wide base, bilateral flexion
        else if (angle_l < 140 && angle_r < 140 && base_width_x > 0.2) {
            double toe_outward_l = std::abs(l_foot[0] - l_ank[0]);
            if (toe_outward_l > 0.05) {
                status = "SHIKO DACHI";
            } else {
                status = "KIBA DACHI";
            }
            color = std::make_tuple(255, 165, 0); // Orange
        }
        // D. NEKO ASHI DACHI (Cat Stance)
        // Short base, one leg heavily loaded
        else if (base_length_y < 0.15 && (angle_l < 100 || angle_r < 100)) {
            status = "NEKO ASHI DACHI";
            color = std::make_tuple(0, 255, 255); // Cyan
        }
        // E. SANCHIN DACHI
        // Tight base, internal rotation
        else if (base_length_y < 0.15 && angle_l < 155 && angle_r < 155) {
            status = "SANCHIN DACHI";
            color = std::make_tuple(0, 128, 255); // Light Blue
        }
        // F. FUDO DACHI
        // Rooted stance
        else if (base_length_y > 0.15 && angle_l < 140 && angle_r < 140) {
            status = "FUDO DACHI";
            color = std::make_tuple(0, 100, 0); // Dark Green
        }
        // G. KAKE DACHI
        // Crossed legs logic
        else if ((l_ank[0] > r_ank[0] && l_hip[0] < r_hip[0]) || 
                 (r_ank[0] < l_ank[0] && r_hip[0] > l_hip[0])) {
            status = "KAKE DACHI";
            color = std::make_tuple(128, 0, 128); // Purple
        }

        return std::make_tuple(status, color, (int)angle_l, (int)angle_r);
    }
    
    // Utility to reset session
    void reset() {
        prev_speed = 0.0;
        prev_time = 0.0;
        prev_wrist = {0.0, 0.0, 0.0};
        com_history.clear();
    }
};

// --- PYTHON BINDINGS ---
PYBIND11_MODULE(karate_core, m) {
    m.doc() = "WKF Karate Physics Engine - C++ Optimized (Phase 6)";

    py::class_<KarateEngine>(m, "KarateEngine")
        .def(py::init<>())
        .def("calculate_angle", &KarateEngine::calculate_angle, "Vector geometry angle calculation")
        .def("track_speed_kime", &KarateEngine::track_speed_kime, "Kinematics & Kime detection")
        .def("track_stability", &KarateEngine::track_stability, "Center of Mass analysis")
        .def("track_stance", &KarateEngine::track_stance, "Heuristic Stance Classifier")
        .def("reset", &KarateEngine::reset, "Reset internal state");
}