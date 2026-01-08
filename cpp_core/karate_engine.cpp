#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <numeric>

namespace py = pybind11;

// Helper struct for 3D points
struct Point3D {
    double x, y, z;
};

class KarateEngine {
private:
    // State variables (όπως το self.prev_wrist)
    Point3D prev_wrist;
    double prev_time;
    double prev_speed;
    
    // Configuration Constants
    const double MIN_SPEED_FOR_KIME = 1100.0;
    const double MIN_ACCEL_FOR_KIME = -9500.0;

public:
    KarateEngine() {
        prev_wrist = {0.0, 0.0, 0.0};
        prev_time = 0.0;
        prev_speed = 0.0;
    }

    // Fast Angle Calculation (Vector Geometry)
    double calculate_angle(std::vector<double> a, std::vector<double> b, std::vector<double> c) {
        // a, b, c are [x, y] coordinates
        double ang = atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]);
        double angle_deg = std::abs(ang * 180.0 / M_PI);
        if (angle_deg > 180.0) angle_deg = 360.0 - angle_deg;
        return angle_deg;
    }

    // Optimized Velocity & Kime Tracker
    // Returns a tuple: (current_speed, is_kime)
    std::pair<double, bool> track_speed_kime(double x, double y, double z, double timestamp) {
        double dt = timestamp - prev_time;
        double current_speed = 0.0;
        bool is_kime = false;

        if (dt > 0 && prev_time > 0) { // Ensure not first frame
            // Euclidean Distance 3D
            double dx = x - prev_wrist.x;
            double dy = y - prev_wrist.y;
            double dz = z - prev_wrist.z;
            double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

            double raw_speed = distance / dt;

            // Exponential Moving Average (EMA) Filter
            // current_speed = 0.7 * prev + 0.3 * raw
            current_speed = 0.7 * prev_speed + 0.3 * raw_speed;
            
            // Noise Gate
            if (current_speed < 50.0) current_speed = 0.0;

            // Acceleration (a = dv/dt)
            double acceleration = (current_speed - prev_speed) / dt;

            // Kime Logic
            if (prev_speed > MIN_SPEED_FOR_KIME && acceleration < MIN_ACCEL_FOR_KIME) {
                is_kime = true;
            }
        }

        // Update State
        prev_wrist = {x, y, z};
        prev_speed = current_speed;
        prev_time = timestamp;

        return {current_speed, is_kime};
    }
    
    // Reset state (useful for new sessions)
    void reset() {
        prev_speed = 0.0;
        prev_time = 0.0;
        prev_wrist = {0.0, 0.0, 0.0};
    }
};

// --- PYBIND11 BINDING CODE ---
// This creates the Python module named 'karate_core'
PYBIND11_MODULE(karate_core, m) {
    m.doc() = "Optimized C++ Karate Physics Engine for Phase 6";

    py::class_<KarateEngine>(m, "KarateEngine")
        .def(py::init<>())
        .def("calculate_angle", &KarateEngine::calculate_angle, "Calculates 2D joint angle")
        .def("track_speed_kime", &KarateEngine::track_speed_kime, "Calculates velocity and detects Kime")
        .def("reset", &KarateEngine::reset, "Resets internal state");
}