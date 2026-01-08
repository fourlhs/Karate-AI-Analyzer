import time
import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append('.')

def benchmark():
    print("="*60)
    print("🔬 PHASE 6.5: ENGINE INTEGRITY & PERFORMANCE TEST")
    print("="*60)

    # --- 1. LOAD ENGINES ---
    print("[1] Loading Engines...")
    
    try:
        import karate_core
        cpp_engine = karate_core.KarateEngine()
        print("   ✅ C++ Core: LOADED")
    except ImportError:
        print("   ❌ C++ Core: MISSING (Build failed?)")
        return

    # --- 2. ACCURACY TEST (Correctness) ---
    print("\n[2] Verifying Mathematical Accuracy (Angle Calculation)...")
    
    # Test Vectors (Hip, Knee, Ankle)
    a = [0.0, 1.0] # Hip
    b = [0.0, 0.0] # Knee (Pivot)
    c = [1.0, 0.0] # Ankle
    
    # Python Calculation (Numpy)
    a_np, b_np, c_np = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c_np[1]-b_np[1], c_np[0]-b_np[0]) - np.arctan2(a_np[1]-b_np[1], a_np[0]-b_np[0])
    py_angle = np.abs(rad * 180.0 / np.pi)
    
    # C++ Calculation
    cpp_angle = cpp_engine.calculate_angle(a, b, c)
    
    print(f"   Python Result: {py_angle:.6f}°")
    print(f"   C++ Result:    {cpp_angle:.6f}°")
    
    if abs(py_angle - cpp_angle) < 0.001:
        print("   ✅ STATUS: MATCH (Precision Excellent)")
    else:
        print("   ❌ STATUS: FAILURE (Math mismatch)")

    # --- 3. PERFORMANCE BENCHMARK (Speed) ---
    print("\n[3] Benchmarking Speed (1,000,000 Operations)...")
    
    ITERATIONS = 1_000_000
    
    # Python Loop
    start_py = time.time()
    for _ in range(ITERATIONS):
        # Raw math implementation in loop to simulate load
        _ = np.arctan2(c_np[1]-b_np[1], c_np[0]-b_np[0]) - np.arctan2(a_np[1]-b_np[1], a_np[0]-b_np[0])
    end_py = time.time()
    py_time = end_py - start_py
    
    # C++ Loop
    start_cpp = time.time()
    for _ in range(ITERATIONS):
        _ = cpp_engine.calculate_angle(a, b, c)
    end_cpp = time.time()
    cpp_time = end_cpp - start_cpp
    
    print(f"   🐍 Python Time: {py_time:.4f} seconds")
    print(f"   🚀 C++ Time:    {cpp_time:.4f} seconds")
    
    speedup = py_time / cpp_time
    print(f"\n   ⚡ SPEED FACTOR: C++ is {speedup:.1f}x FASTER")
    
    print("="*60)
    if speedup > 10:
        print("🏆 VERDICT: OPTIMIZATION SUCCESSFUL.")
    else:
        print("⚠️ VERDICT: OPTIMIZATION WEAK. REVIEW CODE.")
    print("="*60)

if __name__ == "__main__":
    benchmark()