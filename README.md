# OpenSensei Core: Open Source Biomechanics Framework 🥋

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-red)](https://isocpp.org/)

**Karate Sensei Core** is a high-performance computational framework designed to bring laboratory-grade biomechanical analysis to martial arts. 

By bridging **Computer Vision**, **Classical Physics**, and **Machine Learning**, it allows developers and researchers to extract objective metrics (velocity, acceleration, joint angles) from standard video input, removing the need for expensive motion capture suits.

## 🚀 Capabilities

### What is implemented (Current State)
* **Hybrid Engine:** Critical physics calculations are offloaded to a **C++ Core** (`karate_engine.cpp`) for real-time performance, wrapped in Python for ease of use.
* **Multi-Modal Sensing:** Wrappers for **MediaPipe** to extract high-fidelity skeleton keypoints.
* **Stereo Vision Logic:** Algorithms to fuse data from dual-camera setups to solve occlusion (hidden limbs) issues.
* **Kinematic Analysis:** Real-time computation of biomechanical features:
    * Limb Velocity & Acceleration ($v, a$)
    * Impact Force Estimation ($F = m \cdot a$)
    * Joint Angular Velocities
* **Algorithmic Scoring:** Implementation of **Dynamic Time Warping (DTW)** to compare student movements against a "Golden Reference" signal, independent of execution speed.

### Research Roadmap (Under Development)
* **Spatial-Temporal Graph Neural Networks (ST-GCN):** Transitioning from rule-based physics to Deep Learning models that treat the human skeleton as a dynamic graph.
* **Physics-Informed Neural Networks (PINNs):** Embedding physical laws into the loss function of the neural networks to ensure anatomically valid predictions.

## 💻 Usage Example

Karate Sensei Core is designed to be modular. Here is how you can use it to analyze a punch:

```python
from sensors.pose_detector import PoseDetector
from analysis.mechanics import Biomechanics
from cpp_core import karate_engine
```

### 1. Initialize Detectors
```python
detector = PoseDetector(model_complexity=2)
physics = Biomechanics()
```

### 2. Extract Skeleton from Video
```python
landmarks = detector.process_video("gyaky-zuki.mp4")
```

### 3. Calculate Kinematics (using C++ backend acceleration)
```python
velocity, acceleration = karate_engine.compute_kinematics(landmarks)
```

### 4. Compare with Golden Reference using DTW
```python
score = physics.compare_motion(landmarks, reference_data="golden_data/gyaku-zuki-champ.csv")
```

## 📂 Architecture
- cpp_core/: High-performance C++ bindings for heavy math operations.

- sensors/: Abstraction layers for camera input and MediaPipe integration.

- analysis/: Physics engine and DTW comparison logic.

- ui/: Visualization tools (Matplotlib 3D plotting).

- data/: Data pipelines for CSV recording and dataset management.

## 🤝 Contribution
This project is an initiative to democratize sports science. We welcome contributions, especially in:
- Optimizing the ST-GCN architecture.
- Expanding the biomechanics engine to other martial arts.

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
