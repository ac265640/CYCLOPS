# CYCLOPS ↔ BCSE428L *Autonomous Drones* syllabus

Each syllabus module maps to the part of the project that implements or demonstrates it. Use this table during the viva.

| Module | Syllabus topics | Where it lives in CYCLOPS | What to show |
|---|---|---|---|
| **1 – Introduction to Autonomous Drones** | types of drones, airframe, batteries, motors, ESC, propellers | `cyclops/config.py` (`AirframeParams`, `BatteryParams`), `cyclops/vehicle/quadrotor.py` (`QuadXMixer`, `Battery`, motor/ESC first-order lag, `T = k_f ω²`, `Q = k_m T`) | Quad-X motor numbering and spin directions; LiPo voltage sag under load (OSD in the live demo); `scripts/endurance_analysis.py` |
| **2 – Design Fundamentals** | flight controllers, RC transmitters, FPV systems, telemetry, timing gates | `cyclops/control/flight_controller.py` (PX4-style cascade), `cyclops/control/commander.py` (`RCInput` with Mode-2 stick layout), live-demo FPV OSD + telemetry panel | Press **m** in the demo and fly with the arrow keys like an RC transmitter; the FPV view has a goggle-style OSD (horizon, flight-path marker, battery, altitude, speed) |
| **3 – Drone Basics** | preflight checks, flight modes, roll / pitch / throttle / yaw, simulators, manual mode, GPS autopilot, intelligent flight modes | `Commander.preflight_checks()` (battery, IMU, level, GPS, EKF, T/W, camera), modes `MANUAL`, `ALTCTL`, `POSCTL` (GPS), `TAKEOFF`, `MISSION` (intelligent obstacle avoidance), `RTL`, `LAND`; low-battery failsafe → RTL | The commander log in the demo shows the preflight checklist, arming and mode changes; press **h** for RTL and **l** to land |
| **4 – Modelling & Control (MATLAB/Simulink)** | quadcopter physical characteristics, vehicle dynamics, Simulink modelling | `cyclops/vehicle/quadrotor.py`: full 6-DOF Newton–Euler model (the Python equivalent of a Simulink plant block); `cyclops/vehicle/sensors.py`: IMU/GPS/baro + Kalman filter | The equations are in the file header; `scripts/stability_analysis.py` linearises the same model |
| **5 – Stability and Control** | static and dynamic stability, longitudinal control, directional stability and control, roll stability and control | `scripts/stability_analysis.py` → `docs/figures/stability_*.png`, `results/stability.json` | Open-loop eigenvalues at 0 show the bare airframe is only neutrally stable, so it needs active control. Roll step (roll), pitch step (longitudinal), yaw step (directional), gust rejection. Bode margins and the gain at which the roll loop goes unstable |
| **6 – Applications** | search & rescue, infrastructure inspection, mapping & surveying, … | `cyclops/world/world.py` scenarios: `forest_sar`, `urban_inspection`, `dynamic`, `pillars` | Keys **1–4** in the demo switch scenarios |
| **7 – Expanding Drone Abilities** | add a camera and FPV, collect data with other sensors, altering speed, increasing flight time | Monocular camera + deep perception (`cyclops/vision`, `cyclops/perception`), sensor suite, `scripts/endurance_analysis.py` (best-endurance and best-range speed, battery-capacity trade-off) | `docs/figures/endurance.png` |

### Beyond the syllabus
- Deep-learning perception (CenterNet-style detector and U-Net monocular depth), trained from scratch on self-labelled synthetic data.
- EKF multi-target tracking with the reference paper's **Mode 1 (range+bearing) / Mode 2 (bearing-only)** localisation.
- FOV-aware sampling-based APF planner, plus a stopping-distance safety constraint.
- ROS 2 nodes for PX4 SITL + Gazebo (`ros2_ws/`).
