# 🤖 Task 8: Robot Simulation with Gazebo & URDF

This task introduces the simulation tools used in modern robotics workflows. You'll learn to model a differential drive robot using **URDF**, spawn it in **Gazebo**, visualize it in **RViz**, and control it using keyboard teleoperation.

---

## 📌 Objectives

1. 🧠 **Understand Gazebo** (simulation) and **RViz** (visualization)
2. 🛠️ **Write a URDF file** describing a differential drive robot
3. 🌐 **Spawn the robot in Gazebo**
4. 🎮 **Add differential drive plugin** and control the robot using `teleop_twist_keyboard`

---

## 📚 Resources

### 📺 Learning Materials

- [Gazebo Overview – Official Docs](https://gazebosim.org/)
- [RViz Visualization Tool – ROS Docs](https://wiki.ros.org/rviz)
- [URDF Guide – ROS2 Docs](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)

---

## 🗂️ Files Included

| File/Folder                 | Description                                      |
|----------------------------|--------------------------------------------------|
| `urdf/diff_drive_robot.urdf.xacro` | Robot model in URDF/XACRO format               |
| `launch/spawn_robot.launch.py`    | Launch file to spawn robot in Gazebo           |
| `worlds/empty_world.world`        | Gazebo world (optional custom world)           |
| `config/rviz_config.rviz`         | RViz visualization settings                    |
| `scripts/teleop_control.sh`       | Script to run keyboard teleop node             |
| `README.md`                       | This file                                      |

---

## ⚙️ Setup Instructions

Make sure you have:
- ROS2 Humble installed ✅
- Gazebo and RViz installed ✅

Install `teleop_twist_keyboard` if not already:
```bash
sudo apt install ros-humble-teleop-twist-keyboard
