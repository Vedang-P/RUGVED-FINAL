# 🧊 Task 7: Creating a Point Cloud & Performing Object Detection

In this task, you’ll work with RGB and Depth images to reconstruct a 3D **point cloud** and then run **object detection** on that point cloud using the `Open3D` library.

This bridges the gap between traditional 2D computer vision and real-world spatial perception — a crucial skill in autonomous systems, AR/VR, and robotics.

---

## 🧩 Objectives

1. ✅ **Generate a 3D point cloud** from RGB + Depth image pairs
2. ✅ **Perform object detection** on the resulting point cloud

---

## 📚 Resources

- 📺 [YouTube Playlist – Open3D + PointClouds](https://www.youtube.com/watch?v=zF3MreN1w6c&list=PLkmvobsnE0GEZugH1Di2Cr_f32qYkv7aN)

Covers how to:
- Load RGB-D images into Open3D
- Create a point cloud
- Apply object detection or segmentation on the point cloud

---

## 💻 Requirements

Install dependencies:
```bash
pip install open3d numpy opencv-python matplotlib
