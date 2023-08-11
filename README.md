# Crowd_Monitoring_Project
This is the code to monitor crowd in various areas.



<h1 align="center">Anomaly Detection in Crowded Scenes using Python</h1>

<p align="center">
  <img src="https://img.shields.io/badge/WelCome-8A2BE2" alt="License">
</p>

<p align="center">
  <strong>Python-based Anomaly Detection System for Crowded Environments</strong>
</p>

---

## Overview

This repository hosts a sophisticated Python-based program designed for real-time anomaly detection in crowded scenes. By harnessing the capabilities of advanced libraries such as NumPy, pandas, and OpenCV (cv2), the system offers a comprehensive solution for detecting, tracking, and analyzing anomalies in densely populated areas.

## Key Features

- **Human Pose Detection:** The system employs YOLOv8 pose model to accurately identify human poses, forming the foundation for anomaly analysis.
  
- **Crowd Density Estimation:** Utilizing optical flow analysis, the program determines the crowd density in a specified region of interest.

- **Anomaly Identification:** Through intelligent algorithms, the system identifies and highlights potential anomalies based on pose deviations and unusual object movements.

## Prerequisites

Before using the Anomaly Detection System, ensure the following prerequisites are met:

- Python (>=3.6)
- NumPy
- OpenCV (cv2)
- YOLOv8 pose model (download and setup instructions provided)

## Installation

1. Clone the repository: `git clone https://github.com/tosifAN/Crowd_Monitoring_Project.git`
2. Install required Python libraries: `pip install numpy opencv-python`
3. Obtain the YOLOv8 pose model (follow detailed instructions in the repository).

## Usage

1. Run the `working.py` script to initiate the anomaly detection process.
2. Configure input parameters, including file paths and thresholds, to tailor the analysis.
3. Observe the system process the input data and generate comprehensive anomaly reports.

## YOLOv8 Pose Model Info
Download link and Information is available in Offical Github link of ultralytics
`https://github.com/ultralytics/ultralytics/tree/main`


## REFERENCES

1.	Bouwmans, Thierry, et al. "Robust optical flow estimation and its applications: a survey." Image and Vision Computing 31.3 (2013): 263-288.
2. Shi, Jianbo, and Carlo Tomasi. "Good features to track." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 1994.
3.	Wu, Jianqiang, et al. "Robust crowd density estimation with a point-wise head-shoulder detector." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
4.	Ultralytics. "YOLOv8-Pose." GitHub Repository. Available online:(Link Provided In Model Info)
5.	Zivkovic, Zoran. "Improved adaptive Gaussian mixture model for background subtraction." Proceedings of the International Conference Pattern Recognition. 2004.
6.	Kohout, Josef, et al. "Crowd surveillance using multiple PTZ cameras on a mobile robot." Robotics and Autonomous Systems 107 (2018): 73-87.
7.	Zhang, Xiaoyu, et al. "A crowd monitoring system for metro stations." Journal of Visual Communication and Image Representation 64 (2019): 102619.

