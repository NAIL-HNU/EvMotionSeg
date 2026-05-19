## Real-time Motion Segmentation with Event-based Normal Flow

### **Related Publications**

[1] **[Real-time Motion Segmentation with Event-based Normal Flow](https://arxiv.org/pdf/2602.20790)**, *Sheng Zhong\*, Zhongyang Ren\*, Xiya Zhu, Dehao Yuan, Cornelia Fermuller, Yi Zhou*, IEEE International Conference on Robotics and Automation (ICRA), 2026.

[2] **[Event-based Motion Segmentation with Spatio-Temporal Graph Cuts](https://arxiv.org/pdf/2012.08730.pdf)**,
*Yi Zhou, Guillermo Gallego, Xiuyuan Lu, Siqi Liu and Shaojie Shen*,
IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 34(8):4868-4880, 2021.

[3] **[Event-based Motion Segmentation by Cascaded Two-Level Multi-Model Fitting](https://arxiv.org/pdf/2111.03483)**,
*Yi Zhou, Guillermo Gallego, Xiuyuan Lu, Siqi Liu and Shaojie Shen*, IROS 2021.

### 1. Installation

We have tested MotionSeg on machines with the following configurations

- Ubuntu 20.04 LTS + ROS Noetic + OpenCV 4.2.0 + Eigen 3.3.9

The environment setup is identical to [EMSGC](https://github.com/HKUST-Aerial-Robotics/EMSGC). Please refer to the EMSGC repository for detailed dependency installation. Specific dependency items can be found in the `dependencies.yaml`.

### 2. Usage

#### 2.1 Data Preparation

The system expects three input files under `data_file_path`:

| File | Description |
|------|-------------|
| `events.txt` | Event data: timestamp, pixel location, and polarity |
| `undistorted_normalized_xy.txt` | Undistorted coordinates of the normal flow |
| `flow_xy.txt` | Normalized normal flow results |

All normal flow results are computed using [VecKM](https://github.com/dhyuan99/VecKM). We provide the [data](https://drive.google.com/drive/folders/19YwaS8qri0mjAuSba4hX10JlrnwUnYYp?usp=drive_link) used in our paper's experiments for reproducibility.

#### 2.2 Configuration

Modify the parameters in the launch file or set them via ROS parameter server. Key parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data_file_path` | Path to the input data directory 
| `interval` | Time interval for normal flow (s)
| `width` | Image width 
| `height` | Image height 
| `data_term` | Data term weight in graph cut 
| `smooth_term` | Smoothness term weight 
| `label_term` | Label cost term weight 
| `downsample_rate` | Downsampling rate for events
| `fx` / `fy` | Focal length scaling factors 
| `GraphCutIteration` | Number of graph cut iterations
| `MotionSegIteration` | Number of motion segmentation iterations

#### 2.3 Run

Launch the motion segmentation node:

```bash
cd ~/catkin_ws
source devel/setup.bash
roslaunch motion_segmentation EED.launch
```

The segmentation results will be saved to `[data_file_path]/results/`.

### 4. Contact us

For questions or inquiries, please feel free to contact us at zhongyangren@hnu.edu.cn or bell@hnu.edu.cn.

### 5. Acknowledgments

We thank the authors of the following repositories for publicly releasing their work:

- [EMSGC](https://github.com/HKUST-Aerial-Robotics/EMSGC)
- [VecKM](https://github.com/dhyuan99/VecKM)
