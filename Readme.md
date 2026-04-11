# 📘 Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network

This repository contains the official implementation of the paper:

> **Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatiotemporal Forecasting**

---

## 📂 Project Overview

The codebase is organized into two main components:

1. **Dynamic Causal Graph Generation Module**  
2. **Multi-step Prediction Module**

We provide demo implementations for both **Transportation** and **fMRI** datasets.

---

## 📁 Project Structure

```
.
├── graph_generation/ (FMRI or Transportation)
│   ├── configurations/
│   ├── main.py
│   └── ...
├── prediction-code/ (FMRI or Transportation)
│   ├── main.py
│   └── ...
├── README.md
```

---

## ⚙️ Installation

### 🐍 Environment
- Python >= 3.8

---

### 📦 Dependencies

#### Graph Generation Module
- tensorflow  
- scipy  
- numpy  
- matplotlib  
- pandas  
- seaborn  
- scikit-learn  
- argparse  
- configparser  

#### Prediction Module
- torch  
- numpy  
- matplotlib  
- pandas  
- tensorflow  
- scikit-learn  
- argparse  
- csv  

---

### ✅ Verified Working Versions

```
tensorflow==2.5.0
tensorflow-estimator==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
numpy==1.19.5
scipy==1.9.3
matplotlib==3.6.2
pandas==1.4.4
seaborn==0.12.1
scikit-learn==1.1.3
torch
```

---

## 🚀 Usage
You can also Skip the step 1 directory since we put our generated graph into the directory in this repository.
### Step 1: Dynamic Causal Graph Generation

Run:
```
python ./graph_generation/Transportation/main.py
```
or:
```
python ./graph_generation/FMRI/main.py
```

After training, you will obtain parameter files such as Tdrive_normalization_parameter.npz. Run dynamic_graph_trans_.py to generate the dynamic transition matrix, such as dynamic_Tdrive_adj.npy file. For example
dynamic_Tdrive_adj.npy


---

### Step 2: Multi-step Prediction

Move the generated file to:
```
./prediction-code/.../generated_adj/
```

Run:
```
python ./prediction-code/Transportation/main.py
```
or:
```
python ./prediction-code/FMRI/main.py
```

---


## 📖 Citation

```
@ARTICLE{10589693,
  author={Liang, Guojun and Tiwari, Prayag and Nowaczyk, Sławomir and Byttner, Stefan and Alonso-Fernandez, Fernando},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Dynamic Causal Explanation Based Diffusion-Variational Graph Neural Network for Spatiotemporal Forecasting}, 
  year={2025},
  volume={36},
  number={5},
  pages={9524-9537},
  doi={10.1109/TNNLS.2024.3415149}
}
```

---

## 📬 Contact
If you have any questions, please feel free to email me,
gorgen@163.com
