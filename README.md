# MSC_CS_THESIS
# Federated Learning for Cross-Domain Generalization in 3D Cardiac MRI

## 📖 Overview
This repository contains the code, configuration files, and experimental scripts used for my Master's thesis: **"Federated Learning for Cross-Domain Generalization in 3D Cardiac MRI"** as part of the MSc in Computer Science (Data Science strand) at **Trinity College Dublin**. The project investigates the performance of federated learning (FL) algorithms for cross-domain generalization in multi-centre cardiac MRI segmentation. We evaluate multiple FL methods — **FedAvg**, **FedBN**, and **FedDANN** — against a centralized baseline using **3D ResNet-based classification models**. Experiments are conducted under a **Leave-Centre-Out (LCO)** validation strategy, simulating real-world domain shifts across MRI vendors, scanners, and acquisition protocols. This work is an extension of https://github.com/Linardos/federated-HCM-diagnosis

## 🎯 Research Objectives
- **Assess** the impact of domain shift on federated learning in medical imaging.  
- **Compare** FL algorithms in terms of generalization performance on unseen centers.  
- **Investigate** the trade-off between **global model performance** and **personalized/local adaptation**.  
- **Benchmark** computation and communication costs for each method.  

## 🧩 Methods
We implement and compare the following training strategies:  
- **Centralized Training** — All data pooled in a single location.  
- **FedAvg** — Federated averaging of client model weights.  
- **FedBN** — FedAvg variant with local BatchNorm statistics.  
- **FedDANN** — Domain-adversarial training with a gradient reversal layer to align feature distributions.  

All methods are trained using a **3D ResNet classification model** with a **two-class output** (HCM and NOR) using **cross-entropy loss**.

## 📊 Datasets
This study uses multi-centre, multi-vendor cardiac MRI datasets:  
- **ACDC**  
- **SantPau** 
- **Sagrada Familia**  
- **Vall d'Hebron**  

**Data pre-processing includes:**  
- N4 bias field correction  
- Histogram standardization  
- Cropping/padding to uniform dimensions  
- Intensity rescaling  
- End-diastole / End-systole (ED/ES) frame extraction  

⚠ **Note:** Due to licensing restrictions, the datasets are **not included** in this repository.


To carry out preprocessing and then run the experiments
```bash

python do_preprocessing_ACDC.py
python do_preprocessing.py
python run_core_experiments.py
