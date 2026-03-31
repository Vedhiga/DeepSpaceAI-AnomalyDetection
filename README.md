# 🚀 AI-Based Real-Time Anomaly Detection for Deep Space Missions

**An Intelligent System for Autonomous Spacecraft Telemetry Monitoring & Decision Support**

![Project Banner](https://img.shields.io/badge/Deep%20Space-AI%20Anomaly%20Detection-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B)

---

## 📋 Project Overview

This project develops an **AI-powered real-time anomaly detection and autonomous decision support system** designed for deep space missions (e.g., Chandrayaan-3, Aditya-L1, Mars Perseverance Rover), where communication delays with Earth make real-time human intervention impossible.

The system detects anomalies in spacecraft telemetry data, predicts potential failures, and autonomously recommends corrective actions — all running locally on the spacecraft.

### **Key Features**

* Multivariate time-series anomaly detection using:
  * **Isolation Forest** (baseline)
  * **LSTM Autoencoder** (main model)
* Real-time autonomous decision-making logic
* Streamlit-based **Mission Control Dashboard**
* Reproducible pipeline using NASA CMAPSS FD001 dataset
* Complete 21-day structured project implementation

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Spacecraft Telemetry Data<br>(Multivariate Time-Series)] 
    --> B[Data Preprocessing<br>(Normalization + Sliding Window)]
    
    B --> C1[Baseline Model<br>Isolation Forest]
    B --> C2[Main Model<br>LSTM Autoencoder]
    
    C1 & C2 --> D[Anomaly Detection Engine]
    
    D --> E[Autonomous Decision Layer]
    E --> F[Alert & Action Engine<br>(Isolate / Reduce Thrust / Redundancy)]
    
    D & E --> G[Interactive Streamlit Dashboard]
    G --> H[Real-time Visualization + Export]
```

### **Component Breakdown**

#### **Data Layer**
* NASA CMAPSS FD001 Dataset (21 sensors + 3 operational settings)
* Sliding window of 10 cycles for temporal modeling

#### **Model Layer**
* **Isolation Forest:** Fast unsupervised anomaly detection
* **LSTM Autoencoder:** Learns normal sensor behavior and detects high reconstruction error as anomalies

#### **Decision Layer**

Rule-based autonomous logic:
* **Normal →** Continue mission
* **Anomaly →** Reduce thrust / Isolate subsystem / Activate redundancy

#### **Presentation Layer**

Streamlit dashboard featuring:
* Phase-wise 21-day plan navigation
* Live anomaly detection on uploaded telemetry
* Plotly visualizations with anomaly highlighting
* Downloadable results

---

## 🛠️ How the System Works

### **1. Telemetry Ingestion**
* Raw multivariate sensor data (21 sensors)

### **2. Preprocessing**
* Normalization using MinMaxScaler
* Time-series sequence creation (window size: 10)

### **3. Anomaly Detection**
* **Isolation Forest:** Statistical outlier detection
* **LSTM Autoencoder:** Reconstruction error-based detection

### **4. Autonomous Decision Making**
* If anomaly detected → automated corrective action
* No dependency on Earth-based commands

### **5. Visualization & Monitoring**
* Real-time dashboard
* Anomaly markers (red)
* Accessible via web interface

---

## 🚀 How to Run Locally

### **1. Clone the repository**

```bash
git clone https://github.com/VedhigaVB/DeepSpaceAI-AnomalyDetection.git
cd DeepSpaceAI-AnomalyDetection
```

### **2. Install dependencies**

```bash
pip install streamlit tensorflow joblib plotly pandas numpy scikit-learn
```

### **3. Run the dashboard**

```bash
streamlit run app.py
```

Open your browser at:
➡️ **[http://localhost:8501](http://localhost:8501)**

---

## 📊 Results Achieved

* **Isolation Forest:** F1-Score ≈ **0.85**
* **LSTM Autoencoder:** F1-Score ≈ **0.915 – 0.93**
* Accurate anomaly detection with visual highlighting
* Autonomous decisions generated in real-time
* Fully interactive Mission Control Dashboard

---

## 🎯 21-Day Project Completion

* **Phase 1:** Detect abnormal sensor behavior
* **Phase 2:** Model Development (LSTM Autoencoder)
* **Phase 3:** Autonomous Decision Layer
* **Phase 4:** Visualization & Deployment (Streamlit)

---

## 🔮 Future Extensions

* Integration with real ISRO telemetry data
* Reinforcement Learning for optimal action selection
* Deployment on spacecraft hardware
* Multi-fault detection (FD003/FD004)
* Bayesian uncertainty quantification

---

## 📄 Documentation

* **Project Report:**
  👉 [Open in Google Drive](https://drive.google.com/file/d/1ALoNOjEnbGYrL19BKshRZMZMwjRhCPle/view?usp=sharing)
* **Colab Notebook:**
  👉 [Open in Google Colab](https://colab.research.google.com/drive/1MfT9pweH_1e_KQl_M_lnqICwtb-4FUTS?usp=sharing)

---

## ✨ Author

**Made with ❤️ for Deep Space Exploration**

**Author:** Vedhiga V. B  
**Institution:** Avinashilingam University  
**Year:** 2026
