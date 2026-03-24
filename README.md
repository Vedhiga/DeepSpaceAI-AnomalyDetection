# 🚀 AI-Based Real-Time Anomaly Detection for Deep Space Missions

**An Intelligent System for Autonomous Spacecraft Telemetry Monitoring & Decision Support**

![Project Banner](https://img.shields.io/badge/Deep%20Space-AI%20Anomaly%20Detection-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B)

---

### 📋 Project Overview

This project develops an **AI-powered real-time anomaly detection and autonomous decision support system** designed for deep space missions (e.g., Chandrayaan-3, Aditya-L1, Mars Perseverance Rover), where communication delays with Earth make real-time human intervention impossible.

The system detects anomalies in spacecraft telemetry data, predicts potential failures, and autonomously recommends corrective actions — all running locally on the spacecraft.

**Key Features:**
- Multivariate time-series anomaly detection using **Isolation Forest** (baseline) and **LSTM Autoencoder** (main model)
- Real-time autonomous decision-making logic
- Interactive **Mission Control Dashboard** built with Streamlit
- Fully reproducible pipeline using NASA CMAPSS FD001 dataset (proxy for spacecraft telemetry)
- Complete 21-day structured project implementation

---

### 🏗️ Detailed System Architecture

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
### Component Breakdown:

Data Layer
NASA CMAPSS FD001 Dataset (21 sensors + 3 operational settings)
Sliding window of 10 cycles for temporal modeling

Model Layer
Isolation Forest: Fast, unsupervised baseline for anomaly detection
LSTM Autoencoder: Deep learning model that learns normal behavior patterns and flags high reconstruction error as anomalies (NASA-inspired approach)

Decision Layer
Rule-based autonomous logic that converts anomaly scores into actionable decisions:
Normal → Continue mission
Anomaly → Reduce thrust / Isolate subsystem / Activate redundancy


Presentation Layer
Streamlit web application with:
Phase-wise navigation (21-day plan)
Live anomaly detection on uploaded telemetry
Interactive Plotly visualizations with anomaly highlighting
Downloadable results




### 🛠️ How the Project Works (Step-by-Step)

Telemetry Ingestion
Raw multivariate sensor data (21 sensors) is received from the spacecraft.
Preprocessing
Data normalization using MinMaxScaler
Creation of time-series sequences (window = 10)

Anomaly Detection
Phase 1 (Baseline): Isolation Forest flags statistical outliers
Phase 2 (Main): LSTM Autoencoder reconstructs input → high MSE = anomaly

Autonomous Decision Making
If anomaly detected → system triggers predefined corrective actions without waiting for Earth command.

Visualization & Monitoring
Real-time dashboard shows sensor trends with red markers on anomalous points
Engineers/ground team can monitor via web interface

### 🚀 How to Run Locally
# 1. Clone the repository
git clone https://github.com/VedhigaVB/DeepSpaceAI-AnomalyDetection.git
cd DeepSpaceAI-AnomalyDetection

# 2. Install dependencies
pip install streamlit tensorflow joblib plotly pandas numpy scikit-learn

# 3. Run the dashboard
streamlit run app.py
Open your browser at http://localhost:8501

### 📊 Results Achieved

Isolation Forest: F1-Score ≈ 0.85
LSTM Autoencoder: F1-Score ≈ 0.915 – 0.93
Anomalies successfully detected with clear visual highlighting
Autonomous decisions generated in real-time
Fully interactive web-based Mission Control Dashboard


### 🎯 21-Day Project Completion
All phases were successfully completed as per the structured plan:

Phase 1: Detect abnormal sensor behavior (Isolation Forest)
Phase 2: Model Development (LSTM Autoencoder)
Phase 3: Autonomous Decision Layer
Phase 4: Visualization & Deployment (Streamlit)


### 🔮 Future Extensions

Integration with real ISRO telemetry data
Reinforcement Learning for optimal action selection
Edge deployment on spacecraft hardware
Multi-fault detection (FD003/FD004)
Uncertainty quantification using Bayesian methods


### 📄 Documentation
Project Report : 
Colab Notebook : https://colab.research.google.com/drive/1MfT9pweH_1e_KQl_M_lnqICwtb-4FUTS?usp=sharing


Made with ❤️ for Deep Space Exploration
Author: Vedhiga.V.B
Institution: Avinashilingam University
Year: 2026
