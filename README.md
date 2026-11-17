# Digital Twin Enabled Intelligent Decision Framework Using Edge Artificial Intelligence

A full-fledged, web-based prototype implementing a 3-layer architecture (Physical Layer → Edge Layer → Cloud Layer → End Users) for real-time monitoring, anomaly detection, predictive maintenance, and intelligent decision-making powered by Edge AI.

---

## 🌟 Key Features

1. **Physical Layer Simulation**: Realistic multi-sensor industrial equipment simulator (Temperature, Vibration, Humidity, Energy, RPM, Pressure) with 6 operating scenarios and manual slider overrides.
2. **Edge AI Processing**:
   - **Isolation Forest** (Unsupervised ML) for real-time anomaly detection.
   - **Random Forest Classifier** for failure probability & machine health prediction.
   - On-device lightweight inference engine running locally with <15ms latency.
3. **Digital Twin Module**: Real-time synced virtual machine replica with live animated SVG visualization, parameter tracking, and health status indicators.
4. **Intelligent Decision Engine**: Dynamic hybrid rule engine + AI inference system generating actionable, role-based recommendations.
5. **Scenario Demonstration Mode**: Live 1-click simulation of normal operation, overheating, excessive vibration, energy spikes, and critical machine failures.
6. **Edge vs Cloud Benchmarking**: Real-time performance comparison highlighting bandwidth savings (78%) and latency reduction (15-20x speedup).
7. **Role-Based Access Control**: Role-tailored dashboards for Operators, Maintenance Teams, Managers (Decision Makers), and Administrators.

---

## 🛠️ Technology Stack

- **Frontend / Dashboard**: Streamlit, Custom CSS (Industrial Dark Theme), HTML/SVG
- **Data Visualization**: Plotly
- **Machine Learning / Edge AI**: Scikit-learn (Isolation Forest, Random Forest), Joblib
- **Data Engine**: Pandas, NumPy
- **Persistence**: SQLite 3

---

## 🚀 Quick Start Guide (Windows)

### 1. Prerequisites
Ensure Python 3.9+ is installed on your computer.

### 2. Setup & Installation
Open PowerShell in the project directory (`D:\digital-twin-edge-ai`) and execute:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Model Training & Database Initialization
Train the machine learning models and create the SQLite database:

```powershell
python train_models.py
```

### 4. Launch Application
Run the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🔑 Demo Credentials & Roles

| Username | Password | Role | Visible Dashboard Modules |
|---|---|---|---|
| **admin** | `admin123` | Administrator | All 11 pages (Full Access) |
| **operator** | `op123` | Industry Operator | Overview, Live Monitoring, Digital Twin, Alerts, Architecture |
| **maintenance** | `maint123` | Maintenance Team | Overview, Digital Twin, Edge AI, Predictive Maintenance, Alerts, Analytics |
| **manager** | `mgr123` | Decision Maker | Overview, Decision Engine, Historical Analytics, Architecture, Edge vs Cloud |

---

## 🎤 5-Minute Class Presentation Walkthrough

Follow this step-by-step procedure during your live presentation to impress your instructor/evaluator:

1. **Login & Role Overview (1 min)**:
   - Login as `admin` (`admin123`).
   - Highlight the **Overview Page**: Show the live KPI cards (System Health, Latency, Failure Probability) and explain the 3-Layer Edge AI architecture.

2. **Physical Layer & Sensor Data Stream (1 min)**:
   - Go to **Live Monitoring**.
   - Show the real-time sensor graphs updating continuously.
   - Use the **Manual Override Sliders** to increase temperature or vibration, showing how the system immediately captures changes.

3. **Digital Twin & Virtual Sync (1 min)**:
   - Go to **Digital Twin**.
   - Point out the animated industrial motor SVG diagram. Show how the spindle rotation speed dynamically tracks the RPM, and color codes change based on machine health.

4. **Edge AI & Predictive Maintenance (1 min)**:
   - Go to **Edge AI**. Explain that Isolation Forest runs locally with <15ms latency without sending raw data to the cloud.
   - Go to **Predictive Maintenance**. Show the failure probability gauge and the automated maintenance risk rating.

5. **Scenario Simulation & Intelligent Decision Engine (1 min)**:
   - Go to **Scenario Simulation**. Click **"Critical Machine Failure"** or **"Overheating"**.
   - Switch to **Decision Engine**: Demonstrate how the hybrid rule engine combines sensor values + AI outputs to generate an emergency recommendation (e.g., *"🛑 STOP MACHINE — Emergency Maintenance Required"*).
   - Show **Edge vs Cloud** to prove why Edge AI saves 78% bandwidth and avoids cloud latency.

---

## ❓ Evaluator Q&A Preparation

- **Q1: Why do we need Edge AI instead of processing everything in the cloud?**
  - *Answer*: Cloud processing introduces network latency (100-300ms) and requires continuous high bandwidth. Edge AI executes predictions locally in <15ms, allowing instant emergency actions (e.g. stopping a failing motor) even if cloud connectivity is lost.

- **Q2: How does the Digital Twin stay synchronized?**
  - *Answer*: The Digital Twin maintains a virtual state machine updated on every data frame via an MQTT/HTTP simulation loop. It mirrors physical attributes (RPM, temperature, vibration) and computes real-time health metrics.

- **Q3: What ML algorithms are used for Edge AI?**
  - *Answer*: We use **Isolation Forest** (unsupervised) to detect novel sensor anomalies without prior labels, and **Random Forest Classifier** (supervised) to estimate machine failure probability based on historical feature patterns.

- **Q4: How does the Decision Engine work?**
  - *Answer*: It combines deterministic threshold rules with probabilistic ML model outputs. When thresholds or failure probabilities are breached, it generates severity-coded, natural-language recommendations for operators and maintenance teams.
