#  Agentic Framework for Intelligent Fraud Detection and Operational Risk Monitoring

This repository contains the implementation of the AI-driven framework proposed in the paper **Agentic_Fraud_Detection_Paper.pdf**. The system is designed to identify suspicious financial transactions, provide explainable decisions, dispatch real-time alerts, and execute autonomous model maintenance.

---

##  Overview
The rapid growth of digital financial services has created a strong need for real-time, explainable, and adaptive fraud detection systems This project addresses these challenges by combining an offline machine learning training pipeline with a live FastAPI inference service and interactive monitoring dashboards

Unlike static, rule-based engines, this architecture evaluates transactions using multiple aggregated risk signals and leverages a local Large Language Model (LLM) to deliver clear, auditable reasoning for its operational actions

---

##  Key Features

* **Multi-Signal Risk Aggregation:** Computes an intelligent fraud score by combining transaction amount severity, SVM confidence gaps, Fuzzy C-Means (FCM) cluster membership, temporal multipliers, and SHAP cosine reliability
* **Explainable AI (XAI) & Local Reasoning:** Uses SHAP values to identify dominant suspicious features and routes them to a locally deployed Ollama model (`qwen2.5:1.5b`) to generate natural-language explanations without relying on external cloud APIs
* **Hybrid RAG Decision Support:** Implements a FAISS vector index to perform similarity searches over historical transactions and explanation vectors, guiding operators when a transaction falls into an uncertain risk range
* **Automated Alert Dispatch:** Integrates with an n8n webhook layer to instantly route critical `BLOCK_CARD` notifications to external channels like Telegram
* **Six-Zone Monitoring Dashboard:** Dual interface deployment featuring a responsive HTML/JavaScript dashboard for operational analysts and a Python-based Streamlit dashboard for rapid testing and demonstrations
* **Autonomous Retraining Loop:** Monitors system state and triggers an automatic offline model refresh when data drift or injected threat patterns are detected

---

##  System Architecture

The project is highly modular and organized into four decoupled layers

### 1. User Interface Layer
* **HTML Dashboard:** Provides real-time visibility into live scoring histograms, cluster health, action dispatch counts, and SHAP reliability metrics
* **Streamlit Dashboard:** Used for experimentation, tracking outcome tables, and rendering cluster distribution pie charts

### 2. Backend Orchestration Layer (FastAPI)
* Evaluates input payloads through the `/predict` endpoint
* Standardizes features, scores risk metrics dynamically, and enforces deterministic threshold-based actions (Approve, OTP, Human Review, Block)

### 3. Model Artifact & Memory Layer
* Stores trained preprocessing scalers, SVM hyperplanes, FCM centroids, and the dense FAISS similarity index

### 4. Local AI Engine (Ollama)
* Independently translates numerical risk signals into professional sentences to describe why security actions were taken

---

##  Dataset & Model Artifacts

### Data Requirements
To run the offline training pipeline (`offline_training.py`), you must provide a historical transaction dataset named **`train.csv`** in the root project directory

The dataset must contain the following fields
* `amt`: Transaction amount
* `lat` & `long`: Geographic coordinates of the transaction
* `city_pop`: Target city population
* `unix_time`: Epoch timestamp
* `is_fraud`: Binary target label (`0` for legitimate, `1` for fraudulent)
### Generated Artifacts
Upon successful training, the pipeline exports the following binary objects to the `models/` directory
* `standard_scaler.pkl` & `amount_scaler.pkl`
* `svm_model.pkl` & `fcm_centroids.pkl`
* `shap_explainer.pkl` & `cluster_mean_shap.pkl`
* `rag_index.faiss` & `rag_labels.pkl`

---

## Feature Comparison Matrix
As detailed in **Agentic_Fraud_Detection_Paper.pdf**, this framework bridges the gap between pure black-box classification and rigid rule engines
| Feature / Capability | Proposed Agentic System | Rule-Based Engines | Standalone SVM / RF | Deep Learning Black-Box |
| :--- | :---: | :---: | :---: | :---: |
| **Real-Time API Scoring** | **Yes** | Yes | Yes | Yes |
| **Multi-Signal Risk Aggregation** | **Yes** | No | No | No |
| **Fuzzy Cluster Membership** | **Yes** | No | No | No |
| **SHAP Explanation Reliability** | **Yes** | No | Optional | Optional |
| **FAISS/RAG Case Retrieval** | **Yes** | No | No | No |
| **Local LLM Decision Reasoner** | **Yes** | No | No | No |
| **Autonomous Retraining Trigger** | **Yes** | No | No | Optional |
| **Six-Zone Operational Dashboard** | **Yes** | No | No | No |
