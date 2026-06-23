# 🛡️ Agentic Framework for Intelligent Fraud Detection and Operational Risk Monitoring

This repository contains the implementation of the AI-driven framework proposed in the paper **Agentic_Fraud_Detection_Paper.pdf**. The system is designed to identify suspicious financial transactions, provide explainable decisions, dispatch real-time alerts, and execute autonomous model maintenance.

---

## 📖 Overview
The rapid growth of digital financial services has created a strong need for real-time, explainable, and adaptive fraud detection systems[span_0](start_span)[span_0](end_span). This project addresses these challenges by combining an offline machine learning training pipeline with a live FastAPI inference service and interactive monitoring dashboards[span_1](start_span)[span_1](end_span).

Unlike static, rule-based engines, this architecture evaluates transactions using multiple aggregated risk signals and leverages a local Large Language Model (LLM) to deliver clear, auditable reasoning for its operational actions[span_2](start_span)[span_2](end_span).

---

## 🚀 Key Features

* **Multi-Signal Risk Aggregation:** Computes an intelligent fraud score by combining transaction amount severity, SVM confidence gaps, Fuzzy C-Means (FCM) cluster membership, temporal multipliers, and SHAP cosine reliability[span_3](start_span)[span_3](end_span).
* **Explainable AI (XAI) & Local Reasoning:** Uses SHAP values to identify dominant suspicious features and routes them to a locally deployed Ollama model (`qwen2.5:1.5b`) to generate natural-language explanations without relying on external cloud APIs[span_4](start_span)[span_4](end_span).
* **Hybrid RAG Decision Support:** Implements a FAISS vector index to perform similarity searches over historical transactions and explanation vectors, guiding operators when a transaction falls into an uncertain risk range[span_5](start_span)[span_5](end_span).
* **Automated Alert Dispatch:** Integrates with an n8n webhook layer to instantly route critical `BLOCK_CARD` notifications to external channels like Telegram[span_6](start_span)[span_6](end_span).
* **Six-Zone Monitoring Dashboard:** Dual interface deployment featuring a responsive HTML/JavaScript dashboard for operational analysts and a Python-based Streamlit dashboard for rapid testing and demonstrations[span_7](start_span)[span_7](end_span).
* **Autonomous Retraining Loop:** Monitors system state and triggers an automatic offline model refresh when data drift or injected threat patterns are detected[span_8](start_span)[span_8](end_span).

---

## 🏗️ System Architecture

The project is highly modular and organized into four decoupled layers[span_9](start_span)[span_9](end_span):

### 1. User Interface Layer
* **HTML Dashboard:** Provides real-time visibility into live scoring histograms, cluster health, action dispatch counts, and SHAP reliability metrics[span_10](start_span)[span_10](end_span).
* **Streamlit Dashboard:** Used for experimentation, tracking outcome tables, and rendering cluster distribution pie charts[span_11](start_span)[span_11](end_span).

### 2. Backend Orchestration Layer (FastAPI)
* Evaluates input payloads through the `/predict` endpoint[span_12](start_span)[span_12](end_span).
* Standardizes features, scores risk metrics dynamically, and enforces deterministic threshold-based actions (Approve, OTP, Human Review, Block)[span_13](start_span)[span_13](end_span).

### 3. Model Artifact & Memory Layer
* Stores trained preprocessing scalers, SVM hyperplanes, FCM centroids, and the dense FAISS similarity index[span_14](start_span)[span_14](end_span).

### 4. Local AI Engine (Ollama)
* Independently translates numerical risk signals into professional sentences to describe why security actions were taken[span_15](start_span)[span_15](end_span).

---

## 📂 Dataset & Model Artifacts

### Data Requirements
To run the offline training pipeline (`offline_training.py`), you must provide a historical transaction dataset named **`train.csv`** in the root project directory[span_16](start_span)[span_16](end_span). 

The dataset must contain the following fields[span_17](start_span)[span_17](end_span):
* `amt`: Transaction amount[span_18](start_span)[span_18](end_span)
* `lat` & `long`: Geographic coordinates of the transaction[span_19](start_span)[span_19](end_span)
* `city_pop`: Target city population[span_20](start_span)[span_20](end_span)
* `unix_time`: Epoch timestamp[span_21](start_span)[span_21](end_span)
* `is_fraud`: Binary target label (`0` for legitimate, `1` for fraudulent)[span_22](start_span)[span_22](end_span)

### Generated Artifacts
Upon successful training, the pipeline exports the following binary objects to the `models/` directory[span_23](start_span)[span_23](end_span):
* `standard_scaler.pkl` & `amount_scaler.pkl`[span_24](start_span)[span_24](end_span)
* `svm_model.pkl` & `fcm_centroids.pkl`[span_25](start_span)[span_25](end_span)
* `shap_explainer.pkl` & `cluster_mean_shap.pkl`[span_26](start_span)[span_26](end_span)
* `rag_index.faiss` & `rag_labels.pkl`[span_27](start_span)[span_27](end_span)

---

## 📊 Feature Comparison Matrix
As detailed in **Agentic_Fraud_Detection_Paper.pdf**, this framework bridges the gap between pure black-box classification and rigid rule engines[span_28](start_span)[span_28](end_span):

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
