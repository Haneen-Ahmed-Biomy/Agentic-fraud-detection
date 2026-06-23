import os
import numpy as np
import joblib
import faiss
import threading
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import cosine, euclidean

# ==========================================
# 1. API Configuration & Setup
# ==========================================
app = FastAPI(title="Agentic Fraud Detection System", version="1.0.0")

# Enable CORS for the frontend Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. Memory & State Management
# ==========================================
MODELS_PATH = 'models/'

def load_models():
    """Loads all machine learning models and FAISS indexes into memory."""
    try:
        return {
            "scaler": joblib.load(os.path.join(MODELS_PATH, 'standard_scaler.pkl')),
            "amt_scaler": joblib.load(os.path.join(MODELS_PATH, 'amount_scaler.pkl')),
            "svm": joblib.load(os.path.join(MODELS_PATH, 'svm_model.pkl')),
            "centroids": joblib.load(os.path.join(MODELS_PATH, 'fcm_centroids.pkl')),
            "shap_explainer": joblib.load(os.path.join(MODELS_PATH, 'shap_explainer.pkl')),
            "cluster_shap": joblib.load(os.path.join(MODELS_PATH, 'cluster_mean_shap.pkl')),
            "faiss_index": faiss.read_index(os.path.join(MODELS_PATH, 'rag_index.faiss')),
            "rag_labels": joblib.load(os.path.join(MODELS_PATH, 'rag_labels.pkl'))
        }
    except Exception as e:
        print(f"⚠️ [SYSTEM] Error loading models: {e}")
        return None

# Initialize models and drift monitoring state
models = load_models()
drift_monitor = {
    "gap_history": [],
    "is_retraining": False
}

class TransactionRequest(BaseModel):
    features: list  # Expected format: [Amount, Latitude, Longitude, Population, Unix_Time]


# ==========================================
# 3. Multi-Agent System Architecture
# ==========================================

def generate_agentic_explanation(score, action, dominant_feature):
    """
    Agent 2 (Reasoning Agent): Uses a local LLM (Ollama) to generate a natural language explanation.
    """
    prompt = f"""
    You are an expert fraud detection AI agent working for a bank. 
    You just processed a transaction with the following details:
    - Fraud Risk Score: {score:.3f}
    - Action Taken: {action}
    - Key Suspicious Feature: {dominant_feature}
    
    Write exactly ONE professional and brief sentence explaining this decision. 
    Do not add introductory words, just the explanation.
    """
    
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False
    }
    
    print("⏳ [Agent 2] Asking Ollama to explain... Please wait...")
    try:
        response = requests.post(url, json=payload, timeout=20)
        explanation = response.json().get("response", "").strip()
        print("✅ [Agent 2] Ollama replied successfully!")
        return explanation
    except Exception as e:
        print(f"❌ [Agent 2] Ollama connection error: {e}")
        return f"System triggered {action} due to suspicious {dominant_feature} patterns."


def trigger_n8n_alert():
    """
    Agent 3 (Notification Agent): Dispatches a webhook to n8n for Telegram alerts.
    """
    url = "http://localhost:5678/webhook/8db0e22d-ec06-4ef9-a3d9-4eae67f6d21a"
    try:
        requests.post(url, timeout=3)
        print("✅ [Agent 3] Alert dispatched to Telegram via n8n!")
    except Exception as e:
        print(f"⚠️ [Agent 3] Error dispatching alert: {e}")


def trigger_autonomous_retraining():
    """
    Outer Loop: Handles autonomous model retraining when concept drift is detected.
    """
    print("🚨 [OUTER LOOP] Drift Detected! Initiating autonomous retraining...")
    os.system("python offline_training.py")
    
    global models
    models = load_models()
    
    drift_monitor["gap_history"] = []
    drift_monitor["is_retraining"] = False
    print("✅ [OUTER LOOP] Retraining completed. Models dynamically reloaded.")


# ==========================================
# 4. Core Prediction Endpoint (Inner Loop)
# ==========================================

@app.post("/predict")
def predict_fraud(request: TransactionRequest):
    if not models:
        raise HTTPException(status_code=500, detail="Models are currently unavailable.")

    # --- Step 1: Data Preprocessing ---
    raw_input = np.array(request.features).reshape(1, -1)
    scaled_input = models["scaler"].transform(raw_input)
    
    # --- Step 2: Signal Computation ---
    amt_scaled = models["amt_scaler"].transform(raw_input[:, 0].reshape(-1, 1))[0][0]
    severity_signal = 1.5 if amt_scaled > 0.5 else 1.0 
    
    probs = models["svm"].predict_proba(scaled_input)[0]
    pred_class = np.argmax(probs)
    sorted_probs = np.sort(probs)
    gap_svm = sorted_probs[-1] - sorted_probs[-2]
    
    dists = [euclidean(scaled_input[0], c) for c in models["centroids"]]
    mu_pred = 1 / (1 + dists[pred_class])
    
    arima_mult = 1.0 
    
    current_shap = models["shap_explainer"].shap_values(scaled_input)[0]
    mean_shap = models["cluster_shap"][pred_class]
    cosine_sim = 1 - cosine(current_shap, mean_shap)
    shap_reliability = max(0.3, float(cosine_sim)) 
    
    # --- Step 3: Final Intelligent Scoring ---
    final_score = severity_signal * gap_svm * mu_pred * arima_mult * shap_reliability
    
    # --- Step 4: ReAct Decision Engine ---
    feature_names = ['Amount', 'Latitude', 'Longitude', 'Population', 'Time']
    dominant_feature = feature_names[np.argmax(np.abs(current_shap))]
    
    final_action = "APPROVE"
    if final_score > 0.35:
        if shap_reliability > 0.90: 
            final_action = "BLOCK_CARD" 
        else: 
            final_action = "REQUEST_2FA_OTP" 
    elif 0.10 <= final_score <= 0.35: 
        # RAG Augmentation Logic
        query_vec = np.hstack([scaled_input[0], current_shap]).astype('float32').reshape(1, -1)
        _, I = models["faiss_index"].search(query_vec, 5)
        neighbor_labels = models["rag_labels"][I[0]]
        
        if np.mean(neighbor_labels) > 0.5: 
            final_action = "HUMAN_REVIEW" 
        else: 
            final_action = "APPROVE" 
    
    # --- Step 5: Drift Monitoring & Inject Override ---
    drift_monitor["gap_history"].append(gap_svm)
    current_status = "STABLE"
    
    if drift_monitor["is_retraining"]:
        current_status = "RETRAINING"
    elif request.features[0] == 88888.0:
        # Threat Injection Detected -> Force Retrain & Block
        current_status = "RETRAINING"
        drift_monitor["is_retraining"] = True
        final_action = "BLOCK_CARD" 
        threading.Thread(target=trigger_autonomous_retraining).start()
    
    # --- Step 6: Invoke Reasoning Agent (LLM) ---
    explanation = generate_agentic_explanation(final_score, final_action, dominant_feature)

    # --- Step 7: Invoke Notification Agent ---
    if final_action == "BLOCK_CARD":
        threading.Thread(target=trigger_n8n_alert).start()

    # --- Final Response ---
    return {
        "intelligent_score": float(final_score),
        "signals": {
            "S_severity": float(severity_signal),
            "Gap_SVM": float(gap_svm),
            "Mu_predicted": float(mu_pred),
            "ARIMA_trend": float(arima_mult),
            "SHAP_cosine": float(cosine_sim)
        },
        "assigned_cluster": int(pred_class),
        "action": final_action,
        "reasoning": explanation,
        "system_status": current_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)