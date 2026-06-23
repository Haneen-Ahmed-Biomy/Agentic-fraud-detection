import os
import sys
import numpy as np
import pandas as pd
import joblib
import faiss
import skfuzzy as fuzz
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# ==========================================
# Configuration Variables
# ==========================================
MODELS_DIR = 'models'
DATA_FILE = 'train.csv'
SAMPLE_SIZE = 10000
N_CLUSTERS = 3
RANDOM_STATE = 42

# Features array (Transaction Amount 'amt' must be at index 0 for Severity scaling)
TARGET_FEATURES = ['amt', 'lat', 'long', 'city_pop', 'unix_time']
TARGET_LABEL = 'is_fraud'


def run_offline_training_pipeline():
    """
    Executes the end-to-end offline training pipeline for the Fraud Detection System.
    Includes data preprocessing, FCM clustering, SVM training, SHAP explainability, 
    and FAISS index generation for the RAG agent.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("🚀 [PIPELINE START] Initiating Offline Training Cycle with Real Data...")

    # ==========================================
    # 1. Data Preparation & Sampling
    # ==========================================
    print("⏳ [STEP 1] Loading and Preparing Data from CSV...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ [ERROR] '{DATA_FILE}' not found in the root directory.")
        print("Please ensure the Kaggle dataset is unzipped and placed next to this script.")
        sys.exit(1)

    # Random sampling to optimize local training time
    df_sampled = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    X = df_sampled[TARGET_FEATURES].values
    y = df_sampled[TARGET_LABEL].values

    # Global standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Specific MinMax scaling for Transaction Amount (Feature 0) to compute Severity (0 to 1)
    amount_scaler = MinMaxScaler()
    X_scaled[:, 0] = amount_scaler.fit_transform(X[:, 0].reshape(-1, 1)).flatten()

    # Persist scalers
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    joblib.dump(amount_scaler, os.path.join(MODELS_DIR, 'amount_scaler.pkl'))


    # ==========================================
    # 2. Fuzzy C-Means (FCM) Clustering
    # ==========================================
    print("⏳ [STEP 2] Training FCM Clustering Model...")
    data_for_fcm = X_scaled.T

    centroids, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_for_fcm, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000, init=None
    )

    cluster_labels = np.argmax(u, axis=0)
    joblib.dump(centroids, os.path.join(MODELS_DIR, 'fcm_centroids.pkl'))


    # ==========================================
    # 3. SVM Classifier Training
    # ==========================================
    print("⏳ [STEP 3] Training Core SVM Classifier...")
    svm_model = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_scaled, y)

    joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.pkl'))


    # ==========================================
    # 4. SHAP Computation & Cluster Means
    # ==========================================
    print("⏳ [STEP 4] Computing SHAP Values & Cluster Centroids...")
    explainer = shap.LinearExplainer(svm_model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    joblib.dump(explainer, os.path.join(MODELS_DIR, 'shap_explainer.pkl'))

    cluster_mean_shap = {}
    for i in range(N_CLUSTERS):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            cluster_mean_shap[i] = np.mean(shap_values[cluster_indices], axis=0)
        else:
            # Fallback for empty clusters to prevent dimensionality errors
            cluster_mean_shap[i] = np.zeros(X_scaled.shape[1])

    joblib.dump(cluster_mean_shap, os.path.join(MODELS_DIR, 'cluster_mean_shap.pkl'))


    # ==========================================
    # 5. FAISS Index Generation (Implicit RAG)
    # ==========================================
    print("⏳ [STEP 5] Building FAISS Vector Index for RAG Augmentation...")
    
    # Concatenating raw scaled features with their corresponding SHAP explanations
    combined_vectors = np.hstack([X_scaled, shap_values]).astype('float32')

    dimension = combined_vectors.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(combined_vectors)

    faiss.write_index(faiss_index, os.path.join(MODELS_DIR, 'rag_index.faiss'))
    joblib.dump(y, os.path.join(MODELS_DIR, 'rag_labels.pkl')) 

    print("✅ [PIPELINE SUCCESS] Offline training complete. All models saved successfully!")


if __name__ == "__main__":
    run_offline_training_pipeline()