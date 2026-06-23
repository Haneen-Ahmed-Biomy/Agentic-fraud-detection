import numpy as np
from typing import Tuple
from scipy.spatial.distance import cosine

class IntelligentScorer:
    """
    A scoring engine that aggregates multiple signals (Severity, SVM, FAISS, ARIMA, SHAP)
    to compute a final, intelligent fraud risk score.
    """
    
    def __init__(self, weight_severity: float = 0.33, weight_gap: float = 0.33, weight_mu: float = 0.34):
        # Weights for the three primary base signals
        self.weight_severity = weight_severity
        self.weight_gap = weight_gap
        self.weight_mu = weight_mu

    def calculate_shap_cosine(self, current_shap: np.ndarray, cluster_mean_shap: np.ndarray) -> float:
        """
        Calculates the cosine similarity between the current transaction's SHAP values
        and the historical mean SHAP values of its assigned cluster.
        
        Returns:
            float: A similarity score bounded between 0.0 and 1.0.
        """
        # Scipy computes cosine distance, so similarity is (1 - distance)
        similarity = 1 - cosine(current_shap, cluster_mean_shap)
        
        # Floor at 0 to prevent negative multipliers from inverting the final score
        return max(0.0, float(similarity))

    def compute_final_score(
        self, 
        severity_signal: float, 
        svm_confidence_gap: float, 
        cluster_membership_prob: float, 
        arima_trend_multiplier: float, 
        current_shap: np.ndarray, 
        cluster_mean_shap: np.ndarray
    ) -> Tuple[float, float]:
        """
        Computes the final aggregated fraud risk score using a weighted base score 
        adjusted by temporal (ARIMA) and explainability (SHAP) multipliers.
        """
        
        # 1. Compute Explainability Reliability Multiplier
        shap_reliability = self.calculate_shap_cosine(current_shap, cluster_mean_shap)
        
        # 2. Compute Base Score (Weighted Sum of Primary Signals)
        base_score = (
            (self.weight_severity * severity_signal) + 
            (self.weight_gap * svm_confidence_gap) + 
            (self.weight_mu * cluster_membership_prob)
        )
        
        # 3. Apply Multipliers (Temporal Trend & Feature Reliability)
        final_score = base_score * arima_trend_multiplier * shap_reliability
        
        return final_score, shap_reliability


# ==========================================
# Unit Testing Block
# ==========================================
if __name__ == "__main__":
    # Initialize the scoring engine
    scorer = IntelligentScorer()
    
    # Simulated live transaction signals
    simulated_severity = 0.8         # High transaction amount
    simulated_svm_gap = 0.9          # High confidence from the SVM classifier
    simulated_membership = 0.85      # Strong geometric alignment to a fraud cluster
    simulated_arima = 1.2            # Increasing temporal fraud trend
    
    # Simulated SHAP explanation vectors
    live_shap = np.array([0.5, 0.2, 0.1])
    historical_shap = np.array([0.45, 0.25, 0.1])
    
    # Compute the score
    final_risk_score, shap_cos = scorer.compute_final_score(
        severity_signal=simulated_severity,
        svm_confidence_gap=simulated_svm_gap,
        cluster_membership_prob=simulated_membership,
        arima_trend_multiplier=simulated_arima,
        current_shap=live_shap,
        cluster_mean_shap=historical_shap
    )
    
    print("=== Intelligent Scorer Test ===")
    print(f"SHAP Reliability (Cosine): {shap_cos:.4f}")
    print(f"Final Intelligent Score:   {final_risk_score:.4f}")