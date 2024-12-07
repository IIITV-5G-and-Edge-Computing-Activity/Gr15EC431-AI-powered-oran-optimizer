from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_isolation_forest(X, contamination=0.05):
    """
    Train an Isolation Forest model for anomaly detection.
    """
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate the anomaly detection model.
    """
    y_pred = model.predict(X)
    y_pred = np.where(y_pred == -1, 1, 0)  # Map -1 to 1 (anomaly), 1 to 0 (normal)

    # Print evaluation metrics
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))
