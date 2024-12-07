from data_utils import load_data, add_anomalies
from preprocessing import preprocess_data
from model import train_isolation_forest, evaluate_model
import os

if __name__ == "__main__":
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
    data = load_data(data_path)
    augmented_data = add_anomalies(data)
    X, y, scaler = preprocess_data(augmented_data)

    # Train Isolation Forest
    model = train_isolation_forest(X, contamination=0.05)

    # Evaluate the model
    evaluate_model(model, X, y)
