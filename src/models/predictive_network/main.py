from data_utils import load_data, add_future_bandwidth
from preprocessing import preprocess_data
from model import train_model, evaluate_model
from prediction import predict_future_bandwidth
from sklearn.model_selection import train_test_split
import numpy as np
import os

if __name__ == "__main__":
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
    data = load_data(data_path)
    data = add_future_bandwidth(data)
    X, y, scaler = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Simulate predictions
    sample_conditions = np.array([[30, 500, 100, 1, 5],  # Example network conditions
                                   [80, 800, 400, 0.5, 2]])
    future_bandwidth_predictions = predict_future_bandwidth(model, scaler, sample_conditions)

    print("\nFuture Bandwidth Predictions for Sample Conditions:")
    for i, pred in enumerate(future_bandwidth_predictions):
        print(f"Sample {i+1}: Predicted Future Bandwidth = {pred:.2f} Mbps")
