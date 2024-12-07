from data_utils import load_data, add_energy_efficiency_label, preprocess_data
from model import train_model, evaluate_model
from predict import simulate_real_time_optimization
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    # Load and preprocess data
    data_path = os.path.join(os.path.dirname(__file__), "../../data/data.csv")
    data = load_data(data_path)
    data = add_energy_efficiency_label(data)
    X, y, scaler, encoder = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, encoder)

    # Simulate real-time optimization
    simulate_real_time_optimization(model, scaler, encoder)
