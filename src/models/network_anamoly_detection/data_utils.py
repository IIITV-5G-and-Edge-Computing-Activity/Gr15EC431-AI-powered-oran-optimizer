import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the synthetic network data.
    """
    return pd.read_csv(file_path)

def add_anomalies(data, anomaly_fraction=0.05, random_seed=42):
    """
    Add synthetic anomalies to the dataset.
    """
    np.random.seed(random_seed)
    n_anomalies = int(len(data) * anomaly_fraction)

    # Generate anomalies by introducing extreme values in the feature space
    anomalies = data.sample(n=n_anomalies).copy()
    anomalies['latency'] *= np.random.uniform(1.5, 3.0, n_anomalies)  # Increase latency
    anomalies['bandwidth'] *= np.random.uniform(0.1, 0.5, n_anomalies)  # Decrease bandwidth
    anomalies['throughput'] *= np.random.uniform(0.1, 0.5, n_anomalies)  # Decrease throughput
    anomalies['packet_loss'] += np.random.uniform(10, 20, n_anomalies)  # Increase packet loss
    anomalies['jitter'] += np.random.uniform(10, 30, n_anomalies)  # Increase jitter

    anomalies['label'] = 1  # Label anomalies as 1
    data['label'] = 0  # Label normal data as 0

    # Combine normal data with anomalies
    augmented_data = pd.concat([data, anomalies], ignore_index=True)
    return augmented_data
