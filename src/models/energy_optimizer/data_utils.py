import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Load the synthetic network data.
    """
    return pd.read_csv(file_path)

def add_energy_efficiency_label(data):
    """
    Add an energy efficiency label based on existing features.
    """
    energy_score = (
        0.4 * data['bandwidth'] +
        0.3 * data['throughput'] -
        0.2 * data['latency'] -
        0.05 * data['packet_loss'] -
        0.05 * data['jitter']
    )

    # Normalize energy score
    data['energy_score'] = (energy_score - energy_score.min()) / (energy_score.max() - energy_score.min())

    # Add a label: "Efficient" if energy_score > 0.7, else "Inefficient"
    data['energy_label'] = np.where(data['energy_score'] > 0.7, 'Efficient', 'Inefficient')
    return data

def preprocess_data(data):
    """
    Preprocess the data for training.
    """
    X = data[['latency', 'bandwidth', 'throughput', 'packet_loss', 'jitter']].values
    y = data['energy_label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, scaler, encoder
