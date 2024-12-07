import pandas as pd

def load_data(file_path):
    """
    Load the synthetic network data.
    """
    return pd.read_csv(file_path)

def add_future_bandwidth(data, lookahead_steps=1):
    """
    Add a 'future_bandwidth' column to simulate predictive planning.
    """
    data['future_bandwidth'] = data['bandwidth'].shift(-lookahead_steps)
    data = data.dropna()  # Drop rows with NaN values
    return data
