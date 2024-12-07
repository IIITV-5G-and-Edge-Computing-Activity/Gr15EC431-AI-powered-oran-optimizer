import pandas as pd

def load_data(file_path):
    """
    Load the synthetic network data.
    """
    return pd.read_csv(file_path)
