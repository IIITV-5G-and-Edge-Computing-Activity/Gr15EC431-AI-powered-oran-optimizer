from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the data for predictive modeling.
    """
    # Separate features and target variable
    X = data[['latency', 'bandwidth', 'throughput', 'packet_loss', 'jitter']].values
    y = data['future_bandwidth'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler
