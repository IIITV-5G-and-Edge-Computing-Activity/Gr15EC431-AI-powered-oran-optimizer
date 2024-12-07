from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the data for anomaly detection.
    """
    X = data[['latency', 'bandwidth', 'throughput', 'packet_loss', 'jitter']].values
    y = data['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler
