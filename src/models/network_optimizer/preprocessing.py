from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):
    """
    Preprocess the data for training.
    """
    # Separate features and labels
    X = data[['latency', 'bandwidth', 'throughput', 'packet_loss', 'jitter']].values
    y = data['action'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, scaler, encoder
