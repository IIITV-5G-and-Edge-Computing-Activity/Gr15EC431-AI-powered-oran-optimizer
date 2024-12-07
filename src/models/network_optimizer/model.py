from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Classifier model.
    """
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, encoder):
    """
    Evaluate the model on test data.
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
