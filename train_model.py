import pickle
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_clean_data
from sklearn.metrics import accuracy_score

# Train a Random Forest model and save it as model.pkl
def train_and_save_model():
    # Load clean data
    X_train, X_test, y_train, y_test = load_and_clean_data()

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))

    # Save the trained model to file
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved to model.pkl")

# Run this to train
if __name__ == "__main__":
    train_and_save_model()
