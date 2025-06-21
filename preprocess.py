import pandas as pd
from sklearn.model_selection import train_test_split

# Load and clean the heart disease dataset
def load_and_clean_data(path="heart.csv"):
    # Read the dataset
    df = pd.read_csv(path)

    # Remove rows with any missing data
    df.dropna(inplace=True)

    # Encode 'Sex' column: Male → 1, Female → 0
    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0})

    # Select only the features used in the GUI
    selected_columns = ["Age", "Cholesterol", "RestingBP", "MaxHR", "Oldpeak", "Sex"]
    X = df[selected_columns]

    # Target column is whether the patient has heart disease
    y = df["HeartDisease"]

    # Split the dataset into training and testing sets (80/20)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# This runs if you execute the file directly (for testing)
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_clean_data()
    print("Data loaded and processed.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
