import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load and preprocess data
def load_data(filepath):
    """
    Load the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

# Step 2: Text preprocessing
nltk.download('punkt')

def preprocess(text):
    """
    Preprocess text by tokenizing and lowercasing.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

# Step 3: Convert text to numerical features
def vectorize_text(texts):
    """
    Convert text data into numerical features using TF-IDF.

    Args:
        texts (list): List of preprocessed text.

    Returns:
        tuple: (TF-IDF vectorized array, fitted vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    return X, vectorizer

# Step 4: Train/test split
def split_data(features, labels):
    """
    Split data into training and testing sets.

    Args:
        features (numpy.ndarray): Feature matrix.
        labels (numpy.ndarray): Label array.

    Returns:
        tuple: Train/test splits for features and labels.
    """
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 5: Train a model
def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error.

    Args:
        model: Trained model.
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): True labels for testing set.

    Returns:
        float: Mean Squared Error of predictions.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Step 7: Predict new scores
def grade_essay(model, vectorizer, new_essay):
    """
    Predict a score for a new essay.

    Args:
        model: Trained model.
        vectorizer: Fitted TF-IDF vectorizer.
        new_essay (str): Essay to grade.

    Returns:
        float: Predicted score.
    """
    new_essay_cleaned = preprocess(new_essay)
    new_essay_vectorized = vectorizer.transform([new_essay_cleaned]).toarray()
    predicted_score = model.predict(new_essay_vectorized)
    return predicted_score[0]

if __name__ == "__main__":
    # Load data
    data = load_data('data/essay_dataset.csv')
    if data is not None:
        essays = data['essay']
        scores = data['score']

        # Preprocess essays
        essays_cleaned = essays.apply(preprocess)

        # Vectorize essays
        X, vectorizer = vectorize_text(essays_cleaned)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, scores)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        mse = evaluate_model(model, X_test, y_test)
        print(f'Mean Squared Error: {mse}')

        # Example prediction
        new_essay = "This is an example essay to grade."
        predicted_score = grade_essay(model, vectorizer, new_essay)
        print(f"Predicted Score: {predicted_score}")
