import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class ArtistPredictor:
    def __init__(self, model_path, vectorizer_path):
        """
        Initializes the ArtistPredictor with a trained model and vectorizer.
        
        Args:
            model_path (str): Path to the pre-trained model file.
            vectorizer_path (str): Path to the pre-trained vectorizer file.
        """
        # Load the pre-trained model and vectorizer
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict_artist(self, input_text):
        """
        Predicts the artist based on input text.
        
        Args:
            input_text (str): The lyrics text input.
        
        Returns:
            str: Predicted artist name.
        """
        # Preprocess and vectorize the input text
        input_vectorized = self.vectorizer.transform([input_text])
        
        # Predict using the loaded model
        prediction = self.model.predict(input_vectorized)
        
        # Return the predicted artist name
        return prediction[0]

# Example of how to use the class
if __name__ == "__main__":
    # Initialize the ArtistPredictor with paths to the model and vectorizer
    predictor = ArtistPredictor(model_path='logistic_regression_best_model.pkl', 
                                vectorizer_path='vectorizer.pkl')

    # Input text (can be interactive in a notebook or script)
    input_text = input("Enter lyrics: ")

    # Predict the artist
    predicted_artist = predictor.predict_artist(input_text)

    # Display the result
    print(f"\nThe predicted artist is: {predicted_artist}")
