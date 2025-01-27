import argparse
import joblib
from prediction import ArtistPredictor  


def create_arg_parser():
    """
    Creates the argument parser for the CLI.
    
    Returns:
        ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(description="Predict the artist based on lyrics") 
    
    # Define the arguments (text input for prediction)
    parser.add_argument('lyrics', type=str, help="The lyrics text to predict the artist")
    
    return parser

def main():
    """
    Main function to handle the CLI interaction.
    """
    parser = create_arg_parser()   #get the parser
    args = parser.parse_args()  #parsing comand line arguments
    

    model_path = 'logistic_regression_best_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    
    # Initialize the ArtistPredictor with the model and vectorizer paths
    predictor = ArtistPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
    
    # Get the input lyrics from the command line
    input_lyrics = args.lyrics
    
    # Predict the artist based on the input lyrics
    predicted_artist = predictor.predict_artist(input_lyrics)
    
    # Output the result
    print(f"\nThe predicted artist is: {predicted_artist}")

if __name__ == "__main__":
    main()