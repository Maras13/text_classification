import tkinter as tk
from tkinter import messagebox
from prediction import ArtistPredictor  # Import the class from the previous script

class ArtistPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Predictor")
        self.root.geometry("500x300")
        self.root.configure(bg="black")  # Set background color to black
        self.create_widgets()

    def create_widgets(self):
        """
        Creates the widgets for the GUI.
        """
        # Title Label
        self.title_label = tk.Label(self.root, text="Enter Lyrics to Predict Artist", font=("Helvetica", 16, 'bold'), bg="black", fg="white")
        self.title_label.pack(pady=20)
        
        # Lyrics Input Field
        self.lyrics_entry = tk.Entry(self.root, width=50, font=("Helvetica", 12), bd=2, relief="solid", bg="white", fg="black")
        self.lyrics_entry.pack(pady=20)
        
        # Predict Button (styled)
        self.predict_button = tk.Button(self.root, text="Predict Artist", font=("Helvetica", 12), bg="darkblue", fg="white", command=self.predict_artist, relief="raised", bd=4)
        self.predict_button.pack(pady=10)

        # Result Label (initially empty)
        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14), bg="black", fg="white")
        self.result_label.pack(pady=10)

        # Add a clear button to reset input and result
        self.clear_button = tk.Button(self.root, text="Clear", font=("Helvetica", 12), bg="gray", fg="white", command=self.clear_input, relief="raised", bd=4)
        self.clear_button.pack(pady=5)

    def predict_artist(self):
        """
        Handles the prediction when the user clicks the button.
        """
        input_lyrics = self.lyrics_entry.get()
        
        if not input_lyrics:
            messagebox.showerror("Input Error", "Please enter some lyrics.")
            return
        
        # Initialize the ArtistPredictor class
        model_path = 'logistic_regression_best_model.pkl'
        vectorizer_path = 'vectorizer.pkl'
        predictor = ArtistPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
        
        try:
            # Perform the prediction
            predicted_artist = predictor.predict_artist(input_lyrics)
            self.result_label.config(text=f"Predicted Artist: {predicted_artist}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def clear_input(self):
        """
        Clears the lyrics input and result label.
        """
        self.lyrics_entry.delete(0, tk.END)
        self.result_label.config(text="")
        self.lyrics_entry.focus()  # Focus back to the input field


def main():
    root = tk.Tk()
    app = ArtistPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
