import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


rhcp = []
with open('rhcp.txt', 'r') as file:
    for line in file:
        rhcp.append(line.strip())  


madonna = []
with open('madonna.txt', 'r') as file:
    for line in file:
        madonna.append(line.strip())  

class TextProcessor:
    def __init__(self, texts, language='english', font_path=None):
        """
        Initializes the TextProcessor with given texts and language.

        Args:
            texts (list): List of text data (e.g., lyrics, articles).
            language (str): Language for stopword filtering (default is English).
            font_path (str): Path to the font for the word cloud (optional).
        """
        self.texts = texts
        self.language = language
        self.font_path = font_path
        self.stop_words = set(stopwords.words(self.language))   #

    def tokenize_sentences(self):
        """
        Tokenizes the text into sentences.

        Returns:
            list: List of lists containing sentences.
        """
        return [sent_tokenize(text) for text in self.texts]

    def tokenize_words(self):
        """
        Tokenizes the text into words.

        Returns:
            list: List of lists containing tokenized words.
        """
        return [word_tokenize(text) for text in self.texts]

    def filter_words(self, words):
        """
        Filters out stopwords and non-alphabetic words from tokenized words.

        Args:
            words (list): List of tokenized words.

        Returns:
            list: List of filtered words.
        """
        return [
            [word for word in word_list if word.lower() not in self.stop_words and word.isalpha()]
            for word_list in words
        ]

    def get_frequency_distribution(self, filtered_words):
        """
        Calculates the frequency distribution of words.

        Args:
            filtered_words (list): List of filtered words.

        Returns:
            FreqDist: Frequency distribution of words.
        """
        flat_filtered_words = [word for word_list in filtered_words for word in word_list]
        return FreqDist(flat_filtered_words)

    def generate_word_cloud(self, freq_dist, width=800, height=600):
        """
        Generates a word cloud from the frequency distribution.

        Args:
            freq_dist (FreqDist): Frequency distribution of words.
            width (int): Width of the word cloud image.
            height (int): Height of the word cloud image.
        
        Returns:
            WordCloud: WordCloud object.
        """
        word_freq = dict(freq_dist)

        # Custom color function for the word cloud
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return "hsl({}, 100%, 50%)".format(np.random.randint(0, 360))  # Random hue

        # Create the word cloud
        wordcloud = WordCloud(
            font_path=self.font_path,
            width=width,
            height=height,
            prefer_horizontal=0.5,
            background_color="black",  # Set background color
            color_func=color_func,  # Pass the custom color function
            random_state=42
        ).generate_from_frequencies(word_freq)

        return wordcloud

    def plot_word_cloud(self, wordcloud, file_path="wordcloud.png"):
        """
        Plots the generated word cloud.

        Args:
            wordcloud (WordCloud): WordCloud object to be plotted.
        """
        plt.figure(figsize=(10, 14))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Hide the axes
        plt.savefig(file_path, format='png')
        plt.show()



madonna_processor = TextProcessor(madonna, language='english', font_path='/Users/saramaras/Documents/github/text_classification/Kanit/Kanit-Regular.ttf')


madonna_sentences = madonna_processor.tokenize_sentences()
madonna_words = madonna_processor.tokenize_words()
filtered_madonna_words = madonna_processor.filter_words(madonna_words)
freq_dist_madonna = madonna_processor.get_frequency_distribution(filtered_madonna_words)


wordcloud_madonna = madonna_processor.generate_word_cloud(freq_dist_madonna, width=600, height=600)
madonna_processor.plot_word_cloud(wordcloud_madonna, file_path="madonna_wordcloud.png")


rhcp_processor = TextProcessor(rhcp, language='english', font_path='/Users/saramaras/Documents/github/text_classification/Kanit/Kanit-Regular.ttf')


rhcp_sentences = rhcp_processor.tokenize_sentences()
rhcp_words = rhcp_processor.tokenize_words()
filtered_rhcp_words = rhcp_processor.filter_words(rhcp_words)
freq_dist_rhcp = rhcp_processor.get_frequency_distribution(filtered_rhcp_words)

# Generate word cloud for RHCP
wordcloud_rhcp = rhcp_processor.generate_word_cloud(freq_dist_rhcp, width=600, height=600)
rhcp_processor.plot_word_cloud(wordcloud_rhcp, file_path="rhcp_wordcloud.png")
       




