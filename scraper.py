import requests
import re

from bs4 import BeautifulSoup



import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def get_links(url: str, header: dict) -> list:
    """
    Fetches the first 100 lyric links from the given URL.

    Args:
        url (str): The URL of the page to scrape.
        header (dict): Headers to use for the HTTP request.

    Returns:
        list: A list of up to 100 full URLs for lyrics, or an empty list if an error occurs.
    """
    try:
        # Send the HTTP request
        response = requests.get(url, headers=header)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []  # Return an empty list if the request fails

    # Parse the page content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find and filter lyric links
    relative_links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if re.match(r'^/lyric/\d+/[^/]+/[^/]+$', href):
            relative_links.append(href)

    # Construct full URLs
    full_links = [BASE_URL + link for link in relative_links]

    # Return the first 100 links
    return full_links[:100]

BASE_URL = "https://www.lyrics.com"
url = 'https://www.lyrics.com/artist/Madonna/64565'
header = {'User-agent': 'Mozilla/5.0 (X11; Linux i686; rv:2.0b10) Gecko/20100101 Firefox/4.0b10'}
links_madonna= get_links(url, header)

BASE_URL = "https://www.lyrics.com"
url = "https://www.lyrics.com/artist/Red-Hot-Chili-Peppers/5241"
header = {'User-agent': 'Mozilla/5.0 (X11; Linux i686; rv:2.0b10) Gecko/20100101 Firefox/4.0b10'}
links_rhcp = get_links(url, header)

def clean_corpus(links: list, header: dict) -> list:
    """
    Cleans the lyrics from the given song links.

    Args:
        links (list): A list of URLs for songs to scrape lyrics from.
        header (dict): Headers to use for the HTTP request.

    Returns:
        list: A list of cleaned lyric texts.
    """
    corpus = []  # To store the raw HTML content of each page
    
    # Loop through each song link and fetch its content
    for song in links:
        response = requests.get(song, headers=header)
        corpus.append(response.text)  # Append the raw HTML of the song's page
    
    # List to store cleaned lyrics
    lyric_texts = []

    # Loop through the raw HTML content (corpus) of each song
    for page_content in corpus:
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Find and filter <pre> elements with the class 'lyric-body'
        lyric_elements = soup.find_all('pre', class_='lyric-body')
        
        # Loop through each lyric element and clean its text
        for element in lyric_elements:
            lyric_text = element.get_text()

            # Clean unwanted escape sequences like '\\r' and '\\n'
            cleaned_lyric_text = lyric_text.replace("\\r\\n", " ")  # Replace escaped line breaks with a space
            cleaned_lyric_text = cleaned_lyric_text.replace("\\r", "")  # Remove carriage returns
            cleaned_lyric_text = cleaned_lyric_text.replace("\\n", "")  # Remove new lines
            cleaned_lyric_text = cleaned_lyric_text.replace("\\", "")  # Remove extra backslashes

            
            cleaned_lyric_text = ' '.join(cleaned_lyric_text.split())

           
            lyric_texts.append(cleaned_lyric_text.strip())  # Strip to remove leading/trailing spaces

    return lyric_texts


madonna = clean_corpus(links_madonna, header)
rhcp = clean_corpus(links_rhcp, header)


with open('rhcp.txt', 'w') as file:
    file.write('\n'.join(rhcp))  


with open('madonna.txt', 'w') as file:
    file.write('\n'.join(madonna))  # Join list elements with newline characters

