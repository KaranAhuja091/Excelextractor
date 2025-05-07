import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import dateparser

def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        return article_text.strip()
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

def compare_headline_to_article(headline, article):
    try:
        vectorizer = TfidfVectorizer().fit_transform([headline, article])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        return similarity[0][0]
    except:
        return 0

def extract_date(text):
    try:
        date_match = re.findall(r'(\b(?:\d{1,2}[-/thstndrd\s.]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s,.]*\d{2,4})\b|\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b)', text, re.IGNORECASE)
        if date_match:
            parsed_date = dateparser.parse(date_match[0])
            if parsed_date:
                return parsed_date.date()
    except:
        pass
    return None

def classify_article(text):
    text = text.lower()
    if 'india' in text and any(word in text for word in ['positive', 'success', 'support']):
        return 'Pro-India'
    elif 'china' in text and any(word in text for word in ['sanction', 'tariff', 'ban', 'conflict']):
        return 'Anti-China'
    elif 'pakistan' in text and any(word in text for word in ['polio', 'terror', 'crisis', 'restriction']):
        return 'Anti-Pakistan'
    else:
        return 'Miscellaneous'

def process_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    headlines = df.iloc[1:, 1].reset_index(drop=True)
    links = df.iloc[1:, 2].reset_index(drop=True)

    dates = []
    labels = []

    for headline, link in zip(headlines, links):
        article = extract_article_text(link)
        similarity = compare_headline_to_article(headline, article)
        date = extract_date(article)
        label = classify_article(article)

        dates.append(date)
        labels.append(label)

        print(f"Processed: {headline[:60]}... | Similarity: {similarity:.2f} | Date: {date} | Label: {label}")

    # Add new columns
    df.iloc[0, 3] = 'Date'
    df.iloc[0, 4] = 'Classification'
    df.iloc[1:, 3] = dates
    df.iloc[1:, 4] = labels

    # Construct output path in same directory
    output_path = os.path.join(os.path.dirname(file_path), "Updated_" + os.path.basename(file_path))
    df.to_excel(output_path, index=False)

    print(f"\n✅ Updated file saved as: {output_path}")

# === ✅ RUN HERE ===
# Replace the path with your actual file if needed
process_excel("Book1.xlsx")
