import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import dateparser
from io import BytesIO
import json

st.set_page_config(page_title="News Classifier", layout="centered")

# -------------------- Helper Functions --------------------

def fetch_html(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except:
        return ""

def extract_article_text(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs]).strip()
    except:
        return ""

def compare_headline_to_article(headline, article):
    try:
        vectorizer = TfidfVectorizer().fit_transform([headline, article])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        return similarity[0][0]
    except:
        return 0

def extract_date(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Try meta tags
        meta_date = soup.find('meta', {'property': 'article:published_time'}) \
                    or soup.find('meta', {'name': 'pubdate'}) \
                    or soup.find('meta', {'name': 'date'}) \
                    or soup.find('meta', {'itemprop': 'datePublished'})
        if meta_date and meta_date.get('content'):
            parsed = dateparser.parse(meta_date['content'])
            if parsed:
                return parsed.date()

        # Try JSON-LD
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            try:
                data = json.loads(json_ld.string)
                if isinstance(data, dict):
                    date_str = data.get('datePublished') or data.get('dateCreated')
                    if date_str:
                        parsed = dateparser.parse(date_str)
                        if parsed:
                            return parsed.date()
            except Exception:
                pass

        # Fallback to regex in full text
        text = soup.get_text()
        date_match = re.findall(
            r'(\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|\b(?:\d{1,2}[-\s]*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-,\s.]*\d{2,4})\b)',
            text, re.IGNORECASE)
        if date_match:
            parsed = dateparser.parse(date_match[0])
            if parsed:
                return parsed.date()
    except:
        pass
    return None

def classify_article(text):
    text = text.lower()
    if 'india' in text and any(word in text for word in ['positive', 'success', 'support', 'development']):
        return 'Pro-India'
    elif 'china' in text and any(word in text for word in ['sanction', 'tariff', 'ban', 'conflict', 'tension']):
        return 'Anti-China'
    elif 'pakistan' in text and any(word in text for word in ['polio', 'terror', 'crisis', 'restriction', 'attack']):
        return 'Anti-Pakistan'
    else:
        return 'Miscellaneous'

def process_dataframe(df):
    # Assume headlines in col 1 and links in col 2
    headlines = df.iloc[1:, 1].reset_index(drop=True)
    links = df.iloc[1:, 2].reset_index(drop=True)

    dates = []
    labels = []

    for headline, link in zip(headlines, links):
        html = fetch_html(link)
        article = extract_article_text(html)
        compare_headline_to_article(headline, article)  # Optional
        date = extract_date(html)
        label = classify_article(article)
        dates.append(date)
        labels.append(label)

    # Expand DataFrame with new columns
    df_new = df.copy()
    df_new.columns = range(df_new.shape[1])  # Temporarily number columns

    # Add headers in first row
    df_new.at[0, df_new.shape[1]] = 'Date'
    df_new.at[0, df_new.shape[1]] = 'Classification'

    # Add values from second row onward
    for i in range(1, len(df_new)):
        df_new.at[i, df_new.shape[1] - 2] = dates[i - 1]
        df_new.at[i, df_new.shape[1] - 1] = labels[i - 1]

    return df_new

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, header=False)
    output.seek(0)
    return output

# -------------------- Streamlit Interface --------------------

st.title("📰 News Headline Classifier")
st.write("Upload an Excel file with headlines and URLs. The tool will fetch article content, compare it, extract date, and classify it as Pro-India, Anti-China, Anti-Pakistan, or Miscellaneous.")

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=None)
        st.success("✅ File uploaded successfully!")

        with st.spinner("Processing articles..."):
            updated_df = process_dataframe(df)
            output_excel = convert_df_to_excel(updated_df)

        st.success("✅ Processing complete!")
        st.write("🔍 Preview of output:")
        st.dataframe(updated_df.head(10), use_container_width=True)

        st.download_button(
            label="📥 Download Processed Excel",
            data=output_excel,
            file_name="Updated_Headlines.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
