import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import dateparser
import json
from io import BytesIO

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

        # 1. Meta tags
        meta_tags = [
            {'property': 'article:published_time'},
            {'name': 'pubdate'},
            {'name': 'date'},
            {'itemprop': 'datePublished'}
        ]
        for tag in meta_tags:
            meta = soup.find('meta', tag)
            if meta and meta.get('content'):
                parsed = dateparser.parse(meta['content'], settings={'DATE_ORDER': 'DMY'})
                if parsed:
                    return parsed.date()

        # 2. JSON-LD
        json_lds = soup.find_all('script', type='application/ld+json')
        for tag in json_lds:
            try:
                data = json.loads(tag.string)
                if isinstance(data, dict):
                    date_str = data.get('datePublished') or data.get('dateCreated')
                    if date_str:
                        parsed = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
                        if parsed:
                            return parsed.date()
                elif isinstance(data, list):
                    for item in data:
                        date_str = item.get('datePublished') or item.get('dateCreated')
                        if date_str:
                            parsed = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
                            if parsed:
                                return parsed.date()
            except:
                continue

        # 3. <time> tag
        time_tag = soup.find('time')
        if time_tag:
            date_str = time_tag.get('datetime') or time_tag.get_text(strip=True)
            if date_str:
                parsed = dateparser.parse(date_str, settings={'DATE_ORDER': 'DMY'})
                if parsed:
                    return parsed.date()

        # 4. Common class/id based selectors
        selectors = ['date', 'published', 'pubdate', 'post-date', 'article-date']
        for sel in selectors:
            tag = soup.find(attrs={'class': re.compile(sel, re.I)})
            if not tag:
                tag = soup.find(attrs={'id': re.compile(sel, re.I)})
            if tag:
                parsed = dateparser.parse(tag.get_text(strip=True), settings={'DATE_ORDER': 'DMY'})
                if parsed:
                    return parsed.date()

        # 5. JavaScript-embedded date
        for script in soup.find_all('script'):
            if script.string:
                found = re.findall(
                    r'(?i)(?:"datePublished"|\'datePublished\'|published_time)["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    script.string)
                for f in found:
                    parsed = dateparser.parse(f, settings={'DATE_ORDER': 'DMY'})
                    if parsed:
                        return parsed.date()

        # 6. Fallback: Search text for date formats like "15 Dec 2024"
        text = soup.get_text(separator=' ')
        match = re.search(r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b',
                          text, re.IGNORECASE)
        if match:
            parsed = dateparser.parse(match.group(), settings={'DATE_ORDER': 'DMY'})
            if parsed:
                return parsed.date()

    except Exception as e:
        print(f"Date parsing error: {e}")
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
    headlines = df.iloc[1:, 1].reset_index(drop=True)
    links = df.iloc[1:, 2].reset_index(drop=True)

    dates = []
    labels = []

    for headline, link in zip(headlines, links):
        html = fetch_html(link)
        article = extract_article_text(html)
        compare_headline_to_article(headline, article)
        date = extract_date(html)
        if date:
            date = date.strftime("%d/%m/%Y")  # Format to DD/MM/YYYY
        else:
            date = ""
        label = classify_article(article)
        dates.append(date)
        labels.append(label)

    df_new = df.copy()
    df_new.columns = range(df_new.shape[1])

    df_new.at[0, df_new.shape[1]] = 'Extracted Date'
    df_new.at[0, df_new.shape[1]] = 'Classification'

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
st.write("Upload an Excel file with headlines and URLs. This tool will fetch article content, extract the date, compare headline and classify as Pro-India, Anti-China, Anti-Pakistan, or Miscellaneous.")

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
