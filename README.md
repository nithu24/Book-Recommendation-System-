# 📚 Book Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-orange)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **A content-based book recommendation system built on a custom-scraped GoodReads dataset — recommending the top 10 most similar books for any given title using TF-IDF vectorization and cosine similarity, deployed as a live Streamlit web application.**

---

## 🎯 What Makes This Project Unique

Most recommendation system projects use pre-existing Kaggle datasets. This project **builds the dataset from scratch** by scraping GoodReads using BeautifulSoup and Requests — collecting 750+ books across multiple genres with 9 attributes per book. The entire pipeline goes from raw web pages to a live deployed recommendation app.

---

## 📁 Project Structure

```
book-recommendation-system/
│
├── GoodReadsScraping.ipynb           # Web scraping pipeline — GoodReads data collection
├── appending_excel_files.ipynb       # Merging 10+ genre-specific CSV files into master dataset
├── recommendation_system.ipynb       # EDA + preprocessing + TF-IDF + cosine similarity matrix
├── feature_extraction__1_.ipynb      # Feature engineering and vectorization
├── app.py                            # Streamlit web application
│
├── book_dataset.pkl                  # Saved book DataFrame
├── cosinesimilarity.pkl              # Pre-computed cosine similarity matrix
├── indices.pkl                       # Book title to index mapping
├── book.jpg                          # Background image for app
│
└── README.md
```

---

## 📊 Dataset — Custom Scraped from GoodReads

| Property | Details |
|----------|---------|
| **Source** | GoodReads (scraped using BeautifulSoup + Requests) |
| **Size** | 750+ books |
| **Genres Scraped** | Biography, Cookbooks, Poetry, Fiction, Mystery, Romance, Self-Help, Science, History, Fantasy |
| **Attributes per Book** | 9 (title, author, summary, genres, rating, number of ratings, page count, publication year, URL) |
| **Format** | Multiple genre CSVs → merged master DataFrame |

### Why Scrape Instead of Using Kaggle?

- **Complete control** over which attributes to collect — summaries were specifically needed for content-based filtering
- **Domain coverage** — GoodReads is the world's largest book review site with rich metadata
- **Real-world skill** — building your own dataset is a core data engineering capability
- **Genre diversity** — custom selection of genres ensures broad recommendation coverage

---

## 🕷️ Web Scraping Pipeline

### Tools Used
- **Requests** — HTTP GET requests to GoodReads URLs
- **BeautifulSoup** — HTML parsing and data extraction
- **Pandas** — DataFrame construction and CSV export

### Scraping Process

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_goodreads_books(url, num_pages):
    books = []
    for page in range(1, num_pages + 1):
        response = requests.get(f"{url}?page={page}")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for book in soup.find_all('tr', {'itemtype': 'http://schema.org/Book'}):
            title = book.find('a', class_='bookTitle').text.strip()
            author = book.find('a', class_='authorName').text.strip()
            rating = book.find('span', class_='minirating').text.strip()
            # ... extract all 9 attributes
            books.append({...})
        
        time.sleep(1)  # Respectful rate limiting
    
    return pd.DataFrame(books)
```

### Data Consolidation

```python
import glob

# Merge all genre-specific CSVs into master dataset
all_files = glob.glob("*.csv")
master_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
master_df = master_df.drop_duplicates(subset='title')  # Remove books in multiple genres
```

---

## ⚙️ Technical Pipeline

### Step 1 — Text Preprocessing

```python
def clean_title(text):
    text = text.lower()
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    text_tokens = word_tokenize(text)
    text_tokens = ' '.join(text_tokens)
    return text_tokens
```

Custom preprocessing pipeline:
- Lowercasing
- Punctuation removal
- Tokenization using NLTK word_tokenize
- Stopword removal for cleaner summary text

### Step 2 — TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),      # Unigrams, bigrams, trigrams
    max_features=5000,        # Top 5000 most important terms
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(book_data['summary'])
```

**Why TF-IDF with n-grams (1–3)?**
- Unigrams capture individual keywords: "mystery", "detective"
- Bigrams capture phrases: "Victorian London", "murder mystery"
- Trigrams capture longer patterns: "plot twist ending"
- `max_features=5000` keeps the most discriminative terms

### Step 3 — Cosine Similarity Matrix

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Result: 750×750 matrix where cosine_sim[i][j] = similarity between book i and book j
```

**Why Cosine Similarity over Euclidean Distance?**
- Cosine similarity measures the **angle** between vectors — ignores magnitude (document length)
- A short and long book summary about the same topic will have high cosine similarity
- Euclidean distance would penalize shorter summaries unfairly

### Step 4 — Recommendation Engine

```python
def recommend(title, sig=sg):
    idx = indices[title]                           # Get book index
    sg_score = list(enumerate(sg[idx]))            # Get similarity scores
    sg_score = sorted(sg_score, key=lambda x: x[1], reverse=True)  # Sort descending
    sg_score = sg_score[1:11]                      # Top 10 (exclude self)
    book_indices = [i[0] for i in sg_score]
    return book_data[['title', 'genres']].iloc[book_indices]
```

**Pre-computation for Speed:**
- Cosine similarity matrix computed **once** and saved as pickle
- At inference time, just one row lookup + sort — near-instantaneous response
- No re-computation needed per query

### Step 5 — Offline Storage

```python
import pickle

pickle.dump(book_data, open('book_dataset.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosinesimilarity.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))
```

---

## 📈 Exploratory Data Analysis

Key EDA performed in `recommendation_system.ipynb`:

- **Genre distribution** — bar chart showing book count per genre
- **Top 10 most prolific authors** — frequency analysis
- **Rating distribution** — histogram of book ratings across dataset
- **Page count analysis** — distribution of book lengths by genre
- **Summary length analysis** — vocabulary richness per genre
- **WordCloud** — most frequent words in book summaries across all genres

---

## 🚀 Streamlit Web Application

### App Features

- **Text input** — enter any book title from the dataset
- **Top 10 recommendations** — displayed in a clean DataFrame with title and genre
- **Background image** — custom book-themed background

### App Screenshot Preview

<img width="1901" height="976" alt="Screenshot (482)" src="https://github.com/user-attachments/assets/3dda79aa-5334-464f-b848-22636263c0bf" />

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
nltk
textblob
matplotlib
beautifulsoup4
requests
pickle-mixin
```

Install all at once:

```bash
pip install streamlit pandas numpy scikit-learn nltk textblob matplotlib beautifulsoup4 requests
```

---

## 🔍 How Content-Based Filtering Works

```
User Input: "The Da Vinci Code"
         │
         ▼
Clean & Tokenize Title
         │
         ▼
Look up book index in indices dictionary
         │
         ▼
Retrieve row from cosine_similarity matrix
         │
         ▼
Sort all 750 books by similarity score (descending)
         │
         ▼
Return top 10 (excluding the input book itself)
         │
         ▼
Display: Titles + Genres + Similarity Bar Chart
```

---

## 💡 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Content-based filtering** | Recommends items similar to what user liked, based on item features |
| **TF-IDF** | Term Frequency × Inverse Document Frequency — weights words by importance |
| **Cosine similarity** | Measures angle between vectors; high value = similar content |
| **N-grams** | Sequences of N words — captures phrases not just individual words |
| **Cold start problem** | New items can be recommended immediately using content (no user history needed) |
| **Pre-computation** | Similarity matrix built once offline; lookup is O(n log n) at inference |

---

## 🔮 Future Improvements

- **Sentence Transformers** — replace TF-IDF with `all-MiniLM-L6-v2` for richer semantic similarity
- **Collaborative filtering** — add user ratings data from GoodReads API for hybrid recommendations
- **FAISS integration** — scalable approximate nearest neighbor search for millions of books
- **User profiles** — allow users to input multiple liked books for personalized recommendations
- **Diversity constraints** — prevent recommending only very similar books (explore vs exploit balance)
- **Evaluation metrics** — implement Precision@K, Recall@K, NDCG for quantitative evaluation
- **Live GoodReads API** — replace static dataset with real-time scraping for always-fresh data

---

## 📊 Why This Approach Works

The TF-IDF + cosine similarity approach is highly effective for book recommendation because:

1. **Book summaries are rich** — they describe themes, settings, characters, and plot elements that distinguish books
2. **N-grams capture genre signals** — "murder mystery", "Victorian England", "self-help productivity" are powerful discriminative phrases
3. **No user data needed** — works immediately for any new user without a history
4. **Interpretable** — you can understand why two books are similar by looking at their shared high-TF-IDF terms
5. **Computationally efficient** — pre-computed matrix makes real-time inference near-instantaneous

---

## 👩‍💻 Author

**Nithu Anna Ninan**
- 🔗 LinkedIn: [linkedin.com/in/nithu-ninan](https://linkedin.com/in/nithu-ninan)
- 📧 Email: nithuanna24@gmail.com
- 🐙 GitHub: [github.com/nithu-ninan](https://github.com/nithu24)

---

## ⭐ If you found this project useful, please give it a star!

---

*Built with Python · BeautifulSoup · Scikit-learn · NLTK · Streamlit*
