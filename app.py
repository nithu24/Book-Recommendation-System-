import streamlit as st
import pickle 
import base64
import re
import pandas as pd
import string
from matplotlib import pyplot as plt
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


book_data = pd.read_pickle('book_dataset.pkl')
sg = pd.read_pickle('cosinesimilarity.pkl')
indices = pd.read_pickle('indices.pkl')

def clean_title(text):
    text = text.lower()                                                        # lower case
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))         # remove punctuations
    text_tokens = word_tokenize(text)                                          # Tokenization
    text_tokens = ' '.join(text_tokens)  
    return text_tokens
clean = lambda x: clean_title(x)

def recommend(title, sig=sg):
    idx = indices[title]
    sg_score = list(enumerate(sg[idx]))
    sg_score = sorted(sg_score, key= lambda x: x[1], reverse=True)
    sg_score = sg_score[1:11]
    book_indices = [i[0] for i in sg_score]
    return book_data[['title','genres']].iloc[book_indices]


def recommend_fig(title, sig=sg):
    idx = indices[title]
    sg_score = list(enumerate(sg[idx]))
    sg_score = sorted(sg_score, key= lambda x: x[1], reverse=True)
    sg_score = sg_score[1:11]
    height = [i[1] for i in sg_score]
    book_indices = [i[0] for i in sg_score]
    plt.figure(figsize=(30,15))
    return plt.bar(book_data['title'].iloc[book_indices], height, width = 0.7,
                  color = ['red', 'green','blue','orange','yellow','brown','pink','violet','black','skyblue']) 


html_temp = """
    <div style="background-color:violet;padding:10px">
    <h1 style="color:blue;text-align:center;">BOOK RECOMMENDATION SYSTEM</h1>
    </div>
    """
    
st.markdown(html_temp, unsafe_allow_html=True)


main_bg = "book.jpg"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)



st.header("Enter the book title you want to recommend")
book_name = st.text_input("")


if st.button('GIVE ME RECOMMENDATION'):
    data = clean_title(book_name)
    result = recommend(data)
    st.write("BELOW TABLE WILL SHOW YOU TOP TEN RECOMMENDED BOOKS AND THEIR GENRE", result)