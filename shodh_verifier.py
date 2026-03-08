import streamlit as st
import feedparser
import urllib.parse
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def search_google_news(query):

    encoded_query = urllib.parse.quote(query)

    url = f"https://news.google.com/rss/search?q={encoded_query}"

    feed = feedparser.parse(url)

    articles = []

    for entry in feed.entries[:5]:
        articles.append(entry.title)

    return articles


def verify_claim(claim):

    articles = search_google_news(claim)

    if not articles:
        return 0, []

    claim_embedding = model.encode(claim)

    scores = []

    for article in articles:

        article_embedding = model.encode(article)

        similarity = util.cos_sim(claim_embedding, article_embedding)

        scores.append(float(similarity))

    probability = max(scores)

    return probability, articles


st.title("Shodh AI News Verifier")

claim = st.text_input("Enter a claim")

if st.button("Verify"):

    score, articles = verify_claim(claim)

    st.subheader("Related News")

    for a in articles:
        st.write(a)

    st.subheader("Truth Likelihood")

    st.progress(score)

    st.write("Score:", round(score,2))