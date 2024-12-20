import streamlit as st
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from wordcloud import WordCloud

class ScientificPaperRecommender:
    def __init__(self):
        # Initialize tokenizer and model for BERT embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    def fetch_arxiv_papers(self, query, max_results=10):
        """Fetch papers from Arxiv based on a query and return structured info for clustering."""
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(arxiv_url, headers=headers)
        soup = BeautifulSoup(response.text, 'xml')
        
        papers = []
        for entry in soup.find_all('entry'):
            papers.append({
                "title": entry.title.text,
                "summary": entry.summary.text,
                "pdf_url": entry.id.text
            })
        return papers

    def cluster_papers(self, papers, method="tf-idf"):
        """Cluster papers based on TF-IDF or BERT embeddings."""
        texts = [paper["summary"] for paper in papers]
        
        if method == "tf-idf":
            vectorizer = TfidfVectorizer(max_features=100)
            vectors = vectorizer.fit_transform(texts).toarray()
        else:
            vectors = np.array([self.get_embeddings(text) for text in texts])
        
        # Perform clustering
        num_clusters = min(5, len(papers))  # Adjust based on paper count
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        labels = kmeans.labels_
        
        clustered_papers = {}
        for idx, label in enumerate(labels):
            clustered_papers.setdefault(label, []).append(papers[idx])
        return clustered_papers

    def get_embeddings(self, text):
        """Generate embeddings for a given text using BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def extract_text_from_pdf(self, file):
        """Extracts text from uploaded PDF."""
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text

    def answer_question(self, text, question):
        """Answer question based on PDF text."""
        sentences = text.split(". ")
        question_embedding = self.get_embeddings(question)
        similarities = [cosine_similarity([question_embedding], [self.get_embeddings(s)])[0][0] for s in sentences]
        best_answer = sentences[np.argmax(similarities)]
        return best_answer

    def generate_wordcloud(self, text):
        """Generate word cloud for extracted PDF text."""
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

def main():
    st.sidebar.title("Scientific Paper Recommender System")
    
    recommender = ScientificPaperRecommender()
    option = st.sidebar.selectbox("Choose an option", ("Paper Recommendation", "PDF Analysis & Q&A"))

    if option == "Paper Recommendation":
        st.title("Paper Recommendation & Clustering")
        query = st.text_input("Enter your search query:")
        method = st.selectbox("Select Information Retrieval Method", ("tf-idf", "BERT"))

        if query:
            papers = recommender.fetch_arxiv_papers(query)
            clustered_papers = recommender.cluster_papers(papers, method=method)
            
            for cluster_id, paper_list in clustered_papers.items():
                st.subheader(f"Cluster {cluster_id + 1}")
                for paper in paper_list:
                    st.write(f"**{paper['title']}**")
                    st.write(paper['summary'])
                    st.write(f"[PDF Link]({paper['pdf_url']})")

    elif option == "PDF Analysis & Q&A":
        st.title("Upload a PDF for Analysis & Q&A")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        question = st.text_input("Enter your question:")

        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            if question:
                answer = recommender.answer_question(text, question)
                st.write("Answer:", answer)
            
            # PDF Analysis - Word Cloud
            st.subheader("PDF Text Analysis")
            recommender.generate_wordcloud(text)
            st.subheader("Keywords")
            st.write(pd.Series(text.split()).value_counts().head(10))

if __name__ == "__main__":
    main()
