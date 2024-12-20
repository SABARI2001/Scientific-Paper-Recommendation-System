import os  # Import os to access environment variables
import streamlit as st
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from wordcloud import WordCloud
import numpy as np
import torch
import matplotlib.pyplot as plt
from together import Together  # Import the Together AI library
from sklearn.feature_extraction import text as sk_text
import re  # Import regex for reference extraction

class ScientificPaperRecommender:
    def __init__(self, api_key):
        # Initialize Together client with the API key
        self.client = Together(api_key=api_key)
        
        # Initialize SciBERT tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # Initialize stopwords for keyword filtering
        self.stop_words = set(sk_text.ENGLISH_STOP_WORDS)

    def fetch_arxiv_papers(self, query, max_results=10):
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(arxiv_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'xml')
            return [
                {"title": entry.title.text, "summary": entry.summary.text, "pdf_url": entry.id.text}
                for entry in soup.find_all('entry')
            ]
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from Arxiv: {e}")
            return []

    def cluster_papers(self, papers, method="tf-idf"):
        texts = [paper["summary"] for paper in papers]
        if method == "tf-idf":
            vectorizer = TfidfVectorizer(max_features=100)
            vectors = vectorizer.fit_transform(texts).toarray()
        else:
            vectors = np.array([self.get_embeddings(text) for text in texts])

        num_clusters = min(5, len(papers))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
        labels = kmeans.labels_

        clustered_papers = {}
        for idx, label in enumerate(labels):
            clustered_papers.setdefault(label, []).append(papers[idx])
        return clustered_papers

    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def extract_text_from_pdf(self, file):
        reader = PdfReader(file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text

    def summarize_text(self, text):
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=None,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True
        )
        
        summary = ""
        try:
            for token in response:
                if hasattr(token, 'choices') and token.choices:  # Check if choices exist and are not empty
                    summary += token.choices[0].delta.content
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
        
        return summary

    def answer_question(self, text, question):
        input_text = f"Context: {text}\nQuestion: {question}"
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=100,  # Adjust based on your needs
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=False
        )
    
        # Check if the response has choices and get the content directly
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]  # Get the first choice
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                answer = choice.message.content.strip()  # Access content correctly
                return answer
        return "Sorry, I couldn't find an answer."

    def generate_wordcloud(self, text):
        filtered_text = " ".join([word for word in text.split() if word.lower() not in self.stop_words])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

    def extract_references(self, text):
        # Simple regex to extract references (modify according to your needs)
        references = re.findall(r'\[(\d+)\]', text)  # Looks for references in the format [1], [2], etc.
        return references

    def fetch_cross_referenced_papers(self, references):
        # This is a placeholder function. You can replace it with a real lookup in a database or an API call.
        cross_referenced_papers = []
        for ref in references:
            paper_info = self.fetch_arxiv_papers(ref)  # Fetch papers based on reference (assumed to be queryable)
            cross_referenced_papers.extend(paper_info)
        return cross_referenced_papers

def main():
    st.sidebar.title("Scientific Paper Recommender System")
    
    # Directly assign your API key here
    api_key = "YOUR_API_KEY"  # Replace with your actual API key

    if api_key:
        recommender = ScientificPaperRecommender(api_key)  # Pass API key here
    else:
        st.warning("API Key is not set.")
        return  # Exit early if no API key is provided

    option = st.sidebar.selectbox("Choose an option", ("Paper Recommendation", "PDF Analysis", "PDF Q&A", "PDF Summarization"))

    if option == "Paper Recommendation":
        st.title("Paper Recommendation & Clustering")
        query = st.text_input("Enter your search query:")
        method = st.selectbox("Select Information Retrieval Method", ("tf-idf", "BERT"))

        if query:
            papers = recommender.fetch_arxiv_papers(query)
            if papers:
                clustered_papers = recommender.cluster_papers(papers, method=method)
                
                for cluster_id, paper_list in clustered_papers.items():
                    st.subheader(f"Cluster {cluster_id + 1}")
                    for paper in paper_list:
                        st.write(f"**{paper['title']}**")
                        st.write(paper['summary'])
                        st.write(f"[PDF Link]({paper['pdf_url']})")
            else:
                st.write("No papers found for the given query.")

    elif option == "PDF Analysis":
        st.title("Upload a PDF for Analysis")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            
            if text:
                st.subheader("PDF Text Analysis")
                recommender.generate_wordcloud(text)
                
                st.subheader("Keywords (Top 10)")
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                tfidf_matrix = vectorizer.fit_transform([text])
                keywords = vectorizer.get_feature_names_out()
                st.write(", ".join(keywords))
                
                # Cross-Referencing Section
                st.subheader("Cross-Referenced Papers")
                references = recommender.extract_references(text)
                if references:
                    cross_referenced_papers = recommender.fetch_cross_referenced_papers(references)
                    for paper in cross_referenced_papers:
                        st.write(f"**{paper['title']}**")
                        st.write(paper['summary'])
                        st.write(f"[PDF Link]({paper['pdf_url']})")
                else:
                    st.write("No references found in the text.")

    elif option == "PDF Q&A":
        st.title("Question & Answer from PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        question = st.text_input("Enter your question:")

        if uploaded_file and question:
            text = recommender.extract_text_from_pdf(uploaded_file)
            answer = recommender.answer_question(text, question)
            st.write(f"**Answer:** {answer}")

    elif option == "PDF Summarization":
        st.title("Summarization of PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            summary = recommender.summarize_text(text)
            st.write(f"**Summary:** {summary}")

if __name__ == "__main__":
    main()
