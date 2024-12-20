import streamlit as st
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
import torch
from wordcloud import WordCloud
from sklearn.feature_extraction import text as sk_text

class ScientificPaperRecommender:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.qa_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.qa_model = T5ForConditionalGeneration.from_pretrained("t5-base")
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

    def answer_question(self, text, question):
        sentences = text.split(". ")
        chunk_size = 450
        chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        
        best_answer = ""
        for chunk in chunks:
            input_text = f"question: {question} context: {chunk}"
            inputs = self.qa_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
            outputs = self.qa_model.generate(inputs, max_length=50)
            answer = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if answer:
                best_answer = answer
                break
        
        return best_answer if best_answer else "Sorry, I couldn't find an answer."

    def generate_wordcloud(self, text):
        filtered_text = " ".join([word for word in text.split() if word.lower() not in self.stop_words])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

def main():
    st.sidebar.title("Scientific Paper Recommender System")
    recommender = ScientificPaperRecommender()
    
    # Separate menu options
    option = st.sidebar.selectbox("Choose an option", ("Paper Recommendation", "PDF Analysis", "Q&A"))

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
            else:
                st.write("Unable to extract text from the uploaded PDF file.")
        else:
            st.write("Please upload a PDF file for analysis.")

    elif option == "Q&A":
        st.title("Ask a Question About the PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        question = st.text_input("Enter your question:")

        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            
            if text:
                if question:
                    answer = recommender.answer_question(text, question)
                    st.write("Answer:")
                    st.write(answer)
                else:
                    st.write("Please enter a question.")
            else:
                st.write("Unable to extract text from the uploaded PDF file.")
        else:
            st.write("Please upload a PDF file for analysis.")

if __name__ == "__main__":
    main()
