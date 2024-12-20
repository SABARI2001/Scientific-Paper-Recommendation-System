import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from wordcloud import WordCloud
import numpy as np
import torch
import matplotlib.pyplot as plt
from together import Together
from sklearn.feature_extraction import text as sk_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from sklearn.preprocessing import StandardScaler
from rouge import Rouge  # Import the Rouge library for summarization evaluation
import networkx as nx  # Added for PageRank

class ScientificPaperRecommender:
    def __init__(self, api_key):
        self.client = Together(api_key=api_key)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.stop_words = set(sk_text.ENGLISH_STOP_WORDS)
        self.classifier = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def fetch_arxiv_papers(self, query, max_results=10):
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # Retry mechanism
        for attempt in range(3):
            try:
                response = requests.get(arxiv_url, headers=headers, timeout=30)  # Increased timeout
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'xml')
                return [
                    {"title": entry.title.text, "summary": entry.summary.text, "pdf_url": entry.id.text}
                    for entry in soup.find_all('entry')
                ]
            except requests.exceptions.RequestException as e:
                if attempt < 2:  # Retry twice
                    time.sleep(2)  # Wait before retrying
                else:
                    st.error(f"Error fetching data from Arxiv: {e}")
                    return []
    
    
    def cluster_papers(self, papers, retrieval_method="tf-idf", clustering_method="K-Means"):
        texts = [paper["summary"] for paper in papers]
        start_time = time.time()  # Start timing retrieval process
        
        # Vectorization based on the selected retrieval method
        if retrieval_method == "tf-idf":
            vectorizer = TfidfVectorizer(max_features=100)
            vectors = vectorizer.fit_transform(texts).toarray()
        elif retrieval_method == "BERT":
            vectors = np.array([self.get_embeddings(text) for text in texts])
        elif retrieval_method == "Probabilistic":
            # Simulated labels for realistic evaluation
            labels = [0 if i % 2 == 0 else 1 for i in range(len(texts))]
            X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            self.classifier.fit(X_train, y_train)
            predictions = self.classifier.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, predictions)
            st.write(f"Probabilistic Model Accuracy: {accuracy}")
            vectors = TfidfVectorizer().fit_transform(texts).toarray()
            
        retrieval_time = time.time() - start_time  # Calculate elapsed time
        st.write(f"Time taken for {retrieval_method} retrieval: {retrieval_time:.2f} seconds")
        
        # Scale vectors for DBSCAN
        vectors = StandardScaler().fit_transform(vectors)  # Standardize the vectors
        
        # Clustering based on the selected clustering method
        if clustering_method == "K-Means":
            num_clusters = min(5, len(papers))
            clustering = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
            labels = clustering.labels_
        elif clustering_method == "DBSCAN":
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors)  # Adjust eps and min_samples as needed
            labels = clustering.labels_
        elif clustering_method == "Agglomerative":
            num_clusters = min(5, len(papers))  # Adjust the number of clusters if necessary
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(vectors)
        elif clustering_method == "Affinity Propagation":
            clustering = AffinityPropagation().fit(vectors)  # Implement Affinity Propagation clustering
            labels = clustering.labels_
        
        # Evaluate clustering results
        self.evaluate_clustering(labels)
        
        # Create a dictionary of clustered papers
        clustered_papers = {}
        for idx, label in enumerate(labels):
            if label not in clustered_papers:
                clustered_papers[label] = []
            clustered_papers[label].append(papers[idx])  # Append the original paper dict
            
        return clustered_papers

        
        
    

    def evaluate_clustering(self, labels):
        if len(self.ground_truth_labels) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(self.ground_truth_labels, labels, average='weighted')
            
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")

        
            

        
    def get_vectors(self, clustered_papers):
        # Make sure you handle the structure of clustered_papers correctly
        if isinstance(clustered_papers, dict):  # If clustered_papers is a dictionary
            texts = [paper["summary"] for cluster in clustered_papers.values() for paper in cluster]
        else:  # If clustered_papers is a list or array
            texts = [paper["summary"] for paper in clustered_papers]
            # Convert texts to vectors (e.g., using a vectorizer)
            
            
        vectors = self.vectorizer.transform(texts).toarray()  # or however you obtain vectors
        return vectors


    def get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def create_clustered_papers_dict(self, labels, papers):
        clustered_papers = {}
        for idx, label in enumerate(labels):
            clustered_papers.setdefault(label, []).append(papers[idx])
        return clustered_papers

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
                if hasattr(token, 'choices') and token.choices:
                    summary += token.choices[0].delta.content
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
        return summary

    def answer_question(self, text, question):
        input_text = f"Context: {text}\nQuestion: {question}"
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=100,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"],
            stream=False
        )
        if response and hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content.strip()
        return "Sorry, I couldn't find an answer."

    def generate_wordcloud(self, text):
        filtered_text = " ".join([word for word in text.split() if word.lower() not in self.stop_words])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)
    
    def rank_with_pagerank(self, papers):
        graph = nx.Graph()
        for paper in papers:
            graph.add_node(paper["title"])
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                if cosine_similarity([self.get_embeddings(papers[i]["summary"])], [self.get_embeddings(papers[j]["summary"])])[0][0] > 0.5:
                    graph.add_edge(papers[i]["title"], papers[j]["title"])
        pagerank_scores = nx.pagerank(graph)
        return sorted(papers, key=lambda x: pagerank_scores.get(x["title"], 0), reverse=True)
    
    def train_probabilistic_model(self, papers):
        texts = [paper["summary"] for paper in papers]
        labels = [1] * len(texts)  # Assuming all are positive for demo; replace with actual labels if available
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        predictions = self.classifier.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy}")

    # Other methods remain the same as in your code...

def main():
    st.sidebar.title("Scientific Paper Recommender System")
    api_key = "2e70a6712033ad8a9f89c0e3d1028565300bb31c7986767ee87928a9badd71c0"
    if api_key:
        recommender = ScientificPaperRecommender(api_key)
    else:
        st.warning("API Key is not set.")
        return

    option = st.sidebar.selectbox("Choose an option", ("Paper Recommendation", "PDF Analysis", "PDF Q&A", "PDF Summarization"))

    if option == "Paper Recommendation":
        st.title("Paper Recommendation & Clustering")
        query = st.text_input("Enter your search query:")
        retrieval_method = st.selectbox("Select Information Retrieval Method", ("tf-idf", "BERT", "Probabilistic"))
        clustering_method = st.selectbox("Select Clustering Algorithm", ("K-Means", "DBSCAN", "Agglomerative", "Affinity Propagation"))
        cluster_button = st.button("Cluster Papers")

        if query:
            # Measure the time taken to fetch papers
            start_time = time.time()
            papers = recommender.fetch_arxiv_papers(query)
            end_time = time.time()
            st.write(f"Time taken to fetch papers: {end_time - start_time:.2f} seconds")

            if papers:
                # Display papers as they are before clustering
                st.subheader("Fetched Papers ")
                for idx, paper in enumerate(papers, 1):  # Displaying with serial numbering
                    st.markdown(f"{idx}. [{paper['title']}]({paper['pdf_url']})")

                if cluster_button:
                    # Assign dummy labels for evaluation; replace with actual labels if available
                    recommender.ground_truth_labels = [0] * len(papers)  # Dummy labels (for demonstration)
                    clustered_papers = recommender.cluster_papers(papers, retrieval_method=retrieval_method, clustering_method=clustering_method)
                    
                    # Evaluation metrics displayed
                    recommender.evaluate_clustering(recommender.ground_truth_labels)  # Pass the labels to evaluate
                    
                    for cluster_id, paper_list in clustered_papers.items():
                        st.subheader(f"Cluster {cluster_id}")
                        for idx, paper in enumerate(paper_list, 1):  # Serial numbering within clusters
                            st.markdown(f"{idx}. [{paper['title']}]({paper['pdf_url']})")

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

    elif option == "PDF Q&A":
        st.title("PDF Question & Answer")
        uploaded_file = st.file_uploader("Upload a PDF file for Q&A", type="pdf")
        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            question = st.text_input("Ask a question about the PDF:")
            if st.button("Get Answer"):
                answer = recommender.answer_question(text, question)
                st.write(f"**Answer:** {answer}")

    elif option == "PDF Summarization":
        st.title("PDF Summarization")
        uploaded_file = st.file_uploader("Upload a PDF file for summarization", type="pdf")
        if uploaded_file:
            text = recommender.extract_text_from_pdf(uploaded_file)
            if st.button("Summarize"):
                summary = recommender.summarize_text(text)
                st.write(f"**Summary:** {summary}")

if __name__ == "__main__":
    main()

    