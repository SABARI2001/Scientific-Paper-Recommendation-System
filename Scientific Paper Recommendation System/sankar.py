import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForQuestionAnswering
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from rake_nltk import Rake
import time
from collections import Counter
import math
from typing import List, Dict

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class QABot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        # Initialize T5 model for neural IR
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForQuestionAnswering.from_pretrained('t5-small')
        self.rake = Rake()

    def fetch_content(self, url):
        """Fetch and clean content from the given URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            return f"Error fetching content: {str(e)}"

    def preprocess_text(self, text):
        """Split text into sentences"""
        return sent_tokenize(text)

    def method1_tfidf(self, sentences, question):
        """TF-IDF based retrieval"""
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences + [question])
            similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            return similarities[0]
        except:
            return np.zeros(len(sentences))

    def get_bert_embeddings(self, text):
        """Get BERT embeddings for a text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def method2_bert(self, sentences, question):
        """BERT-based semantic similarity"""
        question_embedding = self.get_bert_embeddings(question)
        similarities = []
        for sentence in sentences:
            sent_embedding = self.get_bert_embeddings(sentence)
            similarity = cosine_similarity([question_embedding], [sent_embedding])[0][0]
            similarities.append(similarity)
        return np.array(similarities)

    def method3_keyword(self, sentences, question):
        """Keyword-based retrieval using RAKE"""
        self.rake.extract_keywords_from_text(question)
        question_keywords = set(self.rake.get_ranked_phrases())
        similarities = []
        
        for sentence in sentences:
            self.rake.extract_keywords_from_text(sentence)
            sentence_keywords = set(self.rake.get_ranked_phrases())
            if len(question_keywords) == 0 or len(sentence_keywords) == 0:
                similarities.append(0)
            else:
                similarity = len(question_keywords.intersection(sentence_keywords)) / len(question_keywords)
                similarities.append(similarity)
        
        return np.array(similarities)

    def method4_probabilistic(self, sentences: List[str], question: str) -> np.ndarray:
        """
        Probabilistic retrieval using BM25 scoring
        """
        # Tokenize documents and query
        doc_tokens = [word_tokenize(sent.lower()) for sent in sentences]
        query_tokens = word_tokenize(question.lower())
        
        # Calculate document lengths and average document length
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        
        # Calculate IDF for query terms
        N = len(sentences)
        k1 = 1.5
        b = 0.75
        
        # Count documents containing each query term
        doc_freq = {}
        for term in query_tokens:
            doc_freq[term] = sum(1 for doc in doc_tokens if term in doc)
        
        # Calculate BM25 scores
        scores = []
        for i, doc_toks in enumerate(doc_tokens):
            score = 0
            doc_len = doc_lengths[i]
            
            # Count term frequencies in document
            term_freq = Counter(doc_toks)
            
            for term in query_tokens:
                if term in doc_freq and doc_freq[term] > 0:
                    idf = math.log((N - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                    tf = term_freq.get(term, 0)
                    
                    # BM25 score calculation
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_length)
                    score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return np.array(scores)

    def method5_neural_ir(self, sentences: List[str], question: str) -> np.ndarray:
        """
        Neural Information Retrieval using T5 model
        """
        max_length = 512
        scores = []
        
        # Prepare question prefix
        question_prefix = "question: " + question + " context: "
        
        with torch.no_grad():
            for sentence in sentences:
                # Prepare input
                input_text = question_prefix + sentence
                inputs = self.t5_tokenizer(input_text, 
                                         return_tensors="pt",
                                         max_length=max_length,
                                         truncation=True)
                
                # Get model outputs
                outputs = self.t5_model(**inputs)
                
                # Calculate relevance score using start/end logits
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
                # Compute attention score as the max of start/end logits
                attention_score = torch.max(
                    F.softmax(start_logits, dim=1).max(),
                    F.softmax(end_logits, dim=1).max()
                ).item()
                
                scores.append(attention_score)
        
        return np.array(scores)
    
    def evaluate_answer(self, answer, question, relevant_sentences):
        """Calculate evaluation metrics for an answer"""
        # 1. Response Time
        start_time = time.time()
        _ = self.get_bert_embeddings(answer)
        response_time = time.time() - start_time

        # 2. Semantic Relevance Score
        answer_embedding = self.get_bert_embeddings(answer)
        question_embedding = self.get_bert_embeddings(question)
        semantic_score = cosine_similarity([answer_embedding], [question_embedding])[0][0]

        # 3. Precision and Recall
        answer_words = set(word_tokenize(answer.lower()))
        relevant_words = set(word_tokenize(' '.join(relevant_sentences).lower()))
        
        if len(answer_words) == 0:
            precision = 0
        else:
            precision = len(answer_words.intersection(relevant_words)) / len(answer_words)
        
        if len(relevant_words) == 0:
            recall = 0
        else:
            recall = len(answer_words.intersection(relevant_words)) / len(relevant_words)

        # 4. Calculate F1 Score
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "response_time": round(response_time, 3),
            "semantic_relevance": round(semantic_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1_score, 3)
        }


    def ensemble_retrieve(self, sentences: List[str], question: str) -> Dict:
        """
        Ensemble method combining all retrieval approaches
        """
        # Get scores from all methods
        scores = {
            "tfidf": self.method1_tfidf(sentences, question),
            "bert": self.method2_bert(sentences, question),
            "keyword": self.method3_keyword(sentences, question),
            "probabilistic": self.method4_probabilistic(sentences, question),
            "neural_ir": self.method5_neural_ir(sentences, question)
        }
        
        # Normalize scores
        for method in scores:
            if np.sum(scores[method]) != 0:
                scores[method] = scores[method] / np.sum(scores[method])
        
        # Combine scores with weights
        weights = {
            "tfidf": 0.15,
            "bert": 0.25,
            "keyword": 0.15,
            "probabilistic": 0.20,
            "neural_ir": 0.25
        }
        
        final_scores = np.zeros(len(sentences))
        for method, score in scores.items():
            final_scores += weights[method] * score
        
        # Get top answer
        top_idx = final_scores.argmax()
        
        return {
            "answer": sentences[top_idx],
            "score": final_scores[top_idx],
            "method_scores": {method: scores[method][top_idx] for method in scores}
        }

class WebClusterer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_page_embedding(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            return self.get_bert_embedding(text)
        except:
            return None

    def cluster_urls(self, urls, n_clusters=3):
        # Get embeddings for all URLs
        embeddings = []
        valid_urls = []
        
        for url in urls:
            embedding = self.get_page_embedding(url)
            if embedding is not None:
                embeddings.append(embedding)
                valid_urls.append(url)
        
        if len(embeddings) < n_clusters:
            return None, None
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate similarity scores within clusters
        cluster_similarities = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_embeddings = np.array(embeddings)[cluster_mask]
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                avg_similarity = (similarities.sum() - similarities.shape[0]) / (similarities.shape[0] * (similarities.shape[0] - 1))
                cluster_similarities[i] = avg_similarity
            else:
                cluster_similarities[i] = 1.0
        
        return dict(zip(valid_urls, clusters)), cluster_similarities

def qa_page():
    st.title("Travel Blog Enhancer")
    
    qa_bot = QABot()
    
    url = st.text_input("Enter webpage URL:")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and url and question:
        with st.spinner("Fetching and processing content..."):
            content = qa_bot.fetch_content(url)
            sentences = qa_bot.preprocess_text(content)
            
            if len(sentences) == 0:
                st.error("No content could be extracted from the URL.")
                return

            methods = {
                "TF-IDF": qa_bot.method1_tfidf,
                "BERT": qa_bot.method2_bert,
                "Keyword": qa_bot.method3_keyword,
                "Probabilistic": qa_bot.method4_probabilistic,
                "Neural IR": qa_bot.method5_neural_ir
            }

            # Get ensemble results
            ensemble_result = qa_bot.ensemble_retrieve(sentences, question)
            
            # Display individual method results
            results = []
            relevant_sentences = sorted(sentences, 
                                     key=lambda x: qa_bot.method2_bert([x], question)[0], 
                                     reverse=True)[:3]
            
            for method_name, method_func in methods.items():
                similarities = method_func(sentences, question)
                top_idx = similarities.argmax()
                answer = sentences[top_idx]
                
                metrics = qa_bot.evaluate_answer(answer, question, relevant_sentences)
                results.append({
                    "method": method_name,
                    "answer": answer,
                    "metrics": metrics
                })

            # Display ensemble results first
            st.subheader("Ensemble Results:")
            with st.expander("Ensemble Answer (Combined Methods)", expanded=True):
                st.write("Answer:", ensemble_result["answer"])
                st.write("\nEnsemble Score:", round(ensemble_result["score"], 3))
                st.write("\nIndividual Method Contributions:")
                for method, score in ensemble_result["method_scores"].items():
                    st.write(f"- {method}: {round(score, 3)}")

            # Display individual method results
            st.subheader("Individual Method Results:")
            for result in results:
                with st.expander(f"Answer using {result['method']} method"):
                    st.write("Answer:", result['answer'])
                    st.write("\nMetrics:")
                    metrics = result['metrics']
                    st.write(f"- Response Time: {metrics['response_time']} seconds")
                    st.write(f"- Semantic Relevance: {metrics['semantic_relevance']}")
                    st.write(f"- Precision: {metrics['precision']}")
                    st.write(f"- Recall: {metrics['recall']}")

def clustering_page():
    st.title("Web Page Clustering")
    
    clusterer = WebClusterer()
    
    # Input for multiple URLs
    urls_text = st.text_area("Enter URLs (one per line):")
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=5, value=3)
    
    if st.button("Cluster URLs") and urls_text:
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if len(urls) < n_clusters:
            st.error(f"Please provide at least {n_clusters} valid URLs")
            return
            
        with st.spinner("Clustering URLs..."):
            clusters, similarities = clusterer.cluster_urls(urls, n_clusters)
            
            if clusters is None:
                st.error("Error clustering URLs. Please check if all URLs are accessible.")
                return
            
            # Display clusters
            st.subheader("Clustering Results:")
            
            # Group URLs by cluster
            cluster_groups = {}
            for url, cluster in clusters.items():
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(url)
            
            # Display each cluster with its URLs and similarity score
            for cluster_id in sorted(cluster_groups.keys()):
                with st.expander(f"Cluster {cluster_id + 1}"):
                    st.write(f"Average Similarity Score: {similarities[cluster_id]:.3f}")
                    st.write("URLs in this cluster:")
                    for url in cluster_groups[cluster_id]:
                        st.write(f"- {url}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ["Question Answering", "URL Clustering"])
    
    if page == "Question Answering":
        qa_page()
    else:
        clustering_page()

if __name__ == "__main__":
    main()