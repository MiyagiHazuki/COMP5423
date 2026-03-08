import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from tqdm import tqdm
from utils import log_message, debug, info, warning, error
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# Ensure NLTK resources are only initialized once, with error handling
try:
    # Set NLTK data directory to project internal, avoiding permission issues
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Check if punkt resource has already been downloaded
    if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt")):
        info("Downloading NLTK punkt resource, this may take a while...")
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        info("NLTK punkt resource download completed")
    else:
        debug("NLTK punkt resource already exists")
except Exception as e:
    error(f"Error initializing NLTK: {str(e)}")
    # Try using simple tokenization method if initialization fails
    warning("Using alternative tokenization method")

class DocumentProcessor:
    @staticmethod
    def tokenize(text):
        try:
            return word_tokenize(text.lower())
        except Exception as e:
            warning(f"NLTK tokenization failed: {str(e)}, using alternative method")
            # Alternative tokenization method
            return text.lower().split()


class BaseSearchEngine:
    """The base class of search engine, containing common functions"""
    
    @staticmethod
    def load_documents(processed_path):
        """
        Load document data
        
        Parameters:
            processed_path: Path to processed document data
            
        Returns:
            doc_ids: List of document IDs
            docs: List of document content
        """
        doc_ids = []
        docs = []
        with open(processed_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                doc_ids.append(item["doc_id"])
                docs.append(item["text"])
        return doc_ids, docs
    
    @staticmethod
    def load_questions(val_path):
        """
        Load question data
        
        Parameters:
            val_path: Path to validation set
            
        Returns:
            questions: List of questions (each element is a dictionary containing question information)
        """
        questions = []
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                questions.append(item)  # Keep the whole question item
        return questions
    
    @staticmethod
    def load_bm25(model_path):
        """
        Load BM25 model
        
        Parameters:
            model_path: Path to BM25 model
            
        Returns:
            bm25: Loaded BM25 model
        """
        with open(model_path, 'rb') as f:
            bm25 = pickle.load(f)
        return bm25
    
    @staticmethod
    def load_tfidf(model_path):
        """
        Load TF-IDF model
        
        Parameters:
            model_path: Path to TF-IDF model
            
        Returns:
            tfidf_vectorizer: TF-IDF vectorizer
            tfidf_matrix: TF-IDF matrix, None if not generated
        """
        with open(model_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        return tfidf_vectorizer, None
    
    @staticmethod
    def normalize_scores(scores):
        """
        Normalize scores
        
        Parameters:
            scores: Original score array
            
        Returns:
            Normalized scores
        """
        min_s = np.min(scores)
        max_s = np.max(scores)
        return (scores - min_s) / (max_s - min_s + 1e-8)
    
    @staticmethod
    def extract_answer_from_doc(doc_text, max_words=2):
        """
        Extract answer from document
        
        Parameters:
            doc_text: Document text
            max_words: Maximum number of words to extract
            
        Returns:
            Extracted answer text
        """
        sentences = sent_tokenize(doc_text)
        if not sentences:
            return ""
        first_sentence = sentences[0]
        tokens = word_tokenize(first_sentence)
        return " ".join(tokens[:max_words])
    
    def predict_top_document(self, query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25, method="bm25", alpha=0.5, top_k=5, return_format="results"):
        """
        Predict the most relevant document to the query
        
        Parameters:
            query: Query string
            doc_ids: List of document IDs
            docs: List of document content
            tfidf_vectorizer: TF-IDF vectorizer
            tfidf_matrix: TF-IDF matrix, None if not generated
            bm25: BM25 model
            method: The method used (bm25, tfidf, hybrid)
            alpha: The weight of BM25 in hybrid mode
            top_k: The number of top results to return
            return_format: The format of the return value,可选值为:
                - "results": Return a list of tuples [(idx, doc_id, doc)]
                - "lists": Return two lists (doc_ids, scores)
            
        Returns:
            Depending on the return_format parameter, return different formats:
            - "results": The most relevant document list, each element is a tuple (idx, doc_id, doc)
            - "lists": Two lists (doc_ids, scores)
        """
        debug(f"Using {method} method to retrieve documents related to the query")
        tokens = DocumentProcessor.tokenize(query)

        if method == "bm25":
            scores = bm25.get_scores(tokens)

        elif method == "tfidf":
            query_vec = tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        elif method == "hybrid":
            bm25_scores = bm25.get_scores(tokens)
            query_vec = tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

            norm_bm25 = self.normalize_scores(bm25_scores)
            norm_tfidf = self.normalize_scores(tfidf_scores)

            scores = alpha * norm_bm25 + (1 - alpha) * norm_tfidf

        else:
            raise ValueError("Invalid method: choose 'bm25', 'tfidf', or 'hybrid'")

        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        if return_format == "results":
            results = [(idx, doc_ids[idx], docs[idx]) for idx in top_k_indices]
            return results
        elif return_format == "lists":
            top_doc_ids = [doc_ids[idx] for idx in top_k_indices]
            top_scores = [float(scores[idx]) for idx in top_k_indices]  # 转换为普通float，便于JSON序列化
            return top_doc_ids, top_scores
        else:
            raise ValueError("Invalid return_format: choose 'results' or 'lists'")


class ManualQuerySearch(BaseSearchEngine):
    """The class for manual query search, receiving webui parameters for search"""
    
    def __init__(self, config):
        """
        Initialize the query searcher
        
        Parameters:
            config: The configuration dictionary, should contain the following keys:
                - doc_path: The path to the processed documents
                - bm25_path: The path to the BM25 model
                - tfidf_path: The path to the TF-IDF model
                - method: The search method (bm25, tfidf, hybrid)
                - hybrid_alpha: The weight of BM25 in hybrid mode
                - top_k: The number of top results to return
                - max_words: The maximum number of words to extract as an answer
        """
        self.config = config
        self.method = config.get("method", "bm25")
        self.hybrid_alpha = config.get("hybrid_alpha", 0.7)
        self.top_k = config.get("top_k", 5)
        self.max_words = config.get("max_words", 2)
        
        try:
            # Load documents
            doc_path = config.get("doc_path")
            info(f"Loading documents: {doc_path}")
            self.doc_ids, self.docs = self.load_documents(doc_path)
            info(f"Successfully loaded {len(self.doc_ids)} documents")
            
            # Load BM25 model
            bm25_path = config.get("bm25_path")
            info(f"Loading BM25 model: {bm25_path}")
            self.bm25 = self.load_bm25(bm25_path)
            info("BM25 model loaded successfully")
            
            # Load TF-IDF model if needed
            if self.method in ["tfidf", "hybrid"]:
                tfidf_path = config.get("tfidf_path")
                info(f"Loading TF-IDF model: {tfidf_path}")
                self.tfidf_vectorizer, _ = self.load_tfidf(tfidf_path)
                    
                # Generate TF-IDF matrix on first query to avoid large memory usage
                info("TF-IDF vectorizer loaded successfully, will generate matrix on first query")
                self._tfidf_matrix = None
            else:
                self.tfidf_vectorizer = None
                self._tfidf_matrix = None
                
            info(f"ManualQuerySearch initialized successfully, using method: {self.method}")
        except Exception as e:
            error(f"Error initializing ManualQuerySearch: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @property
    def tfidf_matrix(self):
        """Delayed loading of TF-IDF matrix, only generated when needed"""
        if self._tfidf_matrix is None and self.tfidf_vectorizer is not None:
            info("Generating TF-IDF matrix, this may take a while...")
            self._tfidf_matrix = self.tfidf_vectorizer.transform(self.docs)
            info(f"TF-IDF matrix generated successfully, shape: {self._tfidf_matrix.shape}")
        return self._tfidf_matrix
    
    def search(self, query):
        """
        Using the initialized model to search the query
        
        Parameters:
            query: The query string
            
        Returns:
            A dictionary containing answer and document_id
        """
        debug(f"Processing query: '{query}'")
        
        try:
            # Perform document retrieval
            results = self.predict_top_document(
                query, self.doc_ids, self.docs,
                tfidf_vectorizer=self.tfidf_vectorizer,
                tfidf_matrix=self.tfidf_matrix,
                bm25=self.bm25,
                method=self.method,
                alpha=self.hybrid_alpha,
                top_k=self.top_k,
                return_format="results"
            )
            
            # Analyze results
            if results and len(results) > 0:
                # Extract document ID list and ensure all are strings
                doc_id_list = [str(doc_id) for (_, doc_id, _) in results]
                
                debug(f"Retrieval results: found {len(doc_id_list)} relevant documents")
                return {
                    "question": query,
                    "document_id": doc_id_list
                }
            else:
                # No relevant documents found
                debug("Retrieval results: no relevant documents found")
                return {
                    "question": query,
                    "document_id": []
                }
                
        except Exception as e:
            error(f"Error during search: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "question": query,
                "document_id": []
            }


class BatchEvaluator(BaseSearchEngine):
    """Batch evaluation class, used to evaluate the performance of retrieval methods on the validation set"""
    
    def __init__(self, config):
        """
        Initialize the batch evaluator
        
            Parameters:
            config: The configuration dictionary, should contain the following keys:
                - doc_path: The path to the processed documents
                - bm25_path: The path to the BM25 model
                - tfidf_path: The path to the TF-IDF model
                - method: The retrieval method (hybrid, bm25, tfidf)
                - hybrid_alpha: The weight of BM25 in hybrid mode
                - top_k: The number of top results to return
                - max_words: The maximum number of words to extract as an answer
        """
        super().__init__()
        self.method = config.get("method", "hybrid")  # 检索方法: hybrid/bm25/tfidf
        self.hybrid_alpha = config.get("hybrid_alpha", 0.7)  # hybrid方法中BM25的权重
        self.top_k = config.get("top_k", 5)  # 返回的文档数量
        self.max_words = config.get("max_words", 2)  # 截断文档标题的最大单词数
        
        self.doc_path = config.get("doc_path")  # 文档路径
        self.bm25_path = config.get("bm25_path")  # BM25模型路径
        self.tfidf_path = config.get("tfidf_path")  # TF-IDF模型路径
        self.val_path = config.get("val_path")  # 验证集路径
        self.output_path = config.get("output_path")  # 输出路径
        
        info(f"[BatchEvaluator] Initialized evaluator, method: {self.method}, hybrid ratio: {self.hybrid_alpha}, Top-K: {self.top_k}")
        info(f"[BatchEvaluator] Data paths: documents={self.doc_path}, validation set={self.val_path}")
        debug(f"[BatchEvaluator] Model paths: BM25={self.bm25_path}, TF-IDF={self.tfidf_path}")
        info(f"[BatchEvaluator] Output path: {self.output_path}")
    
    def evaluate(self):
        """
        Evaluate the retrieval method on the validation set

        Returns:
            The path of the output file
        """
        info(f"[BatchEvaluator] Starting evaluation process, method: {self.method}")
        
        # Load documents
        debug(f"[BatchEvaluator] Starting to load documents: {self.doc_path}")
        doc_ids, docs = self.load_documents(self.doc_path)
        info(f"[BatchEvaluator] Successfully loaded {len(doc_ids)} documents")
        
        # Load BM25 and TF-IDF models
        debug(f"[BatchEvaluator] Starting to load BM25 model: {self.bm25_path}")
        bm25 = self.load_bm25(self.bm25_path)
        debug(f"[BatchEvaluator] BM25 model loaded successfully")
        
        debug(f"[BatchEvaluator] Starting to load TF-IDF model: {self.tfidf_path}")
        tfidf_vectorizer, _ = self.load_tfidf(self.tfidf_path)
        # Generate TF-IDF matrix
        debug(f"[BatchEvaluator] Starting to generate TF-IDF matrix")
        tfidf_matrix = tfidf_vectorizer.transform(docs)
        debug(f"[BatchEvaluator] TF-IDF model loaded successfully, matrix shape: {tfidf_matrix.shape}")
        
        # Load validation set questions
        debug(f"[BatchEvaluator] Starting to load validation set questions: {self.val_path}")
        questions = self.load_questions(self.val_path)
        info(f"[BatchEvaluator] Successfully loaded {len(questions)} questions")
        
        # Store evaluation results
        info(f"[BatchEvaluator] Starting evaluation, total {len(questions)} questions")
        results = []
        empty_results = 0
        total_pred_docs = 0
        
        # Evaluate each question
        start_time = time.time()
        for i, question_item in enumerate(questions):
            # Print progress every 10%
            if (i+1) % max(1, len(questions)//10) == 0 or i+1 == len(questions):
                progress = (i+1) / len(questions) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i+1)) * (len(questions) - i - 1) if i > 0 else 0
                info(f"[BatchEvaluator] Progress: {progress:.1f}% ({i+1}/{len(questions)}), Elapsed time: {elapsed:.1f} seconds, Estimated remaining: {eta:.1f} seconds")
            
            # Get query text and reference document ID from question item
            query = question_item["question"]
            ref_doc_id = question_item.get("document_id", "")
            
            # Predict documents
            pred_doc_ids, pred_scores = self.predict_top_document(
                query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25,
                method=self.method, alpha=self.hybrid_alpha, top_k=self.top_k,
                return_format="lists"
            )
            
            if not pred_doc_ids:
                empty_results += 1
                warning(f"[BatchEvaluator] Warning: Question {i+1} not found matching documents: {query[:50]}...")
            
            total_pred_docs += len(pred_doc_ids)
            
            # Record results
            result = {
                "question": query,
                "ref_doc_id": ref_doc_id,
                "pred_doc_ids": pred_doc_ids,
                "pred_scores": pred_scores
            }
            results.append(result)
        
        # Calculate statistics
        avg_docs = total_pred_docs / len(questions) if questions else 0
        info(f"[BatchEvaluator] Evaluation completed, total questions: {len(questions)}, empty results: {empty_results}, average documents: {avg_docs:.2f}")
        
        # Save results
        debug(f"[BatchEvaluator] Starting to save evaluation results to: {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        info(f"[BatchEvaluator] Evaluation results saved to: {self.output_path}")
        
        return self.output_path