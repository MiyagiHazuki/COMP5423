import json
import re
import random
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle
from bs4 import BeautifulSoup
import html2text
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import logging

# Ensure nltk data is downloaded
try:
    nltk.download('punkt', quiet=True)
except:
    logging.warning("Unable to download NLTK punkt data, please ensure it's manually installed")

class DocumentProcessor:
    """Base class for document processing, containing common text processing methods"""
    
    @staticmethod
    def clean_wikipedia_text(text):
        """Text cleaning"""
        # First remove line breaks
        text = re.sub(r'\n', ' ', text)
        # Remove Contents (hide) section and surrounding spaces
        text = re.sub(r'\s*Contents \( hide \)\s*', ' ', text)
        # Remove (edit) marks and surrounding spaces
        text = re.sub(r'\s*\(edit\)\s*', ' ', text)
        # Specifically handle (edit) marks with spaces inside
        text = re.sub(r'\(\sedit\s\)', '', text)  # Match ( edit )
        text = re.sub(r'\(\sedit\)', '', text)    # Match (edit )
        text = re.sub(r'\(edit\s\)', '', text)    # Match ( edit)
       
        # Remove other Wikipedia-specific formats
        text = re.sub(r'Jump to : navigation , search', '', text)
        text = re.sub(r'Categories :.*', '', text)
        text = re.sub(r'Hidden categories :.*', '', text)
        text = re.sub(r'Edit links', '', text)
        text = re.sub(r'Retrieved from .*', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'See also .*', '', text, flags=re.DOTALL)
        text = re.sub(r'References .*', '', text, flags=re.DOTALL)
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()

    @staticmethod
    def tokenize(text):
        """Basic tokenization processing"""
        return word_tokenize(text.lower())


class PlainTextProcessor(DocumentProcessor):
    """Class for processing plain text"""
    
    @classmethod
    def clean_html_to_plaintext(cls, html):
        """Process into plain text"""
        soup = BeautifulSoup(html, 'html.parser')
        # Remove non-content elements
        for script in soup(["script", "style", "table", "ul", "ol", "footer", "nav"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Apply Wikipedia-specific cleaning
        text = cls.clean_wikipedia_text(text)
        return text


class MarkdownProcessor(DocumentProcessor):
    """Class for processing Markdown text"""
    
    @classmethod
    def html_to_markdown(cls, html):
        """Convert to Markdown format"""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.bypass_tables = False
        h.single_line_break = True
        markdown = h.handle(html)
        # Apply Wikipedia-specific cleaning
        markdown = cls.clean_wikipedia_text(markdown)
        return markdown


class DocumentIndexer:
    """Document indexing class for building search models"""
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
    
    def calculate_tfidf(self):
        """Calculate TF-IDF model"""
        logging.info("Starting to calculate TF-IDF model...")
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        logging.info("TF-IDF model calculation completed")
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """Calculate BM25 model"""
        logging.info("Starting to calculate BM25 model...")
        
        # Use tqdm to wrap tokenization process
        tokenized_corpus = []
        for doc in tqdm(self.corpus, desc="Tokenizing", unit="doc"):
            tokenized_corpus.append(DocumentProcessor.tokenize(doc))
        
        logging.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logging.info("BM25 index building completed")
        return self.bm25
    
    def save_models(self, output_prefix):
        """Save models to files"""
        if self.tfidf_vectorizer:
            logging.info(f"Saving TF-IDF model to {output_prefix}_tfidf.pkl...")
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            logging.info(f"TF-IDF model saved")
        if self.bm25:
            logging.info(f"Saving BM25 model to {output_prefix}_bm25.pkl...")
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
            logging.info(f"BM25 model saved")


class DocumentPipeline:
    """Document processing pipeline, integrating the entire workflow"""
    
    def __init__(self, input_path, model_output_dir, data_output_dir, config=None):
        self.input_path = input_path
        # Ensure directory paths end with path separator
        self.model_output_dir = os.path.join(model_output_dir, '') if not model_output_dir.endswith(os.path.sep) else model_output_dir
        self.data_output_dir = os.path.join(data_output_dir, '') if not data_output_dir.endswith(os.path.sep) else data_output_dir
        self.processed_plain = []
        self.processed_markdown = []
        self.config = config
        
    def process_documents(self):
        """Process raw data"""
        # Calculate total lines first to determine progress
        with open(self.input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)                                                                 
        
        logging.info(f"Starting to process documents, total {total_lines} lines...")
        
        # Reopen the file for processing
        with open(self.input_path, 'r', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for line in tqdm(f, total=total_lines, desc="Processing documents", unit="doc"):
                try:
                    doc = json.loads(line)
                    plain_text = PlainTextProcessor.clean_html_to_plaintext(doc['document_text'])
                    markdown_text = MarkdownProcessor.html_to_markdown(doc['document_text'])
                    
                    self.processed_plain.append({
                        "doc_id": doc['document_id'],
                        "text": plain_text
                    })
                    
                    self.processed_markdown.append({
                        "doc_id": doc['document_id'],
                        "text": markdown_text
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON, line content: {line[:100]}... Error: {e}")
                except KeyError as e:
                    logging.error(f"Missing key field in document: {e}")
    
    def save_processed_data(self):
        """Save processed data"""
        plain_output = os.path.join(self.data_output_dir, "processed_plain.jsonl")
        markdown_output = os.path.join(self.data_output_dir, "processed_markdown.jsonl")
        
        logging.info(f"Saving processed plain text data to {plain_output}...")
        with open(plain_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_plain, desc="Saving plain text data", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"Saving processed markdown data to {markdown_output}...")
        with open(markdown_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_markdown, desc="Saving markdown data", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"Plain text data saved to {plain_output}")
        logging.info(f"Markdown data saved to {markdown_output}")
    
    def build_models(self):
        """Build search models"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        logging.info("Starting to build models for plain text...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(os.path.join(self.model_output_dir, "plain"))
        
        logging.info("Starting to build models for markdown...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(os.path.join(self.model_output_dir, "markdown"))
        
        logging.info("All models successfully built and saved")
    
    def run(self):
        """Run the entire processing workflow"""
        self.process_documents()
        self.save_processed_data()
        self.build_models()


class DPRDataProcessor:
    """
    Process data for DPR model training format
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        # Create cache directory
        self.cache_dir = os.path.join("cache", "bert_base_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.documents = {}
        self.document_chunks = {}
        self.train_data = []
        self.val_data = []
        self.tfidf_vectorizer = None
        self.chunk_vectors = None
        self.chunk_list = []
        self.chunk_id_to_idx = {}
        
    def load_documents(self) -> None:
        """Load all documents"""
        doc_file = os.path.join(self.data_dir, "processed_plain.jsonl")
        logging.info(f"Loading document data: {doc_file}")
        
        with open(doc_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc = json.loads(line)
                self.documents[doc["doc_id"]] = doc["text"]
        
        logging.info(f"Successfully loaded {len(self.documents)} documents")
    
    def load_qa_data(self) -> None:
        """Load training and validation data"""
        # Modify path, load data from original_data directory
        original_data_dir = os.path.join(os.path.dirname(self.data_dir), "original_data")
        train_file = os.path.join(original_data_dir, "train.jsonl")
        val_file = os.path.join(original_data_dir, "val.jsonl")
        
        logging.info(f"Loading training and validation data from {original_data_dir}")
        
        # Load training data
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.train_data.append(json.loads(line))
        
        # Load validation data
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.val_data.append(json.loads(line))
                
        logging.info(f"Successfully loaded {len(self.train_data)} training data and {len(self.val_data)} validation data")
    
    def chunk_documents(self, chunk_size: int = 2048, overlap: int = 100) -> None:
        """
        Split documents into overlapping blocks, utilizing the long sequence ability of nomic-bert-2048
        
        Args:
            chunk_size: Approximate number of words per block, can be set larger
            overlap: Number of words in overlap between blocks
        """
        logging.info("Splitting documents into blocks...")
        
        # Check cache - now using cache directory
        cache_file = os.path.join(self.cache_dir, f"chunks_cache_{chunk_size}_{overlap}.pkl")
        if os.path.exists(cache_file):
            logging.info(f"Loading document blocks from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.document_chunks = cache_data['document_chunks']
                self.chunk_list = cache_data['chunk_list']
                for idx, chunk in enumerate(self.chunk_list):
                    self.chunk_id_to_idx[(chunk['doc_id'], chunk['chunk_id'])] = idx
            total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
            logging.info(f"Successfully loaded {total_chunks} blocks from cache")
            return
            
        for doc_id, text in tqdm(self.documents.items()):
            words = text.split()
            chunks = []
            
            # If document is not very long, can directly use the entire document
            if len(words) <= chunk_size:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": 0,
                    "text": text,
                    "start_idx": 0,
                    "end_idx": len(words)
                })
            else:
                # For very long documents, still need to split into blocks, but blocks can be larger
                start = 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": len(chunks),
                        "text": chunk_text,
                        "start_idx": start,
                        "end_idx": end
                    })
                    
                    start += chunk_size - overlap
            
            self.document_chunks[doc_id] = chunks
            
            # Maintain a list for subsequent processing
            for chunk in chunks:
                self.chunk_list.append(chunk)
                self.chunk_id_to_idx[(chunk['doc_id'], chunk['chunk_id'])] = len(self.chunk_list) - 1
        
        total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
        logging.info(f"Successfully split documents into {total_chunks} blocks")
        
        # Save cache - now using cache directory
        logging.info(f"Saving document blocks to cache: {cache_file}")
        cache_data = {
            'document_chunks': self.document_chunks,
            'chunk_list': self.chunk_list
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def build_tfidf_index(self):
        """Build TF-IDF index"""
        logging.info("Building TF-IDF index...")
        
        # Check cache - now using cache directory
        cache_file = os.path.join(self.cache_dir, "tfidf_cache.pkl")
        if os.path.exists(cache_file):
            logging.info(f"Loading TF-IDF index from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.tfidf_vectorizer = cache_data['vectorizer']
                self.chunk_vectors = cache_data['chunk_vectors']
            logging.info("TF-IDF index loading completed")
            return
        
        # Collect text from all blocks
        corpus = [chunk["text"] for chunk in self.chunk_list]
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Convert all document blocks to TF-IDF vectors
        start_time = time.time()
        self.chunk_vectors = self.tfidf_vectorizer.fit_transform(corpus)
        elapsed = time.time() - start_time
        logging.info(f"TF-IDF index building completed, time taken {elapsed:.2f} seconds")
        
        # Save cache - now using cache directory
        logging.info(f"Saving TF-IDF index to cache: {cache_file}")
        cache_data = {
            'vectorizer': self.tfidf_vectorizer,
            'chunk_vectors': self.chunk_vectors
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def find_relevant_chunk(self, doc_id: int, answer: str) -> Dict:
        """
        Find the most relevant document block containing the answer
        
        Args:
            doc_id: Document ID
            answer: Answer text to search for
            
        Returns:
            Document block containing the answer
        """
        if doc_id not in self.document_chunks:
            return None
            
        chunks = self.document_chunks[doc_id]
        best_chunk = None
        highest_overlap = -1
        
        for chunk in chunks:
            if answer.lower() in chunk["text"].lower():
                # Simple use of inclusion as relevance judgment
                # Actual application may require more complex relevance scoring
                overlap_score = len(answer) / len(chunk["text"])
                if overlap_score > highest_overlap:
                    highest_overlap = overlap_score
                    best_chunk = chunk
        
        return best_chunk
    
    def mine_hard_negatives_tfidf(self, question: str, answer: str, doc_id: int, n_samples: int = 3) -> List[Dict]:
        """
        Use TF-IDF to mine hard negative samples - Document blocks related to the question but not containing the answer
        
        Args:
            question: Question text
            answer: Answer text
            doc_id: Positive sample document ID
            n_samples: Number of negative samples to sample
            
        Returns:
            List of hard negative sample blocks
        """
        # Ensure TF-IDF index is built
        if self.tfidf_vectorizer is None or self.chunk_vectors is None:
            self.build_tfidf_index()
        
        # Convert question to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([question])
        
        # Calculate similarity between question and all document blocks
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        
        # Sort by similarity to get most similar indices
        top_indices = similarities.argsort()[::-1]
        
        # Collect hard negative samples that are not the same document and not containing the answer
        hard_negatives = []
        for idx in top_indices:
            chunk = self.chunk_list[idx]
            
            # Exclude blocks from the same document and containing the answer
            if chunk["doc_id"] != doc_id and answer.lower() not in chunk["text"].lower():
                hard_negatives.append(chunk)
                if len(hard_negatives) >= n_samples:
                    break
        
        # If not enough hard negative samples found, return what's found
        return hard_negatives
    
    def sample_negative_chunks(self, question: str, answer: str, positive_chunk: Dict, n_samples: int = 3) -> List[Dict]:
        """
        Sample negative sample blocks, prioritize using hard negative samples
        
        Args:
            question: Question text
            answer: Answer text
            positive_chunk: Positive sample block
            n_samples: Number of negative samples to sample
            
        Returns:
            List of negative sample blocks
        """
        # First try to mine hard negative samples
        hard_negatives = self.mine_hard_negatives_tfidf(
            question=question,
            answer=answer,
            doc_id=positive_chunk["doc_id"],
            n_samples=n_samples
        )
        
        # If enough hard negative samples found, return directly
        if len(hard_negatives) >= n_samples:
            return hard_negatives[:n_samples]
        
        # Otherwise use random negative samples to supplement
        all_doc_ids = list(self.document_chunks.keys())
        needed = n_samples - len(hard_negatives)
        random_negatives = []
        
        while len(random_negatives) < needed and all_doc_ids:
            doc_id = random.choice(all_doc_ids)
            
            if doc_id != positive_chunk["doc_id"] and doc_id in self.document_chunks:
                chunks = self.document_chunks[doc_id]
                if chunks:
                    chunk = random.choice(chunks)
                    # Ensure not containing the answer and not in already found hard negative samples
                    if (answer.lower() not in chunk["text"].lower() and 
                        chunk not in hard_negatives and
                        chunk not in random_negatives):
                        random_negatives.append(chunk)
            
            if len(random_negatives) >= needed:
                break
        
        return hard_negatives + random_negatives
    
    def _process_batch(self, batch_items, n_negatives: int) -> List[Dict]:
        """
        Process a batch of data
        
        Args:
            batch_items: Batch of data
            n_negatives: Number of negative samples per question
            
        Returns:
            Processed sample list
        """
        batch_examples = []
        
        for item in batch_items:
            try:
                # Ensure document ID exists
                doc_id = item.get("document_id") or item.get("doc_id")
                if doc_id is None:
                    continue  # Silent skip, no warning recorded
                
                question = item["question"]
                answer = item["answer"]
                
                positive_chunk = self.find_relevant_chunk(doc_id, answer)
                if positive_chunk:  # Only process if positive sample can be found
                    negative_chunks = self.sample_negative_chunks(
                        question=question,
                        answer=answer,
                        positive_chunk=positive_chunk,
                        n_samples=n_negatives
                    )
                    
                    if negative_chunks and len(negative_chunks) >= n_negatives:
                        example = {
                            "question": question,
                            "answer": answer,
                            "positive_ctx": positive_chunk["text"],  # Note: Single form used here
                            "negative_ctxs": [chunk["text"] for chunk in negative_chunks]
                        }
                        batch_examples.append(example)
            except Exception as e:
                logging.debug(f"Error processing item: {str(e)}")  # Lower log level, reduce output
                continue
        
        return batch_examples
        
    def prepare_dpr_training_data(self, n_negatives: int = 3, batch_size: int = 500, max_workers: int = 4) -> Tuple[List, List]:
        """
        Parallel prepare DPR training data
        
        Args:
            n_negatives: Number of negative samples per question
            batch_size: Batch processing size
            max_workers: Number of parallel processing worker processes
            
        Returns:
            (train_examples, val_examples) tuple
        """
        logging.info("Starting to prepare DPR training data...")
        
        # Ensure all necessary data is loaded
        if not self.documents:
            self.load_documents()
        
        if not self.train_data or not self.val_data:
            self.load_qa_data()
            
        if not self.document_chunks:
            self.chunk_documents()
            
        if self.tfidf_vectorizer is None:
            self.build_tfidf_index()
        
        start_time = time.time()
        
        # Prepare training data
        train_examples = []
        
        # Check training data cache - now using cache directory
        train_cache_file = os.path.join(self.cache_dir, f"train_examples_cache_{n_negatives}.pkl")
        if os.path.exists(train_cache_file):
            logging.info(f"Loading training samples from cache: {train_cache_file}")
            with open(train_cache_file, 'rb') as f:
                train_examples = pickle.load(f)
            logging.info(f"Successfully loaded {len(train_examples)} training samples")
        else:
            logging.info(f"Parallel processing training data, using {max_workers} processes...")
            # Batch process training data
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, len(self.train_data), batch_size):
                    end_idx = min(i + batch_size, len(self.train_data))
                    batch = self.train_data[i:end_idx]
                    futures.append(executor.submit(self._process_batch, batch, n_negatives))
                
                # Collect results
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing training data batches"):
                    batch_examples = future.result()
                    train_examples.extend(batch_examples)
            
            logging.info(f"Prepared {len(train_examples)} training samples")
            
            # Save training sample cache - now using cache directory
            logging.info(f"Saving training samples to cache: {train_cache_file}")
            with open(train_cache_file, 'wb') as f:
                pickle.dump(train_examples, f)
        
        # Prepare validation data
        val_examples = []
        
        # Check validation data cache - now using cache directory
        val_cache_file = os.path.join(self.cache_dir, f"val_examples_cache_{n_negatives}.pkl")
        if os.path.exists(val_cache_file):
            logging.info(f"Loading validation samples from cache: {val_cache_file}")
            with open(val_cache_file, 'rb') as f:
                val_examples = pickle.load(f)
            logging.info(f"Successfully loaded {len(val_examples)} validation samples")
        else:
            logging.info(f"Processing validation data, using {max_workers} processes...")
            # Batch process validation data
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(0, len(self.val_data), batch_size):
                    end_idx = min(i + batch_size, len(self.val_data))
                    batch = self.val_data[i:end_idx]
                    futures.append(executor.submit(self._process_batch, batch, n_negatives))
                
                # Collect results
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing validation data batches"):
                    batch_examples = future.result()
                    val_examples.extend(batch_examples)
            
            logging.info(f"Prepared {len(val_examples)} validation samples")
            
            # Save validation sample cache - now using cache directory
            logging.info(f"Saving validation samples to cache: {val_cache_file}")
            with open(val_cache_file, 'wb') as f:
                pickle.dump(val_examples, f)
        
        elapsed = time.time() - start_time
        logging.info(f"DPR training data preparation completed, time taken {elapsed:.2f} seconds")
        
        return train_examples, val_examples
    
    def save_dpr_data(self, train_examples: List, val_examples: List) -> None:
        """
        Save DPR data to files
        
        Args:
            train_examples: Training sample list
            val_examples: Validation sample list
        """
        train_output_file = os.path.join(self.data_dir, "dpr_train.json") 
        val_output_file = os.path.join(self.data_dir, "dpr_val.json")
        
        logging.info(f"Saving {len(train_examples)} training samples to {train_output_file}")
        with open(train_output_file, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saving {len(val_examples)} validation samples to {val_output_file}")
        with open(val_output_file, 'w', encoding='utf-8') as f:
            json.dump(val_examples, f, ensure_ascii=False, indent=2)
            
        logging.info(f"DPR training data saved to {train_output_file} and {val_output_file}")
    
    def process_all(self, max_workers: int = 4):
        """
        Execute complete DPR data processing workflow
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.load_documents()
        self.load_qa_data()
        self.chunk_documents()
        self.build_tfidf_index()
        train_examples, val_examples = self.prepare_dpr_training_data(max_workers=max_workers)
        self.save_dpr_data(train_examples, val_examples)
        logging.info("DPR data processing workflow completed") 