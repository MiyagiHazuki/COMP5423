import json
import re
from bs4 import BeautifulSoup
import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import os

from utils import log_message, info, debug, warning, error

nltk.download('punkt', quiet=True)

class DocumentProcessor:
    """Base class for document processing, containing common text processing methods"""
    
    @staticmethod
    def clean_wikipedia_text(text):
        """Clean text"""
        # First remove line breaks
        text = re.sub(r'\n', ' ', text)
        # Remove Contents (hide) section and surrounding spaces
        text = re.sub(r'\s*Contents \( hide \)\s*', ' ', text)
        # Remove (edit) tags and surrounding spaces
        text = re.sub(r'\s*\(edit\)\s*', ' ', text)
        # Specifically handle (edit) tags with spaces inside
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
        # Clean excess spaces
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
        """Process to plain text"""
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
        info("Starting to calculate TF-IDF model...")
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        info("TF-IDF model calculation completed")
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """Calculate BM25 model"""
        info("Starting to calculate BM25 model...")
        
        # Use tqdm to wrap tokenization process
        tokenized_corpus = []
        for doc in tqdm(self.corpus, desc="Tokenizing", unit="doc"):
            tokenized_corpus.append(DocumentProcessor.tokenize(doc))
        
        info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        info("BM25 index construction completed")
        return self.bm25
    
    def save_models(self, output_prefix):
        """Save models to files"""
        if self.tfidf_vectorizer:
            info(f"Saving TF-IDF model to {output_prefix}_tfidf.pkl...")
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            info(f"TF-IDF model saved")
        if self.bm25:
            info(f"Saving BM25 model to {output_prefix}_bm25.pkl...")
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
            info(f"BM25 model saved")


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
        # First calculate the number of lines to determine total progress
        with open(self.input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)                                                                 
        
        info(f"Starting to process documents, total {total_lines} lines...")
        
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
                    error(f"Error parsing JSON, line content: {line[:100]}... Error: {e}")
                except KeyError as e:
                    error(f"Missing key field in document: {e}")
    
    def save_processed_data(self):
        """Save processed data"""
        plain_output = os.path.join(self.data_output_dir, "processed_plain.jsonl")
        markdown_output = os.path.join(self.data_output_dir, "processed_markdown.jsonl")
        
        info(f"Saving processed plain text data to {plain_output}...")
        with open(plain_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_plain, desc="Saving plain text data", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        info(f"Saving processed markdown data to {markdown_output}...")
        with open(markdown_output, 'w', encoding='utf-8') as f:
            # Using tqdm to show progress bar
            for item in tqdm(self.processed_markdown, desc="Saving markdown data", unit="doc"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        info(f"Plain text data saved to {plain_output}")
        info(f"Markdown data saved to {markdown_output}")
    
    def build_models(self):
        """Build search models"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        info("Starting to build models for plain text...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(os.path.join(self.model_output_dir, "plain"))
        
        info("Starting to build models for markdown...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(os.path.join(self.model_output_dir, "markdown"))
        
        info("All models have been successfully built and saved")
    
    def run(self):
        """Run the entire processing workflow"""
        self.process_documents()
        self.save_processed_data()
        self.build_models()

'''
if __name__ == "__main__":
    # Configuration parameters
    input_file = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/documents.jsonl"
    data_output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    model_output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/model/"
    
    # Create and run pipeline
    pipeline = DocumentPipeline(input_file, model_output_dir, data_output_dir)
    pipeline.run()
'''