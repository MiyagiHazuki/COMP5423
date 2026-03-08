import json
import re
from bs4 import BeautifulSoup
import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import pickle
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

class DocumentProcessor:
    """Base class for document processing, containing common text processing methods"""
    
    @staticmethod
    def clean_wikipedia_text(text):
        """Clean up text"""
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
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=DocumentProcessor.tokenize, 
            stop_words='english'
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        return self.tfidf_vectorizer, self.tfidf_matrix
    
    def calculate_bm25(self):
        """Calculate BM25 model"""
        tokenized_corpus = [DocumentProcessor.tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        return self.bm25
    
    def save_models(self, output_prefix):
        """Save models to files"""
        if self.tfidf_vectorizer:
            with open(f'{output_prefix}_tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        if self.bm25:
            with open(f'{output_prefix}_bm25.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)


class DocumentPipeline:
    """Document processing pipeline, integrating the entire workflow"""
    
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.processed_plain = []
        self.processed_markdown = []
    
    def process_documents(self):
        """Process raw data"""
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
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
                    print(f"Error decoding JSON for line: {line[:100]}... Error: {e}")
                except KeyError as e:
                    print(f"Missing key in document: {e}")
    
    def save_processed_data(self):
        """Save processed data"""
        plain_output = f"{self.output_dir}processed_plain.jsonl"
        markdown_output = f"{self.output_dir}processed_markdown.jsonl"
        
        with open(plain_output, 'w', encoding='utf-8') as f:
            for item in self.processed_plain:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(markdown_output, 'w', encoding='utf-8') as f:
            for item in self.processed_markdown:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved plain text data to {plain_output}")
        print(f"Saved markdown data to {markdown_output}")
    
    def build_models(self):
        """Build search models"""
        plain_corpus = [doc['text'] for doc in self.processed_plain]
        markdown_corpus = [doc['text'] for doc in self.processed_markdown]
        
        print("Building models for plain text...")
        plain_indexer = DocumentIndexer(plain_corpus)
        plain_indexer.calculate_tfidf()
        plain_indexer.calculate_bm25()
        plain_indexer.save_models(f"{self.output_dir}plain")
        
        print("Building models for markdown...")
        markdown_indexer = DocumentIndexer(markdown_corpus)
        markdown_indexer.calculate_tfidf()
        markdown_indexer.calculate_bm25()
        markdown_indexer.save_models(f"{self.output_dir}markdown")
        
        print("Models built and saved successfully")
    
    def run(self):
        """Run the entire processing workflow"""
        print("Processing documents...")
        self.process_documents()
        self.save_processed_data()
        self.build_models()


if __name__ == "__main__":
    # Configuration parameters
    input_file = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/documents.jsonl"
    output_dir = "/Users/cusnsheep/Documents/dsa_course/S2/COMP5423/project/data_and_code(1)/data/output/"
    
    # Create and run pipeline
    pipeline = DocumentPipeline(input_file, output_dir)
    pipeline.run()