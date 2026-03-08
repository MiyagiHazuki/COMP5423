import gradio as gr
import yaml
import os
from utils import log_message, debug, warning, error
import sys
import traceback
from backbone.llm import QwenChatClient
from backbone.rerank import ReRanker

class WebUI:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize WebUI class
        
        Functions:
        - Load configuration file
        - Set WebUI related parameters (port, whether to display interface)
        - Determine default retrieval type
        
        Parameters:
        - config_path: Configuration file path, default is "config.yaml"
        
        Integration with user interface:
        - Determines the port used when WebUI starts
        - Determines whether to display the WebUI interface
        - Sets the default value for Radio button
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.port = self.config.get("webui", {}).get("port", 8080)
        self.presentation = self.config.get("webui", {}).get("presentation", True)
        self.retrieval_type = self.config.get("modules", {}).get("active", {}).get("retrieval", "hybrid")
        
        # Initialize searcher
        self.init_searcher()
    
    def init_searcher(self):
        """Initialize BM25 searcher"""
        # Get BM25 configuration
        bm25_config = self.config.get("retrieval", {}).get("text_retrieval", {})
        retrieval_method = bm25_config.get("method", "hybrid")
        
        # Handle file paths
        data_output_dir = self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")
        model_output_dir = self.config.get("data", {}).get("model_output_dir", "./model/pkl/")
        
        # Decide which processed document to use based on use_plain_text
        use_plain_text = bm25_config.get("use_plain_text", True)
        doc_type = "plain" if use_plain_text else "markdown"
        doc_path = os.path.join(data_output_dir, f"processed_{doc_type}.jsonl")
        bm25_path = os.path.join(model_output_dir, f"{doc_type}_bm25.pkl")
        tfidf_path = os.path.join(model_output_dir, f"{doc_type}_tfidf.pkl")
        
        log_message(f"Document path: {doc_path}")
        log_message(f"BM25 model path: {bm25_path}")
        log_message(f"TF-IDF model path: {tfidf_path}")
        
        # Check if necessary files exist
        required_files = [doc_path, bm25_path, tfidf_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            for file in missing_files:
                log_message(f"Error: File {file} does not exist")
            log_message("Please run process_mode first to generate necessary model and data files")
            import sys
            sys.exit(1)
        
        # Build query searcher configuration
        self.search_config = {
            "method": retrieval_method,
            "hybrid_alpha": bm25_config.get("hybrid_alpha", 0.7),
            "top_k": bm25_config.get("top_k", 5),
            "max_words": bm25_config.get("max_words", 2),
            "doc_path": doc_path,
            "bm25_path": bm25_path,
            "tfidf_path": tfidf_path
        }
        
        try:
            # Import and instantiate searcher
            from bm25.base import ManualQuerySearch
            log_message("Successfully imported ManualQuerySearch...")
            
            # Add more detailed loading logs
            log_message("Starting to initialize searcher...")
            log_message("Loading documents, this may take some time...")
            log_message("Loading BM25 model, this may take some time...")
            log_message("If using hybrid mode, TF-IDF model will also be loaded...")
            
            # Instantiate searcher
            self.searcher = ManualQuerySearch(self.search_config)
            log_message(f"Successfully initialized ManualQuerySearch with parameters: {self.search_config}")
            debug("Searcher initialization complete, starting to build UI...")
        except Exception as e:
            error(f"Error initializing ManualQuerySearch: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    def load_config(self, config_path):
        """
        Load configuration file
        
        Functions:
        - Read YAML format configuration file
        - Parse into Python dictionary structure
        
        Parameters:
        - config_path: Configuration file path
        
        Return value:
        - config: Dictionary containing configuration information
        
        Integration with user interface:
        - Indirectly determines various configuration parameters of WebUI
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    
    def retrieval_text(self, query):
        """
        Text retrieval module interface [Customizable]
        
        Functions:
        - Use BM25 algorithm to retrieve documents related to the query from document collection
        - BM25 is a classic retrieval algorithm based on term frequency and inverse document frequency
        
        Parameters:
        - query: User input query string
        - context: Optional context information, default is None
        
        Return value:
        - {answer: "str: answer", document: "list: related documents"}

        Integration with user interface:
        - Called when user selects "bm25" in Radio button and submits a question
        - Retrieval results will be processed by process_query method and finally displayed in related documents area of WebUI
        
        Configuration related:
        - Uses parameters from retrieval.bm25 section in config.yaml (k1 and b values)
        """
        if not query or not query.strip():
            return {
                "question": "",
                "document_id": []
            }
        
        # Use initialized searcher
        result = self.searcher.search(query)
        return result
    
    def retrieval_dpr(self, query):
        """
        DPR retrieval module interface [Customizable]
        
        Functions:
        - Use BERT model for semantic retrieval to find documents semantically related to the query
        - Vector retrieval method based on deep learning, capturing semantic similarity between query and documents
        
        Parameters:
        - query: User input query string
        - context: Optional context information, default is None
        
        Return value:
        - List of retrieved related documents
        
        Integration with user interface:
        - Called when user selects "dpr" in Radio button and submits a question
        - Retrieval results will be processed by process_query method and finally displayed in related documents area of WebUI
        
        Configuration related:
        - Uses model configuration from retrieval.dpr section in config.yaml
        """
        from dpr.query_process import query2doclist as q2d
        q2d = q2d(query, self.get_api_key(), "BAAI/bge-m3", "./cache/faiss/bgem3.faiss", 5)
        doc_id_list = q2d.query2doclist()
        return doc_id_list

    def retrieval_hybrid(self, query):
        """
        Hybrid retrieval module interface [Customizable]
        
        Functions:
        - Combine retrieval results from BM25 and BERT, leveraging advantages of multiple retrieval methods
        - Can use different fusion methods (weighted sum, reciprocal rank fusion, etc.)
        
        Parameters:
        - query: User input query string
        - context: Optional context information, default is None
        
        Return value:
        - List of retrieved related documents
        
        Integration with user interface:
        - Called when user selects "hybrid" in Radio button and submits a question
        - Retrieval results will be processed by process_query method and finally displayed in related documents area of WebUI
        
        Configuration related:
        - Uses configuration from retrieval.hybrid section in config.yaml
        - weights parameter determines the weights of BM25 and BERT
        - fusion_method determines the fusion method (weighted_sum or reciprocal_rank, etc.)
        """
        text_result = self.retrieval_text(query)
        text_doc_list = text_result.get("document_id", [])
        dense_doc_list = self.retrieval_dpr(query).get("document_id", [])
        hybrid_doc_list = list(set(text_doc_list + dense_doc_list))
        reranked_index = self.rerank_results(query, hybrid_doc_list, 5)
        reranked_doc_ids = []
        for idx in reranked_index:
            reranked_doc_ids.append(hybrid_doc_list[idx])
        dict = {"document_id": reranked_doc_ids}
        return dict

    def get_api_key(self) -> str:
        """
        Get API key, Silicon Flower's API key
        """
        api_key = self.config.get("generation", {}).get("qwen", {}).get("api_key", "")
        if not api_key:
            raise ValueError("API key not configured")
        return api_key
    
    def retrieve_doc(self, idx_list: list[int]) -> list[str]:
        """Find corresponding documents based on index values"""
        try:
            doc_path = self.search_config.get("doc_path")
            log_message(f"Reading document file: {doc_path}")

            # Convert to string list for matching
            str_idx_list = [str(idx) for idx in idx_list]
            
            import pandas as pd
            df = pd.read_json(doc_path, lines=True, dtype={'doc_id': str})
            
            # Debug output data structure
            debug(f"Data file contains columns: {df.columns.tolist()}")
            debug(f"Sample of first 3 data entries:\n{df.head(3)}")
            
            # Check if necessary fields exist
            if 'doc_id' not in df.columns or 'text' not in df.columns:
                raise KeyError("Data file is missing doc_id or text fields")

            # Batch query and maintain order
            matched_df = df[df['doc_id'].isin(str_idx_list)]
            debug(f"Matched {len(matched_df)} records, expected ID list: {str_idx_list}")

            # Sort by input order and deduplicate
            ordered_docs = []
            seen = set()
            for idx in str_idx_list:
                if idx not in seen:
                    doc = matched_df[matched_df['doc_id'] == idx]
                    if not doc.empty:
                        ordered_docs.append(doc.iloc[0]['text'])
                        seen.add(idx)

            # Record matching results
            found_ids = list(seen)
            if found_ids:
                log_message(f"Successfully matched document IDs: {found_ids}")
            else:
                warning("No matching documents found")
            print(ordered_docs[0])
            return ordered_docs

        except Exception as e:
            error(f"Document retrieval failed: {str(e)}")
            traceback.print_exc()
            return []

    def rerank_results(self, query:str, document_id_list:list[int], top_n:int=1):
        """
        Rerank document_id_list by relevance
        input:
        - query: User input query string
        - document_id_list: list[int], document index list, i.e., document_ID
        - top_n: int, number of document_ids to return
        return: list of indices for top_n documents
        """
        api_token = self.get_api_key()
        document_content_list = self.retrieve_doc(document_id_list)
        query_document_list = [
            [query, document_content_list]
        ]
        reranker = ReRanker(api_token) 
        responses = reranker.async_send_requests_simple(
            query_document_list, use_progress_bar=False, concurrency=1
        )
        _, indices = reranker.extract_json(responses) # return: scores, indices
        return indices[0][:top_n] # take indices for top n documents

    def generate_answer(self, query:str, context:str, n=5)->str:
        """
        Generate answer using QwenChatClient
        Input parameters:
        - query: User input query string
        - context: Retrieved related documents
        - n: Number of answers to generate
        return:
        answer: Generated answer text
        """
        api_token = self.get_api_key()
        client = QwenChatClient(api_token=api_token)
        query_context_list = [(query, context)]
        results = client.batch_request_async_simple(
            query_context_list=query_context_list, 
            concurrency=1, model="Qwen/Qwen2.5-7B-Instruct", 
            n = n)
        extracted_answers_list = client.extract_answer(results)[0]
        # Add debug output
        print("DEBUG - Extracted answers:", extracted_answers_list)
        # Check if all answers are "wrong"
        if all(answer.lower() == "wrong" for answer in extracted_answers_list):
            print("DEBUG - All answers are 'wrong', returning 'wrong'")
            return "wrong"
        # return the most frequent answer
        answer = max(set(extracted_answers_list), key=extracted_answers_list.count)
        print("DEBUG - Final answer:", answer)
        return answer

    def format_document_display(self, document_ids):
        """Format document ID list for display text"""
        if not document_ids:
            return "No related documents found"
            
        # Ensure document IDs are string type
        str_doc_ids = [str(doc_id) for doc_id in document_ids]
            
        # Display up to first 5 document IDs
        doc_list_text = ", ".join(str_doc_ids[:5])
        
        return f"related documents: {doc_list_text}"

    def process_query(self, query, retrieval_method, i=0):
        """
        Core method for processing user queries
        
        Functions:
        - Serves as a bridge between WebUI and backend processing logic
        - Coordinates workflow of various modules (retrieval, reranking, generation)
        
        Parameters:
        - query: User input query string
        - retrieval_method: Retrieval method selected by user ("bm25", "bert_base" or "hybrid")
        - i: Current document index being processed, used for recursive calls
        
        Return value:
        - answer: Generated answer text
        - context_display: Formatted related document text
        
        Integration with user interface:
        - Directly called by submit_button.click event
        - Receives user input and returns results for display on interface
        - Core connection point between user interaction behavior and system processing logic
        
        Workflow:
        1. Call appropriate retrieval interface based on user-selected retrieval method
        2. Rerank retrieval results
        3. Generate answer based on reranked documents
        4. Format results for interface display
        """
        # Input check
        if not query or not query.strip():
            return "Please enter query content", "No related documents"
            
        # Remove leading and trailing whitespace
        query = query.strip()
        
        # Record query information
        log_message(f"User submitted query: '{query}', using retrieval method: {retrieval_method}, document index: {i}")
        
        try:
            # Retrieve documents based on selected retrieval method
            if retrieval_method == "bm25":
                retrieval_results = self.retrieval_text(query)
            elif retrieval_method == "dpr":
                retrieval_results = self.retrieval_dpr(query)
            else:  # hybrid
                retrieval_results = self.retrieval_hybrid(query)
            
            # Get document ID list
            doc_ids = retrieval_results.get("document_id", [])
            
            # If no related documents found, generate answer directly
            if not doc_ids:
                log_message(f"No documents related to query '{query}' found")
                answer = "Cannot find related documents"
                return "None", "Null"
            
            # Rerank
            reranked_doc_ids = doc_ids
            
            try:
                # Try to retrieve document content
                retrieved_docs = self.retrieve_doc(doc_ids)
                if retrieved_docs:
                    answer = self.generate_answer(query, retrieved_docs[i])
                else:
                    # If retrieve_doc returns empty list, use fallback answer
                    log_message(f"Although document IDs were found, could not retrieve related document content, list is empty")
                    raise ValueError("Cannot find related content of the query")
                    sys.exit(1)
            except Exception as e:
                # Catch exceptions that retrieve_doc might throw
                log_message(f"Error retrieving documents: {str(e)}")
                answer = self.generate_answer(query, f"query: {query}")
            
            if answer.lower() == "wrong":
                if i < len(doc_ids)-1:
                    print(f"wrong answer, try the next document: {i+1}")
                    # Use recursive call to process next document
                    return self.process_query(query, retrieval_method, i+1)
                else:
                    answer = "Cannot find any answer from the list of documents"
            
            # Format document IDs as display text
            context_display = self.format_document_display(reranked_doc_ids)
            context_display = context_display + "\n" +"\n" + retrieved_docs[i]  
            
            return answer, context_display
            
        except Exception as e:
            # Catch exceptions in entire processing flow
            error_msg = f"Error processing query: {str(e)}"
            log_message(error_msg)
            import traceback
            traceback.print_exc()
            
            # Try to generate answer directly without relying on retrieval results
            try:
                answer = self.generate_answer(query, "Error processing query, will answer directly.")
            except:
                answer = "Sorry, an error occurred while processing your query, please try again later."
                
            return answer, "Error processing query"

    def build_ui(self):
        """
        Build Gradio interface
        
        Functions:
        - Create and configure WebUI components
        - Set layout and interaction logic between components
        
        Return value:
        - demo: Configured Gradio Blocks interface object
        
        Integration with user interface:
        - Directly defines all interface elements that users can see
        - Sets callback functions triggered by user interactions (such as button clicks)
        
        Interface components:
        1. Title: Displays "Intelligent Q&A System"
        2. Question input box: Text box for user to input query
        3. Retrieval method selection: Radio button group for user to select retrieval method
        4. Submit button: Triggers query processing
        5. Answer display area: Shows answer generated by system
        6. Related document display area: Shows retrieved related documents
        
        Interaction flow:
        1. User enters question in input box
        2. User selects retrieval method (default uses method specified in config file)
        3. User clicks submit button
        4. System calls process_query to process query
        5. Processing results displayed in answer area and related document area
        """
        try:
            debug("Starting to build Gradio interface...")
            with gr.Blocks(title="Q&A System") as demo:
                gr.Markdown("# Intelligent Q&A System")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Question input box - use placeholder instead of value to avoid auto-triggering query
                        query_input = gr.Textbox(
                            label="Please enter your question", 
                            placeholder="Example: What is machine learning?", 
                            lines=2,
                            value=""  # Ensure initial value is empty
                        )
                        
                        # Retrieval method selection
                        retrieval_method = gr.Radio(
                            choices=["bm25", "dpr", "hybrid"],
                            value=self.retrieval_type,
                            label="Retrieval Method",
                            info="Select retrieval method to use"
                        )
                        
                        # Submit button
                        submit_button = gr.Button("Submit Question", variant="primary")
                        
                        # Add example question buttons instead of auto-executing examples
                        with gr.Accordion("Example Questions", open=False):
                            sample1_btn = gr.Button("What is machine learning?")
                            sample2_btn = gr.Button("What is the difference between deep learning and traditional machine learning?")
                    
                    with gr.Column(scale=4):
                        # Answer display area
                        answer_output = gr.Textbox(
                            label="answer", 
                            lines=6,
                            value="model loaded, please input question and click submit."
                        )
                        
                        # Related document display area
                        context_output = gr.Textbox(
                            label="related documents", 
                            lines=10,
                            value="waiting for input..."
                        )
                
                # Set submit button functionality
                submit_button.click(
                    fn=self.process_query,
                    inputs=[query_input, retrieval_method],
                    outputs=[answer_output, context_output]
                )
                
                # Example question button click events
                def set_example_1():
                    return "What is machine learning?"
                
                def set_example_2():
                    return "What is the difference between deep learning and traditional machine learning?"
                
                sample1_btn.click(
                    fn=set_example_1,
                    inputs=[],
                    outputs=[query_input]
                )
                
                sample2_btn.click(
                    fn=set_example_2,
                    inputs=[],
                    outputs=[query_input]
                )
                
            debug("Gradio interface build complete")
            return demo
        except Exception as e:
            error(f"Error building UI: {str(e)}")
            traceback.print_exc()
            raise

    def launch(self):
        """
        Launch WebUI service
        
        Functions:
        - Decide whether to launch WebUI based on configuration
        - Start Gradio server
        
        Return value:
        - Boolean indicating whether WebUI was successfully launched
        
        Integration with user interface:
        - Controls whether WebUI is displayed to users
        - Sets server port, users access this port via browser to view interface
        
        Configuration related:
        - Determines whether to launch interface based on webui.presentation in config.yaml
        - Uses port number specified by webui.port
        
        Launch process:
        1. Check if presentation configuration is true
        2. If true, build UI and start server
        3. If false, output prompt message
        
        User usage:
        - When user runs main.py, launch method is called
        - If launch is successful, user can access http://localhost:port in browser to view interface
        """
        if self.presentation:
            try:
                log_message(f"Starting to build WebUI interface...")
                demo = self.build_ui()
                log_message(f"WebUI interface build complete, attempting to start service on port {self.port}...")
                
                # Use more parameters to ensure stable startup, start Gradio in multi-threaded way
                import threading
                def start_server():
                    try:
                        demo.launch(
                            server_port=self.port,
                            share=False,
                            inbrowser=True,  # Auto open browser
                            show_error=True,  # Show detailed errors
                            server_name="0.0.0.0",  # Bind to all interfaces
                            prevent_thread_lock=True,  # Prevent thread locking
                            quiet=False  # Don't use quiet mode
                        )
                    except Exception as e:
                        error(f"Gradio server startup failed: {str(e)}")
                
                # Start server thread
                server_thread = threading.Thread(target=start_server)
                server_thread.daemon = True  # Set as daemon thread
                server_thread.start()
                
                log_message(f"WebUI service has started, accessible via http://localhost:{self.port}")
                
                # Wait for user interruption
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    log_message("User interrupted, closing WebUI service...")
                
                return True
            except Exception as e:
                error(f"Error starting WebUI service: {str(e)}")
                traceback.print_exc()
                return False
        else:
            print("WebUI not enabled. Set webui.presentation to true in config.yaml to enable WebUI.")
            return False


def create_webui(config_path):
    """
    Convenient function to create and launch WebUI
    
    Functions:
    - Create WebUI instance
    - Launch interface service
    
    Parameters:
    - config_path: Configuration file path
    
    Return value:
    - Boolean indicating whether WebUI was successfully launched
    
    Integration with user interface:
    - Entry point called by main.py
    - Indirectly triggered when user executes command line instructions
    
    Usage scenarios:
    - When user runs python main.py --config config.yaml
    - main.py parses arguments then calls this function
    - Function creates WebUI instance and attempts to launch interface
    """
    webui = WebUI(config_path)
    return webui.launch()
