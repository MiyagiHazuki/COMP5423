import sys
import yaml
import os
import argparse
from utils import initialize_logger, log_message, set_log_level, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, debug, error, info, warning

def check_dependencies():
    """Check if necessary dependencies are installed"""
    try:
        # List of required dependencies
        required_packages = ["gradio", "nltk", "numpy", "sklearn", "tqdm", "rank_bm25", "faiss"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            log_message(f"Missing required dependencies: {', '.join(missing_packages)}")
            log_message("Please use pip install -r requirements.txt to install all dependencies")
            return False
        
        return True
    except Exception as e:
        log_message(f"Error checking dependencies: {str(e)}")
        return False

def load_config(config_path="config.yaml"):
    """Load configuration file"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        sys.exit(1)

def vectorize_with_bgem3(config):
    """Using BGE-M3 to vectorize documents and save as FAISS index"""
    try:
        info("Starting BGE-M3 document vectorization and FAISS index creation process...")
        
        # Import necessary modules
        import bgem3
        import faiss_composer
        import numpy as np
        import json
        import time
        import traceback
        from pathlib import Path
        
        # Get vectorization configuration
        bgem3_config = config.get("retrieval", {}).get("deep_retrieval", {}).get("bgem3", {})
        faiss_config = config.get("retrieval", {}).get("faiss", {})
        
        # Parse configuration
        model_name = bgem3_config.get("model", "BAAI/bge-m3")
        api_key = bgem3_config.get("api_key", "")
        doc_path = bgem3_config.get("doc_path", "./data/processed_data/processed_plain.jsonl")
        intermediate_dir = bgem3_config.get("intermediate_path", "./cache/bge_m3_intermediate")
        chunk_size = bgem3_config.get("chunk_size", 10)
        overlap_ratio = max(0.0, min(0.5, bgem3_config.get("overlap_ratio", 0.2)))
        use_weighted_avg = bgem3_config.get("use_weighted_avg", True)
        auto_load = bgem3_config.get("auto_load", True)
        
        # Get FAISS configuration
        index_name = faiss_config.get("index_name", "bgem3")
        index_path = faiss_config.get("index_path", "./faiss/")
        
        # Output configuration information
        info(f"Vectorization configuration:")
        info(f"- Document path: {doc_path}")
        info(f"- Model: {model_name}")
        info(f"- Overlap ratio: {overlap_ratio:.2f}")
        info(f"- Batch size: {chunk_size}")
        info(f"- Use weighted average: {use_weighted_avg}")
        info(f"- Intermediate results directory: {intermediate_dir}")
        info(f"- Auto load intermediate results: {auto_load}")
        info(f"- FAISS index path: {index_path}")
        info(f"- Index name: {index_name}")
        
        # Define helper functions
        def save_intermediate_results(doc_ids, vectors, save_path=intermediate_dir):
            """Save intermediate results to file"""
            os.makedirs(save_path, exist_ok=True)
            timestamp = int(time.time())
            result_path = f"{save_path}/vectors_{timestamp}.json"
            
            # Convert numpy array to list
            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()
            
            data = {
                "doc_ids": doc_ids,
                "vectors": vectors
            }
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            info(f"Already saved intermediate results to {result_path}, containing {len(doc_ids)} document vectors")
            return result_path

        def load_intermediate_results(result_path):
            """Load intermediate results from file"""
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data["doc_ids"], data["vectors"]

        def get_latest_intermediate_file(intermediate_path):
            """Get the latest intermediate result file"""
            if not os.path.exists(intermediate_path):
                return None
                
            files = [f for f in os.listdir(intermediate_path) if f.startswith("vectors_") and f.endswith(".json")]
            if not files:
                return None
                
            # Sort by timestamp to find the latest file
            latest_file = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)[0]
            return f"{intermediate_path}/{latest_file}"
        
        # Create necessary directories
        os.makedirs(intermediate_dir, exist_ok=True)
        os.makedirs(index_path, exist_ok=True)
        
        # Try to load previous intermediate results
        latest_file = get_latest_intermediate_file(intermediate_dir)
        all_doc_ids = []
        all_vectors = []
        start_index = 0
        
        if latest_file and auto_load:
            info(f"Automatically loading the latest intermediate results: {latest_file}")
            all_doc_ids, all_vectors = load_intermediate_results(latest_file)
            info(f"Loaded {len(all_doc_ids)} document vectors")
        elif latest_file:
            load_choice = input(f"Found previous intermediate results: {latest_file}, load? (y/n): ").strip().lower()
            if load_choice == 'y':
                all_doc_ids, all_vectors = load_intermediate_results(latest_file)
                info(f"Loaded {len(all_doc_ids)} document vectors")
        
        # Load document data
        info("Starting to load document data...")
        if not os.path.exists(doc_path):
            error(f"Document file does not exist: {doc_path}")
            return False
            
        doc_jsonler = bgem3.doc_jsonler(doc_path)
        doc_ids, contexts = doc_jsonler.get_json_data()
        info(f"Successfully loaded {len(doc_ids)} documents")
        
        # Calculate total number of documents and estimated processing cost
        total_docs = len(doc_ids)
        
        # Create a vectorizer instance for estimating processing cost and time
        temp_vectorizer = bgem3.doc_vectorizer(doc_ids, contexts, api_key, model_name)
        estimation = temp_vectorizer.estimate_processing_cost(sample_size=min(100, total_docs))
        
        info("===== Processing estimation =====")
        info(f"Total number of documents: {estimation['total_documents']}")
        info(f"Sample size: {estimation['sample_size']}")
        info(f"Average number of tokens per document: {estimation['avg_tokens_per_doc']:.1f}")
        info(f"Estimated total token number: {estimation['estimated_total_tokens']:.0f}")
        info(f"Estimated processing time: {estimation['estimated_time']['formatted']} (HH:MM:SS)")
        info(f"Estimated API cost: ¥{estimation['estimated_cost_cny']:.2f} CNY")
        info("===================")
        
        # Confirm whether to continue
        if not auto_load and total_docs > 10:
            confirm = input("Continue processing? (y/n): ").strip().lower()
            if confirm != 'y':
                info("Processing cancelled")
                return False
        
        # If intermediate results are loaded, determine the starting index for processing
        if all_doc_ids:
            # Find the position of the last processed document in the original document list
            processed_doc_ids_set = set(all_doc_ids)
            for i in range(len(doc_ids)):
                if doc_ids[i] not in processed_doc_ids_set:
                    start_index = i
                    break
            info(f"Continue processing from the {start_index+1}th document (skipping {start_index} processed documents)")
        
        # Start processing documents
        processed_count = len(all_doc_ids)
        start_time = time.time()
        
        try:
            for i in range(start_index, total_docs, chunk_size):
                try:
                    end_idx = min(i + chunk_size, total_docs)
                    info(f"Processing documents {i+1} to {end_idx} (total {total_docs} documents, {((i+processed_count-start_index)/total_docs*100):.1f}% completed)...")
                    
                    # Get the current batch of documents
                    batch_doc_ids = doc_ids[i:end_idx]
                    batch_contexts = contexts[i:end_idx]
                    
                    # Create a batch vectorizer
                    batch_vectorizer = bgem3.doc_vectorizer(
                        batch_doc_ids, 
                        batch_contexts, 
                        api_key, 
                        model_name
                    )
                    
                    # Process the current batch, using the enhanced processing method with overlap
                    result_ids, result_vectors = batch_vectorizer.process_documents_enhanced(
                        chunk_size=1, 
                        overlap_ratio=overlap_ratio,
                        use_weighted_avg=use_weighted_avg
                    )
                    
                    # Add to results
                    all_doc_ids.extend(result_ids)
                    all_vectors.extend(result_vectors)
                    
                    processed_count += len(result_ids)
                    elapsed_time = time.time() - start_time
                    docs_per_second = (processed_count - len(all_doc_ids) + (i-start_index)) / elapsed_time if elapsed_time > 0 else 0
                    
                    info(f"Completed {len(all_doc_ids)}/{total_docs} documents (speed: {docs_per_second:.2f} documents/second)")
                    
                    # Save intermediate results periodically
                    if i > start_index and (len(all_doc_ids) % 50 == 0 or end_idx == total_docs):
                        save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                    
                    # Dynamically adjust the waiting time based on the document processing speed
                    if end_idx < total_docs:
                        # If the processing speed is fast, reduce the waiting time; if slow, increase the waiting time
                        if docs_per_second > 0.5:  # Process one document every 2 seconds
                            wait_time = 1
                        else:
                            wait_time = 3
                        debug(f"Waiting {wait_time} seconds before processing the next batch...")
                        time.sleep(wait_time)
                    
                except Exception as e:
                    # Save the processed results
                    if all_doc_ids:
                        save_path = save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                        warning(f"Error occurred during processing, intermediate results saved to {save_path}")
                    error(f"Error details: {str(e)}")
                    traceback.print_exc()
                    break
            
            # After processing, save to the FAISS index
            if all_doc_ids:
                info("Saving to the FAISS index...")
                
                # Ensure vectors are normalized
                vectors = np.array(all_vectors)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                zero_mask = norms == 0
                norms[zero_mask] = 1.0  # Avoid division by zero
                normalized_vectors = vectors / norms
                
                # Use FaissSaver to save the index
                faiss_saver = faiss_composer.FaissSaver(all_doc_ids, normalized_vectors.tolist(), index_name, index_path)
                index_path_full = faiss_saver.save()
                info(f"Index saved to: {index_path_full}")
                
                # Query test functionality
                test_query = input("Whether to perform query test? (y/n): ").strip().lower()
                if test_query == 'y':
                    # Import required modules
                    try:
                        from dpr.vectorizer import query_vectorizer
                        
                        while True:
                            # Provide two test methods
                            print("\nQuery test options:")
                            print("[1] Test using example vectors")
                            print("[2] Test using input text")
                            print("[0] Exit test")
                            
                            choice = input("Please select the test method [0-2]: ")
                            
                            if choice == "0":
                                break
                            elif choice == "1":
                                # Test using the first document vector
                                query_vector = [normalized_vectors[0].tolist()]
                                faiss_query = faiss_composer.FaissQuery(
                                    query_vector,
                                    index_name,
                                    index_path,
                                    k=5
                                )
                                try:
                                    result_doc_ids, scores = faiss_query.query()
                                    print("\nExample vector query results:")
                                    for i, (doc_id, score) in enumerate(zip(result_doc_ids, scores)):
                                        print(f"[{i+1}] Document ID: {doc_id}, Similarity score: {score:.4f}")
                                except Exception as e:
                                    error(f"Query test failed: {str(e)}")
                                    
                            elif choice == "2":
                                # Test using input text
                                query_text = input("Please enter the query text: ")
                                if query_text.strip():
                                    # Use query_vectorizer to generate the query vector
                                    vectorizer = query_vectorizer([query_text], api_key, model_name)
                                    print("Generating query vector...")
                                    query_vectors = vectorizer.vectorize()
                                    
                                    # Perform query
                                    faiss_query = faiss_composer.FaissQuery(
                                        query_vectors,
                                        index_name,
                                        index_path,
                                        k=5
                                    )
                                    try:
                                        result_doc_ids, scores = faiss_query.query()
                                        print("\nText query results:")
                                        for i, (doc_id, score) in enumerate(zip(result_doc_ids, scores)):
                                            # Get the original document content
                                            try:
                                                doc_index = doc_ids.index(doc_id)
                                                doc_content = contexts[doc_index]
                                                # Extract the first 100 characters as a preview
                                                preview = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                                                print(f"[{i+1}] Document ID: {doc_id}, Similarity: {score:.4f}")
                                                print(f"     Preview: {preview}\n")
                                            except:
                                                print(f"[{i+1}] Document ID: {doc_id}, Similarity: {score:.4f}")
                                    except Exception as e:
                                        error(f"Query test failed: {str(e)}")
                                        traceback.print_exc()
                            else:
                                print("Invalid option, please select again")
                    except Exception as e:
                        error(f"Failed to initialize query test environment: {str(e)}")
                        traceback.print_exc()
        
        except KeyboardInterrupt:
            info("Detected user interruption, saving current results...")
            if all_doc_ids:
                save_path = save_intermediate_results(all_doc_ids, all_vectors, intermediate_dir)
                info(f"Intermediate results saved to: {save_path}")
        
        info("Vectorization and FAISS index creation process completed!")
        return True
        
    except Exception as e:
        error(f"Vectorization process failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function, run the corresponding mode based on the configuration"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Question Answering System")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and errors")
    parser.add_argument("--mode", type=str, choices=["process", "train", "evaluate", "webui", "vectorize"],
                       help="Running mode: process=data processing, train=model training, evaluate=model evaluation, webui=launch web interface, vectorize=document vectorization")
    args = parser.parse_args()
    
    # Initialize the logger
    initialize_logger()
    
    # Set the log level
    if args.verbose:
        set_log_level(LOG_LEVEL_DEBUG)
        log_message("Verbose logging mode enabled")
    elif args.quiet:
        set_log_level(LOG_LEVEL_WARNING)
        log_message("Silent logging mode enabled")
    else:
        set_log_level(LOG_LEVEL_INFO)
    
    # Check dependencies
    if not check_dependencies():
        log_message("Missing necessary dependencies, program cannot run")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get the running mode (prioritize command line arguments, then get the configuration from the debug section)
    run_mode = args.mode if args.mode else None
    
    # If the command line does not specify the mode, get it from the configuration file
    if not run_mode:
        debug_config = config.get("debug", {})
        process_mode = debug_config.get("process_mode", False)
        evaluate_mode = debug_config.get("evaluate_mode", False)
        train_mode = debug_config.get("train_mode", False)
        vectorize_mode = debug_config.get("vectorize_mode", False)
        rerank_mode = debug_config.get("rerank_mode", False)
    else:
        # Set the mode based on the command line arguments
        process_mode = (run_mode == "process")
        evaluate_mode = (run_mode == "evaluate")
        train_mode = (run_mode == "train")
        vectorize_mode = (run_mode == "vectorize")
        rerank_mode = (run_mode == "rerank")
    # Output detailed configuration information
    log_message(f"Configuration file path: {args.config}")
    debug(f"Complete configuration: {config}")
    log_message(f"Running mode: {'Command line specified: '+run_mode if run_mode else 'Configuration file specified'}")
    log_message(f"Running mode details: process={process_mode}, evaluate={evaluate_mode}, train={train_mode}, vectorize={vectorize_mode}, rerank={rerank_mode}")
    
    if train_mode:
        log_message("Running training mode...")
        
        # Get the current training module
        active_modules = config.get("modules", {}).get("active", {})
        retrieval_method = active_modules.get("retrieval", "hybrid")
        
        log_message(f"Current retrieval method: {retrieval_method}")
        
        # Check if bert_base configuration exists and is not empty
        bert_base_config = config.get("bert_base", {}).get("train", {})
        has_bert_base_config = bool(bert_base_config)
        
        # Check if debug.train_mode is explicitly specified as True
        explicitly_train_bert = config.get("debug", {}).get("train_mode", False)
        
        # Prioritize training bert_base (if the configuration exists and explicitly specifies the training mode)
        if has_bert_base_config and explicitly_train_bert:
            # Import BERT-Base training module
            log_message("Preparing to train BERT-Base model...")
            try:
                from bert_base.train import train_bert_base_model
                train_bert_base_model(config)
            except Exception as e:
                error(f"BERT-Base model training failed: {str(e)}")
                import traceback
                traceback.print_exc()
        elif retrieval_method == "bert_base":
            # If the retrieval method is specified as bert_base, also train bert_base
            log_message("Preparing to train BERT-Base model...")
            try:
                from bert_base.train import train_bert_base_model
                train_bert_base_model(config)
            except Exception as e:
                error(f"BERT-Base model training failed: {str(e)}")
                import traceback
                traceback.print_exc()
        elif retrieval_method == "hybrid" or retrieval_method == "bm25":
            # In other cases, continue using Compass to select the embedding method
            try:
                from dpr.compass import Compass
                # Create Compass instance, pass the complete configuration object instead of the configuration file path
                compass = Compass(config_path=args.config)
                method = compass.choose_embedding_method()
                if method == "bge_m3":
                    log_message("BGE-M3 does not need training, please enable vectorize_mode for vectorization")
                elif method == "dpr":
                    log_message("Using pre-trained DPR model, no training required")
                elif method == "train_dpr":
                    log_message("DPR model training is not yet implemented")
            except Exception as e:
                error(f"Error selecting embedding method: {str(e)}")
                import traceback
                traceback.print_exc()
    elif process_mode:
        log_message("Running processing mode...")
        try:
            from process.process import DocumentPipeline
            
            # Get file path
            input_dir = config.get("data", {}).get("input_file", "./data/origin_data/")
            if not input_dir.endswith("/"):
                input_dir += "/"
            
            input_path = os.path.abspath(os.path.join(input_dir, "documents.jsonl"))
            
            model_output_dir = config.get("data", {}).get("model_output_dir", "./model/pkl")
            if not model_output_dir.endswith("/"):
                model_output_dir += "/"
            model_output_dir = os.path.abspath(model_output_dir)
            
            data_output_dir = config.get("data", {}).get("data_output_dir", "./data/processed_data")
            if not data_output_dir.endswith("/"):
                data_output_dir += "/"
            data_output_dir = os.path.abspath(data_output_dir)
            
            # Ensure the output directories exist
            os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)
            os.makedirs(os.path.dirname(data_output_dir), exist_ok=True)
            
            # Print path information
            log_message(f"Input file path: {input_path}")
            log_message(f"Model output directory: {model_output_dir}")
            log_message(f"Data output directory: {data_output_dir}")
            
            # Create DocumentPipeline instance
            pipeline = DocumentPipeline(
                input_path=input_path, 
                model_output_dir=model_output_dir, 
                data_output_dir=data_output_dir,
                config=config  # Pass the complete configuration
            )
            
            # Call the run method of the instance
            pipeline.run()
            
        except Exception as e:
            log_message(f"Processing mode failed: {str(e)}")
            print(f"Processing mode failed: {str(e)}")
            raise
    elif evaluate_mode:
        log_message("Running evaluation mode...")
        from eval.eval import Compass
        compass = Compass(args.config)
        compass.run_evaluation()
    elif vectorize_mode:
        log_message("Running document vectorization mode...")
        vectorize_with_bgem3(config)
    elif rerank_mode:
        log_message("Running reranking mode...")
        from eval.combine import calculate_score
        from bm25.base import ManualQuerySearch
        from dpr.jsonler import query_jsonler
        from webui.base import WebUI
        
        # Get the validation set path from the configuration file or parameters
        test_path = config.get("data", {}).get("test_path", "./data/original_data/test.jsonl")
        log_message(f"Loading query data: {test_path}")
        
        # Use query_jsonler to load all queries
        query_loader = query_jsonler(test_path)
        queries = query_loader.get_json_data()
        log_message(f"Successfully loaded {len(queries)} queries")
        
        for query in queries:
            txt_doc_id = ManualQuerySearch(query).search()["document_id"]
            dpr_doc_id = None
            query, all_doc_id = calculate_score(query, txt_doc_id, dpr_doc_id)
            rerank_doc_id = WebUI().rerank_results(query, all_doc_id, top_n=5)
            doc_content = WebUI().retrieve_doc(rerank_doc_id)
            answer = WebUI().generate_answer(query, doc_content, n=5)
            result = {
                "question": query 
            }
    else:
        log_message("Running Web UI mode...")
        try:
            from webui.base import create_webui
            log_message("Successfully imported WebUI module...")
            
            # Set environment variables to avoid potential thread issues
            os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Start WebUI
            log_message("Starting WebUI, this may take some time...")
            success = create_webui(args.config)
            
            if not success:
                log_message("WebUI failed to start, please check the logs for details")
                sys.exit(1)
            
        except KeyboardInterrupt:
            log_message("User interrupted, exiting program")
            sys.exit(0)
        except Exception as e:
            error(f"Error starting WebUI: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
if __name__ == "__main__":
    main()
