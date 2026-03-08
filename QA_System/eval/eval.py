import os
import sys
import yaml
import time
import json
from utils import info, debug, warning, error, critical

class Compass:
    """Evaluation Direction Controller, used to select different evaluation methods"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the evaluation controller
        
        Parameters:
            config_path: Configuration file path
        """
        self.config_path = config_path
        info(f"[Compass] Initializing evaluation controller, configuration file path: {config_path}")
        
        start_time = time.time()
        self.config = self.load_config(config_path)
        self.eval_config = self.config.get("evaluation", {})
        self.retrieval_config = self.config.get("retrieval", {})
        
        debug(f"[Compass] Configuration loaded, time elapsed: {time.time() - start_time:.2f} seconds")
        info(f"[Compass] Evaluation configuration: {', '.join([f'{k}={v}' for k, v in self.eval_config.items() if not isinstance(v, dict)])}")
    
    def load_config(self, config_path):
        """Load configuration file"""
        debug(f"[Compass] Starting to load configuration file: {config_path}")
        if not os.path.exists(config_path):
            error_msg = f"Configuration file does not exist: {config_path}"
            error(f"[Compass] {error_msg}")
            raise FileNotFoundError(error_msg)
            
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            debug(f"[Compass] Configuration file loaded successfully: {config_path}")
            return config
        except Exception as e:
            error_msg = f"Failed to load configuration file: {str(e)}"
            error(f"[Compass] {error_msg}")
            raise
    
    def choose_eval_method(self):
        """Prompt user to select evaluation method or use default method"""
        # Check if automatically select default method
        auto_select = self.eval_config.get("auto_select", False)
        default_method = self.eval_config.get("default_method", "text")
        
        debug(f"[Compass] Select evaluation method - Auto select: {auto_select}, Default method: {default_method}")
        
        if auto_select:
            info(f"[Compass] Automatically select evaluation method: {default_method}")
            return default_method
        
        # Prompt user to choose method
        info(f"[Compass] Display evaluation method selection menu")
        print("\n" + "="*50)
        print("Please select evaluation method:")
        print("[1] Text Retrieval Evaluation (BM25/TF-IDF)")
        print("[2] Deep Embedding Evaluation (DPR)")
        print("[3] Hybrid Retrieval Evaluation (Hybrid)")
        print("="*50)
        
        choice = input("Please enter option number [1/2/3]: ")
        debug(f"[Compass] User input option: {choice}")
        
        if choice == "1":
            method = "text"
        elif choice == "2":
            method = "deep_embedding"
        elif choice == "3":
            method = "hybrid"
        else:
            warning(f"[Compass] Invalid option: {choice}, using default method: {default_method}")
            print(f"Invalid option: {choice}, using default method: {default_method}")
            method = default_method
        
        info(f"[Compass] Selected evaluation method: {method}")
        return method
    
    def get_retrieval_params(self, method_type):
        """
        Get parameter configuration for specific retrieval method
        
        Parameters:
            method_type: Retrieval method type (text/deep_embedding/hybrid)
            
        Returns:
            Retrieval method parameters dictionary
        """
        debug(f"[Compass] Get {method_type} retrieval method parameters")
        
        # Get common retrieval parameters
        params = {
            "top_k": self.retrieval_config.get("top_k", 5),
            "max_words": self.retrieval_config.get("max_words", 50),
            "use_plain_text": self.retrieval_config.get("use_plain_text", True)
        }
        
        # Get specific parameters based on retrieval type
        if method_type == "text":
            text_config = self.retrieval_config.get("text_retrieval", {})
            params["method"] = text_config.get("method", "hybrid")
            params["hybrid_alpha"] = text_config.get("hybrid_alpha", 0.7)
            info(f"[Compass] Loaded text retrieval parameters - Method: {params['method']}, Hybrid ratio: {params['hybrid_alpha']}")
            
        elif method_type == "deep_embedding":
            deep_config = self.retrieval_config.get("deep_retrieval", {})
            m3_config = deep_config.get("api_embedding", {})
            params["model"] = m3_config.get("model", "BAAI/bge-m3")
            params["api_key"] = m3_config.get("api_key", "sk-proj-1234567890")
            params["doc_path"] = m3_config.get("doc_path", "./data/processed_data/processed_plain.jsonl")
            params["intermediate_path"] = m3_config.get("intermediate_path", "./cache/bge_m3_intermediate")
        elif method_type == "hybrid":
            print("Needs further improvement")
            pass
        
        debug(f"[Compass] Common parameters - top_k: {params['top_k']}, max_words: {params['max_words']}, use_plain_text: {params['use_plain_text']}")
        return params
    
    def evaluate_text_retrieval(self):
        """Evaluate text retrieval method (BM25/TF-IDF)"""
        info(f"[Compass] Starting text retrieval evaluation")
        start_time = time.time()
        
        # Get text retrieval parameters
        text_params = self.get_retrieval_params("text")
        
        info(f"[Compass] Using {text_params['method']} method for text retrieval evaluation")
        
        # Process file paths
        data_output_dir = self.config.get("data", {}).get("data_output_dir", "./data/processed_data/")
        model_output_dir = self.config.get("data", {}).get("model_output_dir", "./model/pkl/")
        val_path = self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl")
        
        debug(f"[Compass] Path configuration - Data output directory: {data_output_dir}")
        debug(f"[Compass] Path configuration - Model output directory: {model_output_dir}")
        debug(f"[Compass] Path configuration - Validation set path: {val_path}")
        
        # Ensure paths end with /
        if not data_output_dir.endswith("/"):
            data_output_dir += "/"
        if not model_output_dir.endswith("/"):
            model_output_dir += "/"
            
        # Convert to absolute paths
        data_output_dir = os.path.abspath(data_output_dir)
        model_output_dir = os.path.abspath(model_output_dir)
        val_path = os.path.abspath(val_path)
        
        debug(f"[Compass] Absolute path - Data output directory: {data_output_dir}")
        debug(f"[Compass] Absolute path - Model output directory: {model_output_dir}")
        debug(f"[Compass] Absolute path - Validation set path: {val_path}")
        
        # Decide which processed document to use based on use_plain_text
        use_plain_text = text_params.get("use_plain_text", True)
        doc_type = "plain" if use_plain_text else "markdown"
        doc_path = os.path.join(data_output_dir, f"processed_{doc_type}.jsonl")
        bm25_path = os.path.join(model_output_dir, f"{doc_type}_bm25.pkl")
        tfidf_path = os.path.join(model_output_dir, f"{doc_type}_tfidf.pkl")
        
        info(f"[Compass] Using {'plain text' if use_plain_text else 'Markdown'} format documents")
        debug(f"[Compass] Document path: {doc_path}")
        debug(f"[Compass] BM25 model path: {bm25_path}")
        debug(f"[Compass] TF-IDF model path: {tfidf_path}")
        
        # Check if necessary files exist
        required_files = [doc_path, bm25_path, tfidf_path, val_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            error(f"[Compass] Missing necessary files:")
            for file in missing_files:
                error(f"[Compass] - File does not exist: {file}")
            critical("[Compass] Please run process mode first to generate necessary model and data files")
            print("Error: Missing necessary files, please check logs for details")
            sys.exit(1)
        else:
            debug(f"[Compass] All necessary files check passed")
        
        # Get output path
        output_path = self.eval_config.get("output", {}).get("text", 
                          os.path.join(data_output_dir, f"text_evaluation_results.jsonl"))
        
        # Check if output file already exists
        if os.path.exists(output_path):
            info(f"[Compass] Found existing evaluation results: {output_path}")
            
            # Check if file has content
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                if first_line:
                    info(f"[Compass] Skip evaluation process, directly use existing results to calculate metrics")
                    # Jump directly to metrics calculation
                    calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
                    if calculate_metrics:
                        self._calculate_metrics(output_path)
                    
                    info(f"[Compass] Text retrieval evaluation process completed")
                    return output_path
                else:
                    info(f"[Compass] Existing result file is empty, will re-execute evaluation process")
            except Exception as e:
                warning(f"[Compass] Error checking existing result file: {str(e)}, will re-execute evaluation process")
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            debug(f"[Compass] Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        info(f"[Compass] Evaluation results will be saved to: {output_path}")
        
        # Build evaluator configuration
        eval_config = {
            "method": text_params.get("method", "hybrid"),
            "hybrid_alpha": text_params.get("hybrid_alpha", 0.7),
            "top_k": text_params.get("top_k", 5),
            "max_words": text_params.get("max_words", 2),
            "doc_path": doc_path,
            "bm25_path": bm25_path,
            "tfidf_path": tfidf_path,
            "val_path": val_path,
            "output_path": output_path
        }
        
        debug(f"[Compass] Evaluator configuration prepared")
        
        # Use custom evaluation method to directly generate format compatible with metrics_calculation
        try:
            info(f"[Compass] Starting execution of evaluation process")
            output_path = self._run_batch_evaluation_with_format(eval_config)
            info(f"[Compass] Evaluation completed, time elapsed: {time.time() - start_time:.2f} seconds")
            info(f"[Compass] Evaluation results saved to: {output_path}")
        except Exception as e:
            error(f"[Compass] Error in evaluation process: {str(e)}")
            print(f"Error in evaluation process: {str(e)}")
            raise
        
        # If configured to calculate evaluation metrics
        calculate_metrics = self.eval_config.get("metrics", {}).get("calculate", False)
        if calculate_metrics:
            self._calculate_metrics(output_path)
        
        info(f"[Compass] Text retrieval evaluation process completed")
        return output_path
        
    def _run_batch_evaluation_with_format(self, config):
        """
        Run batch evaluation and directly output results in format compatible with metrics_calculation
        
        Parameters:
            config: Evaluation configuration
            
        Returns:
            Output file path
        """
        debug(f"[Compass] Starting batch evaluation, using output format compatible with metrics_calculation")
        
        # Import BatchEvaluator class but not directly use its evaluate method
        from bm25.base import BatchEvaluator, BaseSearchEngine
        
        # Create custom evaluator
        evaluator = BatchEvaluator(config)
        max_words = config.get("max_words", 2)
        
        # Load documents
        doc_ids, docs = evaluator.load_documents(config["doc_path"])
        info(f"[Compass] Successfully loaded {len(doc_ids)} documents")
        
        # Load BM25 and TF-IDF models
        bm25 = evaluator.load_bm25(config["bm25_path"])
        tfidf_vectorizer, _ = evaluator.load_tfidf(config["tfidf_path"])
        tfidf_matrix = tfidf_vectorizer.transform(docs)
        
        # Load validation set questions
        questions = evaluator.load_questions(config["val_path"])
        info(f"[Compass] Successfully loaded {len(questions)} questions")
        
        # Store evaluation results (using format required by metrics_calculation)
        results = []
        
        # Evaluate each question
        start_time = time.time()
        for i, question_item in enumerate(questions):
            # Print progress every 10%
            if (i+1) % max(1, len(questions)//10) == 0 or i+1 == len(questions):
                progress = (i+1) / len(questions) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i+1)) * (len(questions) - i - 1) if i > 0 else 0
                info(f"[Compass] Progress: {progress:.1f}% ({i+1}/{len(questions)}), Time elapsed: {elapsed:.1f} seconds, Estimated remaining: {eta:.1f} seconds")
            
            # Get query text from question item
            query = question_item["question"]
            
            # Predict documents
            pred_doc_ids, _ = evaluator.predict_top_document(
                query, doc_ids, docs, tfidf_vectorizer, tfidf_matrix, bm25,
                method=config["method"], alpha=config["hybrid_alpha"], top_k=config["top_k"],
                return_format="lists"
            )
            
            # If there are prediction results, extract answer from first document; otherwise use empty string
            answer = ""
            if pred_doc_ids and len(pred_doc_ids) > 0:
                doc_idx = doc_ids.index(pred_doc_ids[0])
                doc_text = docs[doc_idx]
                answer = BaseSearchEngine.extract_answer_from_doc(doc_text, max_words=max_words)
            
            # Record results in format expected by metrics_calculation
            result = {
                "question": query,
                "answer": answer,
                "document_id": pred_doc_ids  # List type
            }
            results.append(result)
        
        # Save results
        output_path = config["output_path"]
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        info(f"[Compass] Evaluation results saved, compatible with metrics_calculation format")
        return output_path
    
    def evaluate_deep_embedding(self):
        """Evaluate deep embedding retrieval method"""
        retrieval_config = self.config.get("retrieval", {})
        m3_config = retrieval_config.get("deep_retrieval", {}).get("bgem3", {})
        import pandas as pd
        query_path = m3_config.get("result_path", "./data/result/val_query_vectors.jsonl")
        combine = pd.read_json(query_path, lines=True)
        question = combine.question.tolist()
        vector = combine.vector.tolist()
        from faiss_composer.base import FaissQuery
        faiss_path = m3_config.get("faiss_path", "./cache/faiss/bgem3.faiss")
        top_k = retrieval_config.get("top_k", 5)

        eval_config = retrieval_config.get("evaluation", {})
        eval_path = eval_config.get("output", {}).get("deep_embedding", "./data/evaluation/deep_embedding_evaluation_results.jsonl")
        for i in range(len(question)):
            faiss_query = FaissQuery([vector[i]], top_k=top_k, faiss_path=faiss_path)
            doc_ids, scores = faiss_query.query()
            for doc_id in doc_ids:
                doc_ids[doc_ids.index(doc_id)] = int(doc_id)
            dict = {
                "question": question[i],
                "answer": "Placeholder",
                "document_id": doc_ids,
            }
            with open(eval_path, "a") as f: 
                json.dump(dict, f)
                f.write("\n")
    
    def evaluate_hybrid(self):
        """Evaluate hybrid retrieval method"""
        info(f"[Compass] Hybrid retrieval evaluation feature has been removed")
        print("Hybrid retrieval evaluation feature has been removed, please contact system administrator for more information.")
        return None
    
    def run_evaluation(self):
        """Run evaluation process"""
        info(f"[Compass] ====== Starting evaluation process ======")
        start_time = time.time()
        
        # Let user choose evaluation method
        eval_method = self.choose_eval_method()
        info(f"[Compass] Selected evaluation method: {eval_method}")
        
        # Execute corresponding evaluation based on selected method
        result = None
        try:
            if eval_method == "text":
                result = self.evaluate_text_retrieval()
            elif eval_method == "deep_embedding":
                result = self.evaluate_deep_embedding()
            elif eval_method == "hybrid":
                result = self.evaluate_hybrid()
            else:
                warning(f"[Compass] Unknown evaluation method: {eval_method}")
                print(f"Unknown evaluation method: {eval_method}")
                return None
        except Exception as e:
            error(f"[Compass] Error in evaluation process: {str(e)}")
            print(f"Error in evaluation process: {str(e)}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            info(f"[Compass] ====== Evaluation process ended, total time: {elapsed_time:.2f} seconds ======")
        
        return result

    def _calculate_metrics(self, output_path):
        """
        Calculate and save evaluation metrics
        
        Parameters:
            output_path: Evaluation results path
            
        Returns:
            metrics_results: Dictionary of calculated metric results
        """
        info(f"[Compass] Start calculating evaluation metrics")
        
        try:
            # Import metrics calculation module
            from eval.metrics_calculation import calculate_metrics as calc_metrics
            
            # Get validation set path and prediction results path
            val_path = self.config.get("data", {}).get("val_path", "./data/original_data/val.jsonl")
            
            # Get metric types
            metric_types = self.eval_config.get("metrics", {}).get("types", ["recall@5", "mrr@5"])
            info(f"[Compass] Calculate the following evaluation metrics: {', '.join(metric_types)}")
            
            info(f"[Compass] Using validation set: {val_path}")
            info(f"[Compass] Using prediction results: {output_path}")
            
            # Calculate evaluation metrics
            metrics_results = calc_metrics(val_path, output_path)
            
            # Only keep document retrieval related metrics
            doc_metrics = {
                'recall@5': metrics_results['recall@5'],
                'mrr@5': metrics_results['mrr@5']
            }
            
            # Print evaluation results
            info(f"[Compass] Document retrieval evaluation metrics calculation completed")
            for metric_name, value in doc_metrics.items():
                info(f"[Compass] {metric_name}: {value:.4f}")
            
            # Save evaluation results to file
            metrics_output_path = self.eval_config.get("metrics", {}).get("output_path", 
                              os.path.join(os.path.dirname(output_path), "retrieval_metrics.json"))
            
            with open(metrics_output_path, "w", encoding="utf-8") as f:
                json.dump(doc_metrics, f, ensure_ascii=False, indent=2)
            
            info(f"[Compass] Evaluation metrics saved to: {metrics_output_path}")
            return doc_metrics
            
        except Exception as e:
            error(f"[Compass] Error calculating evaluation metrics: {str(e)}")
            print(f"Error calculating evaluation metrics: {str(e)}")
            return None


# Main function for testing
if __name__ == "__main__":
    info("Starting independent run of evaluation module")
    try:
        compass = Compass()
        result = compass.run_evaluation()
        if result:
            print(f"Evaluation completed, results saved to: {result}")
        else:
            print("Evaluation not completed or no results generated")
    except Exception as e:
        error(f"Error occurred during evaluation: {str(e)}")
        print(f"Error occurred during evaluation: {str(e)}")