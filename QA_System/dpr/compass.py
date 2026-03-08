import os
import time
import yaml
from utils import info, debug, warning, error, critical

class Compass:
    """Embedding method selector, used to select different embedding methods in training/vectorization mode"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the embedding method selector
        
        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        info(f"[Compass] Initializing embedding method selector, config path: {config_path}")
        
        start_time = time.time()
        self.config = self.load_config(config_path)
        self.retrieval_config = self.config.get("retrieval", {})
        self.eval_config = self.config.get("evaluation", {})
        
        debug(f"[Compass] Configuration loaded, time taken: {time.time() - start_time:.2f} seconds")
        info(f"[Compass] Retrieval configuration: {', '.join([f'{k}={v}' for k, v in self.retrieval_config.items() if not isinstance(v, dict)])}")
    
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
    
    def choose_embedding_method(self):
        """Prompt user to select embedding method"""
        
        debug(f"[Compass] Select Embedding method")
        
        # Prompt user to select method
        info(f"[Compass] Displaying Embedding method selection menu")
        print("\n" + "="*50)
        print("Please select Embedding method:")
        print("[1] API Embedding (BGE-M3)")
        print("[2] Local Embedding (DPR pre-trained model)")
        print("[3] Train DPR model")
        print("="*50)
        
        choice = input("Please enter option number [1/2/3]: ")
        debug(f"[Compass] User input option: {choice}")
        
        if choice == "1":
            method = "bge_m3"
        elif choice == "2":
            method = "dpr"
        elif choice == "3":
            method = "train_dpr"
        else:
            warning(f"[Compass] Invalid option: {choice}, please select again")
            return self.choose_embedding_method()
        
        info(f"[Compass] Selected Embedding method: {method}")
        return method