from .process import DocumentProcessor, PlainTextProcessor, DocumentIndexer, DocumentPipeline

__all__ = ['DocumentProcessor', 'PlainTextProcessor', 'DocumentIndexer', 'DocumentPipeline', 'process_documents']

def process_documents(config):
    """
    Main function for processing documents
    
    Parameters:
        config: Configuration object
    """
    # Fix file paths, ensure paths exist and are correct
    # Get file paths from configuration, ensure paths end with /
    input_dir = config.get("data", {}).get("input_file", "./data/origin_data/")
    if not input_dir.endswith("/"):
        input_dir += "/"
    
    # Use absolute paths and ensure directories exist
    import os
    input_path = os.path.abspath(os.path.join(input_dir, "documents.jsonl"))
    
    # Output directory path processing
    model_output_dir = config.get("data", {}).get("model_output_dir", "./model/pkl")
    if not model_output_dir.endswith("/"):
        model_output_dir += "/"
    model_output_dir = os.path.abspath(model_output_dir)
    
    data_output_dir = config.get("data", {}).get("data_output_dir", "./data/processed_data")
    if not data_output_dir.endswith("/"):
        data_output_dir += "/"
    data_output_dir = os.path.abspath(data_output_dir)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(data_output_dir), exist_ok=True)
    
    # Import logging functions
    from utils import log_message
    
    # Print path information
    log_message(f"Input file path: {input_path}")
    log_message(f"Model output directory: {model_output_dir}")
    log_message(f"Data output directory: {data_output_dir}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        error_msg = f"Error: Input file {input_path} does not exist"
        log_message(error_msg)
        print(error_msg)
        import sys
        sys.exit(1)
    
    log_message("Successfully imported DocumentPipeline...")
    
    # Create DocumentPipeline instance
    pipeline = DocumentPipeline(
        input_path=input_path, 
        model_output_dir=model_output_dir, 
        data_output_dir=data_output_dir
    )
    log_message(f"Successfully initialized DocumentPipeline..., current parameters: input_path={input_path}, model_output_dir={model_output_dir}, data_output_dir={data_output_dir}")
    
    # Call the run method of the instance
    pipeline.run()
    
    log_message(f"Conventional statistical method models training completed, models saved to: {model_output_dir}, processed data saved to: {data_output_dir}")
    
    return {
        "input_path": input_path,
        "model_output_dir": model_output_dir,
        "data_output_dir": data_output_dir
    }

