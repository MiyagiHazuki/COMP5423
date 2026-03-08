import os
import torch
import logging
import random
import numpy as np
# Import Windows platform patch
from bert_base.patch_windows import *
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from bert_base.model import DPRModel
from bert_base.dataset import create_dataloader

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dpr_data(config):
    """
    Prepare data needed for DPR training
    
    Args:
        config: Configuration dictionary
    
    Returns:
        bool: Whether data preparation was successful
    """
    # Import data processor
    try:
        from bert_base.dpr_processor import DocumentPipeline, DPRDataProcessor
    except ImportError:
        logging.error("Unable to import data processing module, please ensure bert_base.dpr_processor module is installed")
        return False
    
    # Get data path
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    if not data_output_dir.endswith("/"):
        data_output_dir += "/"
    
    # Ensure path exists
    os.makedirs(data_output_dir, exist_ok=True)
    
    # Check if processed text files exist
    processed_file = os.path.join(data_output_dir, "processed_plain.jsonl")
    
    # If processed files don't exist, run data processing workflow
    if not os.path.exists(processed_file):
        logging.info("Missing processed files, please run data processing workflow first")
        logging.info("You can use the following command: python main.py --mode process")
        return False
    
    # Get DPR training parameters
    dpr_config = config.get("bert_base", {}).get("train", {})
    chunk_size = dpr_config.get("max_ctx_length", 2048)
    chunk_overlap = dpr_config.get("chunk_overlap", 100)
    n_negatives = dpr_config.get("n_negatives", 3)
    max_workers = dpr_config.get("num_workers", 4)
    
    # Prepare DPR training data
    logging.info("Starting to prepare DPR training data...")
    
    try:
        # Create DPR data processor
        processor = DPRDataProcessor(data_dir=os.path.dirname(data_output_dir))
        
        # Execute complete data processing workflow
        logging.info("Loading documents...")
        processor.load_documents()
        
        logging.info("Loading QA data...")
        processor.load_qa_data()
        
        logging.info("Chunking documents...")
        processor.chunk_documents(chunk_size=chunk_size, overlap=chunk_overlap)
        
        logging.info("Building TF-IDF index...")
        processor.build_tfidf_index()
        
        logging.info("Preparing DPR training data...")
        train_examples, val_examples = processor.prepare_dpr_training_data(
            n_negatives=n_negatives,
            max_workers=max_workers
        )
        
        logging.info("Saving DPR training data...")
        processor.save_dpr_data(train_examples, val_examples)
        
        logging.info("DPR training data preparation complete")
        return True
    
    except Exception as e:
        logging.error(f"Error preparing DPR training data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def train(config):
    """Train DPR model"""
    # Get training configuration
    train_config = config.get("bert_base", {}).get("train", {})
    
    # Set random seed
    seed = config.get("system", {}).get("seed", 42)
    set_seed(seed)
    
    # Get path configuration
    paths_config = config.get("paths", {})
    model_root = paths_config.get("model_root", "./models/")
    cache_dir = paths_config.get("cache_dir", "./cache/")
    
    # Create output directory
    output_dir = os.path.join(model_root, "bert_base")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data path
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    
    # Data file paths
    train_file = os.path.join(data_output_dir, "dpr_train.json")
    val_file = os.path.join(data_output_dir, "dpr_val.json")
    
    # Check if training and validation files exist
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        logging.error(f"Training data files don't exist: {train_file} or {val_file}")
        logging.info("Please use --mode process for data processing first, or ensure DPR training data has been generated")
        return 0, 0
    
    # Get model configuration
    model_name_or_path = train_config.get("model_name_or_path", "nomic-ai/nomic-bert-2048")
    shared_weights = train_config.get("shared_weights", False)
    temperature = train_config.get("temperature", 0.05)
    
    # Training parameters
    num_train_epochs = train_config.get("num_train_epochs", 5)
    batch_size = train_config.get("batch_size", 2)
    learning_rate = train_config.get("learning_rate", 3e-5)
    weight_decay = train_config.get("weight_decay", 0.01)
    max_query_length = train_config.get("max_query_length", 128)
    max_ctx_length = train_config.get("max_ctx_length", 2048)
    warmup_steps = train_config.get("warmup_steps", 0)
    adam_epsilon = train_config.get("adam_epsilon", 1e-8)
    eval_steps = train_config.get("eval_steps", 0)
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 2)
    num_workers = train_config.get("num_workers", 4)
    
    # Device
    device_name = config.get("system", {}).get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, cache_dir=cache_dir)
    
    # Load data
    logging.info("Creating training data loader...")
    train_dataloader = create_dataloader(
        data_file=train_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=True,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Load validation data
    logging.info("Creating validation data loader...")
    val_dataloader = create_dataloader(
        data_file=val_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=True,  # Still need negative samples to calculate contrastive loss
        shuffle=False,
        num_workers=num_workers
    )
    
    # Create model
    logging.info(f"Creating DPR model using pretrained model: {model_name_or_path}")
    model = DPRModel(
        query_encoder_name=model_name_or_path,
        ctx_encoder_name=model_name_or_path,
        shared_weights=shared_weights,
        temperature=temperature
    )
    model.to(device)
    
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Calculate total training steps
    if gradient_accumulation_steps > 1:
        # Consider gradient accumulation
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    else:
        t_total = len(train_dataloader) * num_train_epochs
    
    # Create optimizer and learning rate scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total
    )
    
    # Whether to use mixed precision training
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    logging.info("***** Start Training *****")
    logging.info(f"  Samples per batch = {batch_size}")
    logging.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
    logging.info(f"  Total training steps = {t_total}")
    logging.info(f"  Maximum context length = {max_ctx_length}")
    if eval_steps > 0:
        logging.info(f"  Evaluation every {eval_steps} steps")
    else:
        logging.info(f"  Evaluation only at the end of each epoch")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_train_epochs):
        logging.info(f"Starting Epoch {epoch+1}/{num_train_epochs}")
        model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move data to device
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # Add debug logging, output some batch information for tracking
            if step == 0:
                logging.info(f"Batch shape: query_input_ids={batch['query_input_ids'].shape}, "
                           f"pos_ctx_input_ids={batch['pos_ctx_input_ids'].shape}, "
                           f"neg_ctx_input_ids={batch['neg_ctx_input_ids'].shape}")
            
            # Use mixed precision training
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    # Forward propagation
                    query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                        query_input_ids=batch["query_input_ids"],
                        query_attention_mask=batch["query_attention_mask"],
                        query_token_type_ids=batch["query_token_type_ids"],
                        pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                        pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                        pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                        neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                        neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                        neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
                    )
                    
                    # Calculate loss
                    loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
                    
                    # Gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward propagation
                scaler.scale(loss).backward()
                
                # Update parameters every specified step
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # Forward propagation
                query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                    query_token_type_ids=batch["query_token_type_ids"],
                    pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                    pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                    pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                    neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                    neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                    neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
                )
                
                # Calculate loss
                loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
                
                # Gradient accumulation
                loss = loss / gradient_accumulation_steps
            
                # Backward propagation
                loss.backward()
                
                # Update parameters every specified step
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Evaluate every eval_steps steps
            if eval_steps > 0 and global_step > 0 and global_step % eval_steps == 0:
                val_loss = evaluate(config, model, val_dataloader, device)
                
                # If validation loss is better, save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logging.info(f"Saving best model, validation loss: {val_loss:.4f}")
                    save_model(config, model, tokenizer, f"best_model_step_{global_step}")
                
                model.train()  # Go back to training mode
        
        # Handle gradient for the last batch
        if (step + 1) % gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluate at the end of each epoch
        if eval_steps <= 0:  # Only evaluate at the end of the epoch if not doing intermediate evaluation
            logging.info("Performing validation...")
            val_loss = evaluate(config, model, val_dataloader, device)
            
            # If validation loss is better, save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"Saving best model, validation loss: {val_loss:.4f}")
                save_model(config, model, tokenizer, "best_model")
        
        # Save model for each epoch
        logging.info(f"Saving Epoch {epoch+1} model")
        save_model(config, model, tokenizer, f"epoch_{epoch+1}")
    
    # Save final model
    logging.info("Saving final model")
    save_model(config, model, tokenizer, "final_model")
    
    return global_step, best_val_loss

def evaluate(config, model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move data to device
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            
            # Forward propagation
            query_vectors, pos_ctx_vectors, neg_ctx_vectors = model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                query_token_type_ids=batch["query_token_type_ids"],
                pos_ctx_input_ids=batch["pos_ctx_input_ids"],
                pos_ctx_attention_mask=batch["pos_ctx_attention_mask"],
                pos_ctx_token_type_ids=batch["pos_ctx_token_type_ids"],
                neg_ctx_input_ids=batch["neg_ctx_input_ids"],
                neg_ctx_attention_mask=batch["neg_ctx_attention_mask"],
                neg_ctx_token_type_ids=batch["neg_ctx_token_type_ids"]
            )
            
            # Calculate loss
            loss = model.compute_loss(query_vectors, pos_ctx_vectors, neg_ctx_vectors)
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    logging.info(f"Validation loss: {avg_loss:.4f}")
    
    return avg_loss

def save_model(config, model, tokenizer, prefix):
    """Save model and tokenizer"""
    paths_config = config.get("paths", {})
    model_root = paths_config.get("model_root", "./models/")
    
    output_dir = os.path.join(model_root, "bert_base", prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save encoder
    model.save_encoders(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training configuration
    train_config = config.get("bert_base", {}).get("train", {})
    with open(os.path.join(output_dir, "training_config.txt"), "w") as f:
        for key, value in train_config.items():
            f.write(f"{key}: {value}\n")

def train_bert_base_model(config):
    """Main function to start BERT-Base model training"""
    train_mode = config.get("debug", {}).get("train_mode", False)
    
    if not train_mode:
        logging.info("Training mode not enabled, skipping BERT-Base model training")
        return
    
    # Prepare data
    data_config = config.get("data", {})
    data_output_dir = data_config.get("data_output_dir", "./data/processed_data/")
    train_file = os.path.join(data_output_dir, "dpr_train.json")
    val_file = os.path.join(data_output_dir, "dpr_val.json")
    
    # Check if training files exist
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        logging.info("DPR training data doesn't exist, attempting automatic preparation...")
        success = prepare_dpr_data(config)
        if not success:
            logging.error("Unable to automatically prepare DPR training data, please run data processing steps first")
            logging.info("You can use the following command: python main.py --mode process")
            logging.info("Then ensure the processed_plain.jsonl file has been generated")
            logging.info("Or manually create dpr_train.json and dpr_val.json files")
            return
    
    logging.info("Starting BERT-Base model training...")
    global_step, best_val_loss = train(config)
    logging.info(f"Training completed, total steps: {global_step}, best validation loss: {best_val_loss:.4f}")