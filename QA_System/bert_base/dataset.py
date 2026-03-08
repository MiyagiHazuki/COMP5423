import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer

class DPRDataset(Dataset):
    """
    Dataset for DPR model
    """
    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        max_query_length: int = 128,
        max_ctx_length: int = 2048,
        is_training: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_file: Data file path
            tokenizer: Tokenizer
            max_query_length: Maximum length of query text
            max_ctx_length: Maximum length of context text (utilizing nomic-bert-2048's long sequence capability)
            is_training: Whether in training mode
        """
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_ctx_length = max_ctx_length
        self.is_training = is_training
        
        # Try two ways to load data
        try:
            # First try to load as regular JSON
            with open(data_file, 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
                
            # If it's a single JSON object, convert it to a list
            if isinstance(json_obj, dict):
                self.data = [json_obj]
            else:
                self.data = json_obj
        except json.JSONDecodeError:
            # If that fails, try loading as JSONL format line by line
            self.data = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except json.JSONDecodeError:
                            continue
            
        print(f"Loaded {len(self.data)} records")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get data item
        
        Args:
            idx: Data index
            
        Returns:
            Dictionary containing model inputs
        """
        item = self.data[idx]
        
        # Get query, positive sample and negative sample texts
        query = item["question"]
        
        # Compatible with two formats: positive_ctx (temp version) and positive_ctxs (original version)
        if "positive_ctx" in item:
            # temp version format
            positive_ctx = item["positive_ctx"]
        elif "positive_ctxs" in item and isinstance(item["positive_ctxs"], list) and len(item["positive_ctxs"]) > 0:
            # Original format, but take the text of the first positive sample
            if isinstance(item["positive_ctxs"][0], dict) and "text" in item["positive_ctxs"][0]:
                positive_ctx = item["positive_ctxs"][0]["text"]
            else:
                positive_ctx = item["positive_ctxs"][0]
        else:
            # Fallback, use empty string
            positive_ctx = ""
            
        # Get negative samples, compatible with two formats
        negative_ctxs = []
        if "negative_ctxs" in item:
            if isinstance(item["negative_ctxs"], list):
                if len(item["negative_ctxs"]) > 0:
                    if isinstance(item["negative_ctxs"][0], dict) and "text" in item["negative_ctxs"][0]:
                        # Original format, need to extract text
                        negative_ctxs = [neg["text"] for neg in item["negative_ctxs"]]
                    else:
                        # temp version, directly text list
                        negative_ctxs = item["negative_ctxs"]
        
        # Encode query text
        query_encoding = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode positive sample text
        pos_ctx_encoding = self.tokenizer(
            positive_ctx,
            max_length=self.max_ctx_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension (DataLoader will add batch dimension)
        query_encoding = {k: v.squeeze(0) for k, v in query_encoding.items()}
        pos_ctx_encoding = {k: v.squeeze(0) for k, v in pos_ctx_encoding.items()}
        
        # Build input dictionary
        inputs = {
            "query_input_ids": query_encoding["input_ids"],
            "query_attention_mask": query_encoding["attention_mask"],
            "query_token_type_ids": query_encoding.get("token_type_ids", None),  # Compatible with tokenizers that don't return token_type_ids
            "pos_ctx_input_ids": pos_ctx_encoding["input_ids"],
            "pos_ctx_attention_mask": pos_ctx_encoding["attention_mask"],
            "pos_ctx_token_type_ids": pos_ctx_encoding.get("token_type_ids", None),
        }
        
        # If in training mode, add negative samples
        if self.is_training:
            # Encode all negative sample texts
            neg_ctx_encodings = []
            for neg_ctx in negative_ctxs:
                neg_encoding = self.tokenizer(
                    neg_ctx,
                    max_length=self.max_ctx_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # Remove batch dimension
                neg_encoding = {k: v.squeeze(0) for k, v in neg_encoding.items()}
                neg_ctx_encodings.append(neg_encoding)
            
            # Stack all negative sample encodings
            n_neg = len(neg_ctx_encodings)
            if n_neg > 0:
                neg_input_ids = torch.stack([enc["input_ids"] for enc in neg_ctx_encodings])
                neg_attention_mask = torch.stack([enc["attention_mask"] for enc in neg_ctx_encodings])
                
                if "token_type_ids" in neg_ctx_encodings[0]:
                    neg_token_type_ids = torch.stack([enc["token_type_ids"] for enc in neg_ctx_encodings])
                    inputs["neg_ctx_token_type_ids"] = neg_token_type_ids
                
                inputs.update({
                    "neg_ctx_input_ids": neg_input_ids,
                    "neg_ctx_attention_mask": neg_attention_mask,
                })
        
        return inputs


def create_dataloader(
    data_file: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_query_length: int = 128,
    max_ctx_length: int = 2048,
    is_training: bool = True,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create data loader
    
    Args:
        data_file: Data file path
        tokenizer: Tokenizer
        batch_size: Batch size
        max_query_length: Maximum length of query text
        max_ctx_length: Maximum length of context text (utilizing nomic-bert-2048's long sequence capability)
        is_training: Whether in training mode
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader object
    """
    dataset = DPRDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        max_ctx_length=max_ctx_length,
        is_training=is_training
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader