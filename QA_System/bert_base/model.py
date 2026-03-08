import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union

class DPREncoder(nn.Module):
    """
    DPR encoder module, based on BERT
    """
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-bert-2048",
        pooling: str = "cls",
        normalize: bool = True
    ):
        """
        Initialize encoder
        
        Args:
            model_name: Pretrained model to use
            pooling: Pooling method ('cls', 'mean', 'max')
            normalize: Whether to perform L2 normalization on output vectors
        """
        super(DPREncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling = pooling
        self.normalize = normalize
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Encoded vector
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the last layer's hidden state
        hidden_state = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # Choose different processing methods based on pooling type
        if self.pooling == "cls":
            # Use [CLS] token representation as sequence representation
            vector = hidden_state[:, 0, :]  # [batch_size, hidden_size]
        elif self.pooling == "mean":
            # Take average over the entire sequence
            vector = torch.mean(hidden_state, dim=1)  # [batch_size, hidden_size]
        elif self.pooling == "max":
            # Take maximum value over the entire sequence
            vector = torch.max(hidden_state, dim=1)[0]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Normalize vector
        if self.normalize:
            vector = F.normalize(vector, p=2, dim=1)
            
        return vector


class DPRModel(nn.Module):
    """
    DPR dual-tower model
    """
    def __init__(
        self,
        query_encoder_name: str = "nomic-ai/nomic-bert-2048",
        ctx_encoder_name: str = "nomic-ai/nomic-bert-2048",
        shared_weights: bool = False,
        temperature: float = 0.05
    ):
        """
        Initialize DPR model
        
        Args:
            query_encoder_name: Pretrained model for query encoder
            ctx_encoder_name: Pretrained model for context encoder
            shared_weights: Whether to share weights between the two encoders
            temperature: Temperature coefficient for similarity calculation
        """
        super(DPRModel, self).__init__()
        
        # Create query encoder
        self.query_encoder = DPREncoder(model_name=query_encoder_name)
        
        # Whether to share weights
        if shared_weights:
            self.ctx_encoder = self.query_encoder
        else:
            self.ctx_encoder = DPREncoder(model_name=ctx_encoder_name)
            
        self.temperature = temperature
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode query"""
        return self.query_encoder(input_ids, attention_mask, token_type_ids)
    
    def encode_ctx(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode context text"""
        return self.ctx_encoder(input_ids, attention_mask, token_type_ids)
    
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_token_type_ids: Optional[torch.Tensor] = None,
        pos_ctx_input_ids: Optional[torch.Tensor] = None,
        pos_ctx_attention_mask: Optional[torch.Tensor] = None,
        pos_ctx_token_type_ids: Optional[torch.Tensor] = None,
        neg_ctx_input_ids: Optional[torch.Tensor] = None,
        neg_ctx_attention_mask: Optional[torch.Tensor] = None,
        neg_ctx_token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            query_*: Query text inputs
            pos_ctx_*: Positive sample text inputs
            neg_ctx_*: Negative sample text inputs
            
        Returns:
            (query_vectors, pos_ctx_vectors, neg_ctx_vectors)
        """
        # Encode query
        query_vectors = self.encode_query(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids
        )
        
        # Encode positive sample text
        pos_ctx_vectors = None
        if pos_ctx_input_ids is not None:
            pos_ctx_vectors = self.encode_ctx(
                pos_ctx_input_ids,
                pos_ctx_attention_mask,
                pos_ctx_token_type_ids
            )
        
        # Encode negative sample text
        neg_ctx_vectors = None
        if neg_ctx_input_ids is not None:
            # Check if negative sample input is a 3D tensor [batch_size, n_neg, seq_len]
            if len(neg_ctx_input_ids.shape) == 3:
                batch_size, n_neg, seq_len = neg_ctx_input_ids.shape
                
                # Reshape to 2D tensor [batch_size * n_neg, seq_len]
                neg_ctx_input_ids = neg_ctx_input_ids.reshape(batch_size * n_neg, seq_len)
                neg_ctx_attention_mask = neg_ctx_attention_mask.reshape(batch_size * n_neg, seq_len)
                
                if neg_ctx_token_type_ids is not None:
                    neg_ctx_token_type_ids = neg_ctx_token_type_ids.reshape(batch_size * n_neg, seq_len)
                
                # Encode negatives
                neg_ctx_vectors = self.encode_ctx(
                    neg_ctx_input_ids,
                    neg_ctx_attention_mask,
                    neg_ctx_token_type_ids
                )
                
                # Reshape back to 3D tensor [batch_size, n_neg, hidden_size]
                hidden_size = neg_ctx_vectors.shape[-1]  # Use the last dimension as hidden_size
                neg_ctx_vectors = neg_ctx_vectors.reshape(batch_size, n_neg, hidden_size)
            else:
                # Normal encoding
                neg_ctx_vectors = self.encode_ctx(
                    neg_ctx_input_ids,
                    neg_ctx_attention_mask,
                    neg_ctx_token_type_ids
                )
        
        return query_vectors, pos_ctx_vectors, neg_ctx_vectors
    
    def compute_similarity(
        self,
        query_vectors: torch.Tensor,
        ctx_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate similarity between query and context vectors
        
        Args:
            query_vectors: Query vectors [batch_size, hidden_size]
            ctx_vectors: Context vectors [batch_size, hidden_size] or [batch_size, n_contexts, hidden_size]
            
        Returns:
            Similarity scores [batch_size] or [batch_size, n_contexts]
        """
        # Check ctx_vectors dimension
        if len(ctx_vectors.shape) == 3:
            # [batch_size, n_contexts, hidden_size]
            batch_size, n_contexts, hidden_size = ctx_vectors.shape
            # Reshape query_vectors for broadcasting
            query_vectors = query_vectors.unsqueeze(1).expand(-1, n_contexts, -1)
            # Calculate dot product
            scores = torch.sum(query_vectors * ctx_vectors, dim=2) / self.temperature
        else:
            # [batch_size, hidden_size]
            scores = torch.sum(query_vectors * ctx_vectors, dim=1) / self.temperature
            
        return scores
        
    def compute_loss(
        self,
        query_vectors: torch.Tensor,
        pos_ctx_vectors: torch.Tensor,
        neg_ctx_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss
        
        Args:
            query_vectors: Query vectors [batch_size, hidden_size]
            pos_ctx_vectors: Positive sample vectors [batch_size, hidden_size]
            neg_ctx_vectors: Negative sample vectors [batch_size, n_neg, hidden_size]
            
        Returns:
            Contrastive loss
        """
        # Ensure neg_ctx_vectors is a 3D tensor
        if len(neg_ctx_vectors.shape) != 3:
            raise ValueError(f"neg_ctx_vectors should be a 3D tensor, but got shape: {neg_ctx_vectors.shape}")
        
        batch_size, n_neg, _ = neg_ctx_vectors.shape
        
        # Compute query-positive similarity
        pos_scores = self.compute_similarity(query_vectors, pos_ctx_vectors)  # [batch_size]
        
        # Compute query-negative similarities
        neg_scores = self.compute_similarity(query_vectors, neg_ctx_vectors)  # [batch_size, n_neg]
        
        # Concatenate positive and negative scores
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch_size, 1 + n_neg]
        
        # Target is always the 0th index (positive sample)
        target = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
        
        # Compute contrastive loss with temperature scaling
        loss = F.cross_entropy(scores, target)
        
        return loss
    
    def save_encoders(self, output_dir: str) -> None:
        """
        Save encoders to files
        
        Args:
            output_dir: Output directory path
        """
        import os
        
        # Save query encoder
        query_output_dir = os.path.join(output_dir, "query_encoder")
        os.makedirs(query_output_dir, exist_ok=True)
        self.query_encoder.bert.save_pretrained(query_output_dir)
        
        # Save context encoder (only if not shared with query encoder)
        if self.ctx_encoder is not self.query_encoder:
            ctx_output_dir = os.path.join(output_dir, "ctx_encoder")
            os.makedirs(ctx_output_dir, exist_ok=True)
            self.ctx_encoder.bert.save_pretrained(ctx_output_dir)
    
    @classmethod
    def load_encoders(
        cls,
        model_dir: str,
        shared_weights: bool = False,
        temperature: float = 0.05
    ) -> "DPRModel":
        """
        Load encoders from files
        
        Args:
            model_dir: Model directory path
            shared_weights: Whether query and context encoders share weights
            temperature: Temperature parameter
            
        Returns:
            DPR model instance
        """
        import os
        
        # Check query encoder path
        query_encoder_dir = os.path.join(model_dir, "query_encoder")
        if not os.path.exists(query_encoder_dir):
            raise ValueError(f"Query encoder not found at {query_encoder_dir}")
        
        # If shared_weights, use query encoder for both
        if shared_weights:
            model = cls(
                query_encoder_name=query_encoder_dir,
                ctx_encoder_name=query_encoder_dir,  # Use same path
                shared_weights=True,
                temperature=temperature
            )
            return model
        
        # Check context encoder path
        ctx_encoder_dir = os.path.join(model_dir, "ctx_encoder")
        if not os.path.exists(ctx_encoder_dir):
            # Fall back to shared weights if context encoder not found
            return cls.load_encoders(model_dir, shared_weights=True, temperature=temperature)
        
        # Load with separate encoders
        model = cls(
            query_encoder_name=query_encoder_dir,
            ctx_encoder_name=ctx_encoder_dir,
            shared_weights=False,
            temperature=temperature
        )
        
        return model