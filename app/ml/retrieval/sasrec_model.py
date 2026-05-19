import torch
import torch.nn as nn
import numpy as np

class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation.
    Uses a transformer encoder to model user interaction sequences.
    """
    def __init__(
        self,
        item_count: int,
        max_seq_len: int = 50,
        hidden_units: int = 128,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
    ):
        super(SASRec, self).__init__()
        self.item_count = item_count
        self.max_seq_len = max_seq_len

        # Embeddings
        self.item_emb = nn.Embedding(item_count + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Transformer blocks
        self.attention_layers = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.attention_layernorms = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate, batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(hidden_units, hidden_units),
                )
            )

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def forward(self, log_seqs):
        """
        Args:
            log_seqs: Tensor of item IDs with shape (batch, seq_len)
        Returns:
            User embeddings (batch, hidden_units) - representing the user's intent after the sequence.
        """
        # 1. Embeddings
        seqs = self.item_emb(log_seqs) # (batch, seq_len, hidden_units)
        
        # Position embeddings
        positions = torch.arange(log_seqs.shape[1], device=log_seqs.device).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)
        
        seqs = self.emb_dropout(seqs)

        # 2. Causality Masking (Self-Attention shouldn't look at future items)
        timeline_mask = (log_seqs == 0) # (batch, seq_len)
        seqs *= (~timeline_mask.unsqueeze(-1)).float() # mask padding out

        # Sequence mask (causal)
        sz = seqs.shape[1]
        attn_mask = torch.triu(torch.ones(sz, sz, device=seqs.device), diagonal=1).bool()

        # 3. Transformer Blocks
        for i in range(len(self.attention_layers)):
            # Self-Attention
            mha_input = self.attention_layernorms[i](seqs)
            
            # Pass only attn_mask. Do not pass key_padding_mask to avoid NaN when a query has all keys masked.
            attn_output, _ = self.attention_layers[i](
                mha_input, mha_input, mha_input, 
                attn_mask=attn_mask
            )
            seqs = seqs + attn_output # Residual
            seqs *= (~timeline_mask.unsqueeze(-1)).float() # Zero out padding positions
            
            # Feed-Forward
            ffn_input = self.forward_layernorms[i](seqs)
            ffn_output = self.forward_layers[i](ffn_input)
            seqs = seqs + ffn_output # Residual
            seqs *= (~timeline_mask.unsqueeze(-1)).float() # Zero out padding positions

        seqs = self.last_layernorm(seqs) # (batch, seq_len, hidden_units)
        
        # Return the last item's embedding as the user's current state
        # We need to find the actual last non-zero item
        # Simplified: take the embedding at the last position of the sequence
        return seqs[:, -1, :] 

    def predict(self, user_emb, item_indices):
        """
        Predict scores for a list of items given user embeddings.
        Args:
            user_emb: (batch, hidden_units)
            item_indices: (batch, num_candidates)
        Returns:
            Scores: (batch, num_candidates)
        """
        item_embs = self.item_emb(item_indices) # (batch, num_candidates, hidden_units)
        # Dot product
        return (user_emb.unsqueeze(1) * item_embs).sum(dim=-1)
