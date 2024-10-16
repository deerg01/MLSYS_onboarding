import torch.nn as nn

from .Transformer import TransformerBlock
from embedding import BERTEmbedding

"""
bert.py
"""

class BERT(nn.Module) :
    
    def __init__(self, vocab_size, hidden = 768, n_layers = 12, attn_heads = 12, dropout = 0.1) :
        super().__init__()
        self.hidden = hidden
        self.n_layer = n_layers
        self.attn_heads = attn_heads
        
        # paper : use 4*hidden_size for ff network hidden size
        self.feed_forward_hidden = hidden * 4
        
        # embedding for BERT = token + segment + position
        self.embedding = BERTEmbedding(vocab_size = vocab_size, embed_size = hidden)
        
        # transformer block
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )
        
    def forward(self, x, segment_info) :
        # attention masking
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        
        # run multiple transformer block
        for transformer in self.transformer_blocks :
            x = transformer.forward(x, mask)
        
        return x