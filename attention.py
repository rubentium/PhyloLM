# import math for square root in scaled dot-product attention
import math

# import torch
import torch

# import torch neural network module
import torch.nn as nn


class SimpleMultiHeadAttention(nn.Module):

    """
    Simple multi-head attention for a 4D Phyloformer-style input.
    
    Input shape:
        x: (batch_size, nb_row, nb_col, embed_dim)

    Output shape:
        out: (batch_size, nb_row, nb_col, embed_dim)

    What this module does:
        - takes the input tensor
        - makes Q, K, V using linear layers
        - splits them into multiple heads
        - computes attention scores
        - applies softmax to get attention weights
        - uses the weights to mix V
        - combines the heads back together
        - applies a final linear layer  
     """
    

    def __init__(self, embed_dim, num_heads):

        """
        set up the attention module.

        Input:
            embed_dim: size of feature dimension
            num_heads: number of attention heads
        """

        # call parent class constructor
        super().__init__()

        # save basic settings
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # make sure embed_dim can be split evenly across heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        # size of each head
        self.head_dim = embed_dim // num_heads

        # linear layer to make queries
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # linear layer to make keys
        self.k_proj = nn.Linear(embed_dim, embed_dim)  

        # linear layer to make values
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    
    def forward(self, x):

        """
        Run the forward pass of attention.
        
        Input:
            x: tensor of shape (batch_size, nb_row, nb_col, embed_dim)
            
        Output:
            out: tensor of shape (batch_size, nb_row, nb_col, embed_dim)
        """

        # unpack input shape
        batch_size, nb_row, nb_col, embed_dim = x.shape

        # check that input feature size matches what the layer expects
        if embed_dim != self.embed_dim:
            raise ValueError("Last dimension of x must match embed_dim")
        
        # make query tensor
        q = self.q_proj(x)

        # make key tensor
        k = self.k_proj(x)

        # make value tensor
        v = self.v_proj(x)

        # reshape Q to split feature dimension into heads
        # before: (batch_size, nb_row, nb_col, embed_dim)
        # after: (batch_size, nb_row, nb_col, num_heads, head_dim)
        q = q.view(batch_size, nb_row, nb_col, self.num_heads, self.head_dim)

        # reshape K the same way
        k = k.view(batch_size, nb_row, nb_col, self.num_heads, self.head_dim)

        # reshape V the same way
        v = v.view(batch_size, nb_row, nb_col, self.num_heads, self.head_dim)   

        # move num_heads before nb_col
        # new shape: (batch_size, nb_row, num_heads, nb_col, head_dim)
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        # transpose K on the last two dimensions so we can do QK^T
        # k_t shape: (batch_size, nb_row, num_heads, head_dim, nb_col)
        k_t = k.transpose(-1, -2)

        # compute raw attention scores
        # score shape: (batch_size, nb_row, num_heads, nb_col, nb_col)
        scores = torch.matmul(q, k_t) 

        # scale scores by sqrt(head_dim)
        scores = scores / math.sqrt(self.head_dim)

        # apply softmax over the last dimension to turn scores into attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # use attention weights to mix the values
        # context shape: (batch_size, nb_row, num_heads, nb_col, head_dim)
        context = torch.matmul(attn_weights, v)

        # move num_heads back after nb_col
        # new shape: (batch_size, nb_row, nb_col, num_heads, head_dim)
        context = context.transpose(2, 3)

        # combine all heads back into one feature dimension
        # new shape: (batch_size, nb_row, nb_col, embed_dim)
        context = context.contiguous().view(batch_size, nb_row, nb_col, self.embed_dim)

        # apply final output projection
        out = self.out_proj(context)

        return out