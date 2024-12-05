import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os


class sequence(nn.Module):
    def __init__(self, input_dim, num_heads=4, layer_num=4, dropout=0.1):

        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.layer_num = layer_num
        
        self.dropout = dropout

        self.time_embedding = nn.Embedding(48, self.input_dim)
        self.attn = AttentionLayer(self.input_dim, self.num_heads, self.dropout)
        self.predict = nn.Linear(self.input_dim, self.input_dim)


    def forward(self, x, history_time_encoding):
        time_encoding = self.time_embedding(history_time_encoding).unsqueeze(0)
        x = x + time_encoding
        for i in range(self.layer_num):
            x = self.attn(x, x, x)
        out = self.predict(x)
        out = out[:,-1].unsqueeze(1)

        return out



class AttentionLayer(nn.Module): 
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        self.dropout = dropout
        self.feed_forward_dim = self.input_dim * 4

        self.FC_Q = nn.Linear(self.input_dim, self.input_dim)
        self.FC_K = nn.Linear(self.input_dim, self.input_dim)
        self.FC_V = nn.Linear(self.input_dim, self.input_dim)

        self.out_proj = nn.Linear(self.input_dim, self.input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dim, self.feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feed_forward_dim, self.input_dim),
        )

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        

    def forward(self, query, key, value):
        residual = query

        batch_size = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out

