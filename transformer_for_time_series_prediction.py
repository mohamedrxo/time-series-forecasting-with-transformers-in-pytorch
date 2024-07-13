#first we should import the dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#create class for PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        
        
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


#then creating the main class for time sieries prediction

class Attention(nn.Module):
    def __init__(self,d_model,num_heads,dropout = 0.1,num_layers=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embadding  = nn.Linear(1,d_model)
        self.input_embedding  = nn.Conv1d(1, d_model, kernel_size=5)
        self.position_encoder = PositionalEncoding(d_model=20, 
                                                   dropout=dropout,
                                                   max_len=200)
        self.encoderlayer= nn.TransformerEncoderLayer(20, nhead=num_heads,dim_feedforward=d_model*4,dropout=dropout ,batch_first=True)
        self.transformer_encoder= nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)
        self.l1 = nn.Linear(580,500)
        self.l2 = nn.Linear(500,200)
        
        self.l3 = nn.Linear(200,100)
        self.l4= nn.Linear(100,1,bias=False)
        self.dropout = nn.Dropout(0.2)
        self.gelu=nn.GELU()
        

    def generate_attention_mask(self,sequence_length):
        attention_mask = torch.triu(torch.ones((sequence_length, sequence_length)), diagonal=1)
        attention_mask *= float('-inf')
        attention_mask[attention_mask != attention_mask] = 0
        return attention_mask
    def forward(self,x):
        
        x=x.view(x.shape[0],x.shape[1],1)
        x = F.pad(x, (0, 0, 3, 0), "constant", -1)
        out= input_embedding(x.transpose(1, 2))
        out = out.transpose(1, 2)
        
        out  = self.position_encoder(out)
        out  = self.transformer_encoder(out , mask =self.generate_attention_mask(out.shape[1]))
        out = out.view(out.shape[0],out.shape[1]*out.shape[2])
        out  = self.gelu(self.l1(out))
        out = self.dropout(out)
        out  = self.gelu(self.l2(out))
        out = self.dropout(out
        out  = self.gelu(self.l3(out))        
        out = self.dropout(out)
        out = self.l4(out)
        return out

# init the model and you use
model  = Attention(10,2)
