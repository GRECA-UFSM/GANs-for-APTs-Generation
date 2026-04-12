import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, cat_sizes: list[int], cat_dims: list[int], num_cont: int, sequence_length: int, num_labels: int):
        super().__init__()
        
        inputs = sum(cat_dims) + num_cont
        
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim) for size, dim in zip(cat_sizes, cat_dims)])
        
        self.linear_1 = nn.Linear(sequence_length * inputs, 32)
        self.relu_1 = nn.LeakyReLU()
        
        self.linear_2 = nn.Linear(32, 32)
        self.relu_2 = nn.LeakyReLU()
        
        self.head = nn.Linear(32, num_labels)
        
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        cat_embeds = [self.embeddings[i](x_cat[:, :, i]) for i in range(x_cat.shape[2])]
        
        x = torch.cat(cat_embeds + [x_cont], dim=-1)
        x = x.view(x.size(0), -1)
        
        x = self.linear_1(x)
        x = self.relu_1(x)
        
        x = self.linear_2(x)
        x = self.relu_2(x)
        
        x = self.head(x)
        
        return x
    
class Transformer(nn.Module):
    def __init__(self, cat_sizes: list[int], cat_dims: list[int], num_cont: int, embed_dim: int, num_heads: int, num_layers: int, sequence_length: int, num_labels: int):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, dim) for size, dim in zip(cat_sizes, cat_dims)])
        self.pos_embedding = nn.Embedding(sequence_length + 1, embed_dim)
        
        self.fc_inputs = nn.Linear(sum(cat_dims) + num_cont, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        
        self.head = nn.Linear(embed_dim, num_labels)
        
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x_cat.size()
        
        cat_embeds = [self.cat_embeddings[i](x_cat[:, :, i]) for i in range(x_cat.shape[2])]
        
        seq_tokens = torch.cat(cat_embeds + [x_cont], dim=-1)
        seq_tokens = self.fc_inputs(seq_tokens)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, seq_tokens], dim=1)
        
        positions = torch.arange(sequence_length + 1, device=x.device).unsqueeze(0)
        
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        
        logits = self.head(cls_output)
        
        return logits
    
class Transformer2(nn.Module):
    def __init__(self, cat_sizes: list[int], num_cont: int, embed_dim: int, num_heads: int, num_layers: int, sequence_length: int, num_labels: int):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleList([nn.Embedding(size, embed_dim) for size in cat_sizes])
        self.cont_projection = nn.Linear(num_cont, embed_dim)
        
        self.pos_embedding = nn.Embedding(sequence_length + 1, embed_dim)


        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        
        self.head = nn.Linear(embed_dim, num_labels)
        
    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x_cat.size()
        
        cat_embeds = [self.cat_embeddings[i](x_cat[:, :, i]) for i in range(x_cat.shape[2])]
        cont_embed = self.cont_projection(x_cont)

        seq_tokens = torch.stack(cat_embeds + [cont_embed], dim=-2).sum(dim=-2)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, seq_tokens], dim=1)
        
        positions = torch.arange(sequence_length + 1, device=x.device).unsqueeze(0)
        
        x = x + self.pos_embedding(positions)
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        
        logits = self.head(cls_output)
        
        return logits