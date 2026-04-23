import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    """Additive attention to aggregate sequence into a
       single vector using a learned query."""
    def __init__(self, dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, x, mask=None):
        # x: (batch, seq_len, dim)
        e = torch.tanh(self.proj(x))     # (batch, seq, hidden)
        scores = self.query(e).squeeze(-1)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)  # (batch, seq)
        return torch.bmm(
            weights.unsqueeze(1), x).squeeze(1) # (batch, dim)
    
class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads=16,
                 head_dim=16, dropout=0.2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=news_dim, num_heads=num_heads,
            batch_first=True)
        self.additive_attn = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, clicked_news_vecs, mask=None):
        # clicked_news_vecs: (batch, history_len, news_dim)
        x, _ = self.multihead_attn(
            clicked_news_vecs, clicked_news_vecs,
            clicked_news_vecs)
        x = self.dropout(x)
        user_vec = self.additive_attn(x, mask)
        return user_vec  # (batch, news_dim)