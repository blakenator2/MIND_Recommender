import torch
import torch.nn as nn
import torch.nn.functional as F
from src.news_encoder import NewsEncoder
from src.user_encoder import UserEncoder
    
class NRMSModel(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16,
                 head_dim=16, dropout=0.2):
        super().__init__()
        news_dim = num_heads * head_dim
        self.news_encoder = NewsEncoder(
            embedding_matrix, num_heads, head_dim, dropout)
        self.user_encoder = UserEncoder(
            news_dim, num_heads, head_dim, dropout)


    def forward(self, history_ids, candidate_ids, hist_mask=None):
        # history_ids: (batch, hist_len, title_len)
        # candidate_ids: (batch, num_candidates, title_len)
        batch, hist_len, tlen = history_ids.shape
        _, n_cand, _ = candidate_ids.shape


        # Encode clicked news history
        hist_flat = history_ids.view(-1, tlen)
        hist_vecs = self.news_encoder(hist_flat)
        hist_vecs = hist_vecs.view(batch, hist_len, -1)


        # Encode candidate news
        cand_flat = candidate_ids.view(-1, tlen)
        cand_vecs = self.news_encoder(cand_flat)
        cand_vecs = cand_vecs.view(batch, n_cand, -1)


        # User representation
        user_vec = self.user_encoder(hist_vecs, hist_mask)


        # Click prediction scores
        scores = torch.bmm(
            cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores  # (batch, num_candidates)