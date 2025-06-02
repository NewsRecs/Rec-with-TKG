import torch
import torch.nn as nn
from utils.layer import *
from torch_geometric.nn import Sequential, GCNConv
import pandas as pd

class NewsEncoder(nn.Module):
    def __init__(self, cfg, pretrained_word_embedding):
        super().__init__()
        token_emb_dim = cfg.word_embedding_dim
        self.news_dim = cfg.head_num * cfg.head_dim

        self.dataset_lang = cfg.dataset_lang
        if self.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:   # Adressa
            pretrained_emb = torch.tensor(pretrained_word_embedding, dtype=torch.float)
            self.word_encoder = nn.Embedding.from_pretrained(pretrained_emb, freeze=False, padding_idx=0)
        
        attention_input_dim = cfg.num_filters * cfg.window_size# + cfg.category_emb_dim + cfg.subcategory_emb_dim   # 300 + 100 + 100

        
        from torch import Tensor  # 꼭 필요
        
        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (MultiHeadAttention(
                attention_input_dim,
                attention_input_dim,
                attention_input_dim,
                cfg.head_num,
                cfg.head_dim
            ), 'x, x, x, mask -> x'),

            (nn.LayerNorm(attention_input_dim), 'x -> x'),
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),

            (AttentionPooling(
                attention_input_dim,
                cfg.attention_hidden_dim
            ), 'x, mask -> x'),

            (nn.LayerNorm(attention_input_dim), 'x -> Tensor'),
        ])        
        self.last_encoder = nn.Linear(300, 300)   # (attention_input_dim, self.news_dim); 1000 -> 500
        
        self.attetio = Sequential('x, mask', [   # 최종 출력 타입은 Tensor
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(
                token_emb_dim,
                token_emb_dim,
                token_emb_dim,
                cfg.head_num,
                cfg.head_dim
            ), 'x, x, x, mask -> x'),
            (nn.LayerNorm(self.news_dim), 'x -> x'),
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (AttentionPooling(
                self.news_dim,
                cfg.attention_hidden_dim
            ), 'x, mask -> x'),
            (nn.LayerNorm(self.news_dim), 'x -> Tensor'),  # 마지막은 반드시 'x -> Tensor'
        ])

    # --------------------------------------------------------------------- #
    # forward: attetio ➜ concat ➜ attention ➜ last_encoder                 #
    # --------------------------------------------------------------------- #
    def forward(self, title_idx, category_idx, subcategory_idx):
        device = self.word_encoder.weight.device
        
        # ❷ 1-D 입력이면 배치 차원 추가
        if title_idx.dim() == 1:
            title_idx = title_idx.unsqueeze(0)              # (1, L)
                    
        # 단어 인덱스 시퀀스를 임베딩하여 [B, seq_len, embed_dim] 텐서를 얻습니다.
        title_embeddings = self.word_encoder(title_idx)
        mask = (title_idx != 0)

        # 어텐션 마스크 생성 (패딩 토큰 인덱스 0은 False로 설정)
        # mask.shape: [B, seq_len], pad 위치=False, 단어 위치=True

        # Multi-Head Self-Attention + 어텐션 풀링으로 제목 임베딩 벡터 획득 (shape: [B, news_dim])
        title_vector = self.attetio(title_embeddings, mask)  # 제목 시퀀스 -> 제목 벡터

        # 카테고리 및 서브카테고리 인덱스를 임베딩 벡터로 변환합니다.

        # 제목, 카테고리, 서브카테고리 벡터를 연결하여 하나의 벡터로 결합합니다.
        fuse_word_emb = title_vector # [B, title_dim]
        fuse_word_emb  = fuse_word_emb.unsqueeze(1)                                # (B, 1, 500)
        fuse_mask = torch.ones(fuse_word_emb.size(0), 1, device=device, dtype=torch.bool)

        # 결합된 벡터에 대해 2차 어텐션 레이어를 적용하여 뉴스 표현을 강화합니다.
        attention_output = self.attention(fuse_word_emb, fuse_mask)   

        # 최종 선형 변환을 통해 출력 차원을 [B, news_dim]으로 줄여 최종 뉴스 벡터를 얻습니다.
        news_vector = self.last_encoder(attention_output)   # (B, 1, 300)
        news_vector = news_vector.squeeze(1)
        
        return news_vector