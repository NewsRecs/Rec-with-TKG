import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general.attention.additive import AdditiveAttention
import random
import numpy as np


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_num}" if torch.cuda.is_available() else "cpu")
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)
        self.mlp = nn.Sequential(
            nn.Linear(config.num_filters, config.num_filters * 3),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_probability),
            # (원한다면 LayerNorm이나 Residual 블록 추가 가능)
        )


    def forward(self, title_idx, category_idx, subcategory_idx):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters * 3
        """
        
        if self.config.use_batch:
            title_vector = F.dropout(self.word_embedding(title_idx),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        else:
            title_vector = F.dropout(self.word_embedding(title_idx.unsqueeze(0)),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        
        
        # Part 3: calculate weighted_title_vector
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        # batch_size, num_filters * 3
        news_vector = self.mlp(weighted_title_vector)
        return news_vector
