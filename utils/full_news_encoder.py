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
        self.category_embedding = nn.Embedding(config.num_categories_for_NewsEncoder + config.num_subcategories_for_NewsEncoder - 1,   
                                               config.num_filters,
                                               padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)

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

        # # ===== DEBUG 코드 추가 =====
        # if (category_idx >= self.config.num_categories_for_NewsEncoder + self.config.num_subcategories_for_NewsEncoder - 1) or (category_idx < 0):
        #     raise ValueError(f"[Category] invalid index {category_idx} (num_categories={self.config.num_categories})")
        # if (subcategory_idx >= self.config.num_categories_for_NewsEncoder + self.config.num_subcategories_for_NewsEncoder - 1) or (subcategory_idx < 0):
        #     raise ValueError(f"[SubCategory] invalid index {subcategory_idx} (num_categories={self.config.num_categories})")
        # # ==========================
        
        if self.config.use_batch:
            # Part 1: calculate category_vector
            # batch_size, num_filters
            category_vector = self.category_embedding(torch.tensor(category_idx, device=self.device).long())

            # Part 2: calculate subcategory_vector & title_vector
            # batch_size, num_filters
            # subcategory 임베딩 직전에
            num_scats = self.category_embedding.num_embeddings
            assert all(0 <= sc < num_scats for sc in subcategory_idx), \
                f"subcategory_idx 범위 초과! max_sc={max(subcategory_idx)}, num_embeddings={num_scats}"

            subcategory_vector = self.category_embedding(torch.tensor(subcategory_idx, device=self.device).long())

            # batch_size, num_words_title, word_embedding_dim
            title_vector = F.dropout(self.word_embedding(title_idx),
                                    p=self.config.dropout_probability,
                                    training=self.training)
        else:
            # batch_size, num_filters
            category_vector = self.category_embedding(torch.tensor(category_idx, device=self.device).long().unsqueeze(0))

            # Part 2: calculate subcategory_vector & title_vector
            # batch_size, num_filters
            subcategory_vector = self.category_embedding(torch.tensor(subcategory_idx, device=self.device).long().unsqueeze(0))

            # batch_size, num_words_title, word_embedding_dim
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
        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1)
        return news_vector
