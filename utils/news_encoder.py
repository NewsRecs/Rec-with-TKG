import torch
import torch.nn as nn
import torch.nn.functional as F
from general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### NewsEncoder class 정의
class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        ### 사전 학습된 단어 임베딩이 없는 경우 새로 생성
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        ### 사전 학습된 단어 임베딩이 있으면 해당 임베딩으로 초기화
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        # ### 뉴스의 카테고리 데이터 embedding layer 정의
        # self.category_embedding = nn.Embedding(config.num_categories,
        #                                        config.num_filters,
        #                                        padding_idx=0)
        ### CNN window 크기 1 이상 &홀수인지 검증
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            in_channels=1,
            out_channels=config.num_filters,
            kernel_size=(config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        ### 중요한 제목 선택하는 attention layer 정의
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)

    ### 뉴스 데이터 encoding 수행
    def forward(self, news_indices):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, num_filters (* 3)
        """
        # Part 1: calculate category_vector

        # batch_size, num_filters
        
        # ### category & subcategory를 embedding 벡터로 변환
        # category_vector = self.category_embedding(news['category'].to(device))

        # Part 2: calculate subcategory_vector

        # # batch_size, num_filters
        # subcategory_vector = self.category_embedding(
        #     news['subcategory'].to(device))

        # Part 3: calculate weighted_title_vector

        # batch_size, num_words_title, word_embedding_dim
        ### title에 포함된 단어들을 embedding 벡터로 변환
        # title_vector = self.word_embedding(news_indices)   # (num of words, dim of word embedding)
        # title_vector = title_vector.unsqueeze(0)  # (1, 3, 100)  -> 배치 차원 추가

        title_vector = F.dropout(self.word_embedding(news_indices),
                                 p=self.config.dropout_probability,
                                 training=self.training).unsqueeze(0)
        # batch_size, num_filters, num_words_title
        ### title embedding 벡터에 CNN 적용하여 문맥 정보 학습
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        ### unsqueeze(dim=1)
        ##### 변경 전 형태: (batch_size, num_words_title, word_embedding_dim)
        ##### 변경 후 형태: (batch_size, 1, num_words_title, word_embedding_dim)
        ### squeeze(dim=3)
        ##### 변경 전 형태: (batch_size, num_filters, num_words_title, 1)
        ##### 변경 후 형태: (batch_size, num_filters, num_words_title)
        
        # batch_size, num_filters, num_words_title
        ### CNN 출력에 ReLU 함수 적용하여 음수 제거하고, dropout 적용
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        # batch_size, num_filters
        ### 앞선 출력에 attention 메커니즘 적용
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        # batch_size, num_filters * 3
        ### category, subcategory, title을 concat하여 최종 뉴스 벡터 생성 후 반환
        # news_vector = torch.cat(
        #     [category_vector, subcategory_vector, weighted_title_vector],
        #     dim=1)
        news_vector = weighted_title_vector
        
        return news_vector


