import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model.LSTUR.news_encoder import NewsEncoder
from model.LSTUR.user_encoder import UserEncoder
from general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### LSTUR class 정의
class LSTUR(torch.nn.Module):
    """
    LSTUR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    ### 초기화
    def __init__(self, config, pretrained_word_embedding=None):
        """
        # ini
        user embedding: num_filters * 3
        news encoder: num_filters * 3
        GRU:
        input: num_filters * 3
        hidden: num_filters * 3

        # con
        user embedding: num_filter * 1.5
        news encoder: num_filters * 3
        GRU:
        input: num_fitlers * 3
        hidden: num_filter * 1.5
        """
        ### torch.nn.Module 의 init 호출하여 초기 설정
        super(LSTUR, self).__init__()
        self.config = config
        ### 뉴스 인코더 초기화
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        ### 유저 인코더 초기화
        self.user_encoder = UserEncoder(config)
        ### 클릭 확률 예측기는 내적 함수로 세팅
        self.click_predictor = DotProductClickPredictor()
        ### filter 수가 1.5배 증가해도 정수인지 확인
        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        ### UserId를 embedding 벡터로 변환
        self.user_embedding = nn.Embedding(
            config.num_users,
            config.num_filters * 3 if config.long_short_term_method == 'ini'
            else int(config.num_filters * 1.5),
            padding_idx=0)

    # def filter_recent_news(self, clicked_news, days=7):
    #     """
    #     Filters clicked_news to only include news clicked within the last `days` days from `reference_time`.
        
    #     Args:
    #         clicked_news: List of dictionaries, each with "news" and "timestamp" keys.
    #         reference_time: The timestamp of the current event (click).
    #         days: Number of days to look back.
        
    #     Returns:
    #         filtered_news: List of clicked news within the last `days` days.
    #     """
    #     # Calculate the cutoff time
    #     cutoff_time = clicked_news['time'] - timedelta(days=days)
        
    #     # Filter to include only news clicked within the last `days`
    #     filtered_news = [
    #         news_item for news_item in clicked_news 
    #         if news_item['timestamp'] >= cutoff_time
    #     ]
        
    #     return filtered_news

    ### 전체 구조 실행하는 forward 함수 정의
    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
        """
        Args:
            user: batch_size,
            clicked_news_length: batch_size,
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # batch_size, 1 + K, num_filters * 3
        ### 후보 뉴스 벡터 뉴스 인코더로 embedding 벡터로 변환
        ### 열 방향으로 tensor를 쌓아줌
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        # TODO what if not drop
        ### masking 확률에 따라 유저 드롭아웃
        user = F.dropout2d(self.user_embedding(
            user.to(device)).unsqueeze(dim=0),
                           p=self.config.masking_probability,
                           training=self.training).squeeze(dim=0)
        # batch_size, num_clicked_news_a_user, num_filters * 3
        ### 클릭 히스토리를 뉴스 인코더로 embedding 벡터로 변환
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters * 3
        ### 유저 인코더로 클릭 히스토리를 통해 최종 유저 벡터 생성
        user_vector = self.user_encoder(user, clicked_news_length,
                                        clicked_news_vector)
        # batch_size, 1 + K
        ### 유저, 후보 뉴스 벡터의 내적으로 클릭 확률 계산
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    ### 뉴스 벡터 구하는 함수
    def get_news_vector(self, news):
        # batch_size, num_filters * 3
        return self.news_encoder(news)

    ### 유저 벡터 구하는 함수
    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user: batch_size
            clicked_news_length: batch_size
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        # ini: batch_size, num_filters * 3
        # con: batch_size, num_filters * 1.5
        user = self.user_embedding(user.to(device))
        # batch_size, num_filters * 3
        return self.user_encoder(user, clicked_news_length,
                                 clicked_news_vector)

    ### 내적으로 클릭 확률 구하는 함수
    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
