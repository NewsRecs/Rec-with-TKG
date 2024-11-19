import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from datetime import timedelta

### User Encoder class 정의
class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        ### torch.nn.Module 초기화
        super(UserEncoder, self).__init__()
        self.config = config
        ### filter의 수가 1.5배 해도 정수인지 확인
        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        ### 유저 클릭 히스토리에 적용할 GRU layer 초기화
        self.gru = nn.GRU(
            config.num_filters * 3,
            config.num_filters * 3 if config.long_short_term_method == 'ini'
            else int(config.num_filters * 1.5))

    ### 유저 embedding 벡터 구하는 함수 정의
    def forward(self, user, clicked_news_length, clicked_news_vector):
        """
        Args:
            user:
                ini: batch_size, num_filters * 3
                con: batch_size, num_filters * 1.5
            clicked_news_length: batch_size,
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters * 3
        Returns:
            (shape) batch_size, num_filters * 3
        """
        ### user가 클릭한 뉴스가 존재하지 않으면, 해당 길이를 1로 설정
        clicked_news_length[clicked_news_length == 0] = 1
        # 1, batch_size, num_filters * 3
        ### 1) 'ini' framework
        if self.config.long_short_term_method == 'ini':
            ### user를 GRU의 첫 번째 hidden state로 초기화
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_vector,
                                      user.unsqueeze(dim=0))
            ### GRU의 마지막 hidden state를 유저 벡터로 반환
            return last_hidden.squeeze(dim=0)
        ### 2) 'con' framework
        else:
            ### GRU의 초기 state를 user로 초기화하지 않음
            packed_clicked_news_vector = pack_padded_sequence(
                clicked_news_vector,
                clicked_news_length,
                batch_first=True,
                enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_vector)
            ### GRU의 last hidden state와 user를 concat하여 최종 유저 벡터 반환
            return torch.cat((last_hidden.squeeze(dim=0), user), dim=1)
