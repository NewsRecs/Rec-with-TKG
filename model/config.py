# NewsEncoder 설정
class Config:
    num_words = 1 + 330899   # 실제 단어 수(330899)에 패딩 토큰(index=0)을 더함; index = 0: 존재하지 않는 단어들
    word_embedding_dim = 100   # 사전 학습된 단어 embedding 차원
    num_categories = 2 + 106   # 실제 카테고리 수(127)에 패딩 토큰(index=0)을 더함; index = 0: No category, No subcategory 케이스
    num_filters = 100   # snapshots에서 news, user, category embedding 차원
    query_vector_dim = 200   # NewsEncoder query vector 차원
    window_size = 3
    dropout_probability = 0.2