# NewsEncoder 설정
class Config:
    gpu_num = 0
    seed = 28
    use_batch = True
    
    week = 7
    hop = 3
    interval_minutes = 30 # 30, 720, 1440, 2160
    
    num_words = 1 + 330899   # 실제 단어 수(330899)에 패딩 토큰(index=0)을 더함; index = 0: 존재하지 않는 단어들
    word_embedding_dim = 100   # 사전 학습된 단어 embedding 차원
    if week == 7:
        num_categories = 35#35   # nyheter category를 nyheter의 subcategory로 대체하고 No category case까지 포함한 수
        num_categories_for_NewsEncoder = 17#16
        num_subcategories_for_NewsEncoder = 93#80
    elif week == 3:
        num_categories = 34  
        num_categories_for_NewsEncoder = 16
        num_subcategories_for_NewsEncoder = 80
    else:
        num_categories = 26  
        num_categories_for_NewsEncoder = 14
        num_subcategories_for_NewsEncoder = 55
        
    num_filters = 100   # snapshots에서 news, user, category embedding 차원 * (1/3)
    query_vector_dim = 200   # NewsEncoder query vector 차원
    window_size = 3
    dropout_probability = 0.2
    
    ### for MSA NewsEncoder
    method = 'multihead_self_attention' # 'cnn_attention'

    head_num = 20
    head_dim = 15
    dataset_lang = 'norwegian'
    category_emb_dim = 100
    subcategory_emb_dim = 100
    attention_hidden_dim = 100
    
    ### for ablation studies
    no_category     = False   # NewsEncoder category 사용 여부
    unique_category = False   # category 1개로 GCN edge message passing
    adjust_score    = True    # 수명 고려한 스코어 조정