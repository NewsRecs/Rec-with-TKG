import torch
import pandas as pd
from model.config_3w import Config


def ns_indexing(ns_file_path, batch_size, user_num=84989, test=False):
    # user_idx_batch = []    
    ns_df = pd.read_csv(ns_file_path, sep='\t')

    all_user_ids = [i for i in range(user_num)]

    # batch_user_emb = []

    # *** train_batch_embs의 shape: batch 수
    # *** train_batch_embs의 각 요소: [snapshot idx, batch user embeddings]
    # A = 0
    prev_batch = 0
    batch = 0
    # train_batch = []   # 최종 정보들을 담은 리스트
    # batch_subgraph = []
    # batch_clicked_pairs = []
    # batch_e_ids = []
    # batch_seed_list = []
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    # batch_seed_list = [[] for _ in range(len(batch_num))]

    # ns에 필요한 데이터 저장하는 변수들
    ns_idx_batch = []
    test_cand_score_weight_batch = []
    b_num = 0
    for b in range(batch_num):
        # b_num += 1
        # if b_num > 5:
        #     break
        prev_batch = b * batch_size
        batch = min((b+1) * batch_size, user_num)
        batch_user_ids = all_user_ids[prev_batch:batch]   # ex) 0 ~ 499, 500 ~ 999, ..., 84500 ~ 84989
        
        """
        추가한 부분 (for negative sampling)
        목표: user_score_idx 생성 및 저장
        """
        batch_ns_df = ns_df[ns_df['user_int'].isin(batch_user_ids)]
        # user_idx 처리
        # user_tensor = torch.tensor(batch_ns_df['user_int'].tolist(), dtype=torch.long)
        # user_idx_batch.append(user_tensor)
        # negative samples 포함한 news_idx 처리
        ns_idx_list = []
        test_cand_weight_list = []
        for _, row in batch_ns_df.iterrows():
            # positive 뉴스 id (이미 news2int 매핑된 정수값)
            pos = int(row['news_int'])
            
            # negative_samples 처리: 공백으로 구분된 문자열을 리스트로 변환
            neg_str = row['negative_samples']
            # 각 요소를 int로 변환
            neg_list = [int(x) for x in neg_str.split()]
            negs = neg_list
            ns_idx_list.append([pos] + negs)
            if test and Config.adjust_score:
                ### score weight 추가
                candidate_weight_str = row['candidate_weight']
                # 각 요소를 float로 변환
                candidate_weight_list = [float(x) for x in candidate_weight_str.split()]
                test_cand_weight_list.append(candidate_weight_list)
        
        # 리스트를 텐서로 변환 (shape: [row_num, 5])
        ns_idx_tensor = torch.tensor(ns_idx_list, dtype=torch.long)
        ns_idx_batch.append(ns_idx_tensor)
        
        test_cand_score_weight_batch.append(test_cand_weight_list)
    
    return ns_idx_batch, test_cand_score_weight_batch#, user_idx_batch
