# model_name.py

import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from GCRNN import GCRNN

def main():
    # 0) device 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) 저장된 snapshots, wor2int, 사전 학습된 단어 embeddings, negative samples 및 df 불러오기
    # snapshots_embeddings.pkl 로드
    snapshots_path = "./Adressa_4w/history/snapshots.pkl"
    with open(snapshots_path, 'rb') as f:
        snapshots = pickle.load(f)
    
    print(f"Loaded snapshots' embeddings from {snapshots_path}. #={len(snapshots)}")

    # 사전 학습된 단어 로드
    word2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'word2int.tsv'), sep='\t')

    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('./Adressa_4w/history/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)   # (330900, 100)
    
    # negative samples 로드 - 추후 수정 필요
    ns_tsv = pd.read_csv(os.path.join('./Adressa_4w/history/', 'behaviors_parsed_ns4_lt36.tsv'), sep='\t')
    ns_datas = ns_tsv[['candidate_news_pop', 'candidate_news_sqrt_pop', 'candidate_news_rev_log_pop']]
    
    # df 로드
    def tokenize_title(title: str) -> list:
        """
        2.2) 타이틀을 공백 기준으로 단순 토크나이징
        """
        return title.split()
    
    file_path = './Adressa_4w/history/history_tkg_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    user_num = len(df['history_user'].unique())
    # 뉴스별 제목 집계
    news_info = df.groupby('clicked_news', as_index=False).agg({'title': 'first'})

    # title -> token -> index
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [word_to_idx[w] if w in word_to_idx else 0 for w in words]
    )

    # news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
    news_info_df = news_info[['clicked_news', 'title_idx']].rename(
        columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
    )

    # news_id_to_info
    news_id_to_info = news_info_df.set_index('news_id')\
        [['title_idx']].to_dict(orient='index')
        # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장
    
    news2int_df = pd.read_csv('./Adressa_4w/history/news2int.tsv', sep='\t')
    df2 = pd.merge(df, news2int_df, left_on='clicked_news', right_on='news_id', how='left')
    df2 = df2.sort_values('news_int')
    all_news_ids = news2int_df['news_id'].unique()

    # news2int = {nid[1:]: i for i, nid in enumerate(all_news_ids)}
    category2int = pd.read_csv('./Adressa_4w/history/category2int.tsv', sep='\t')
    cat_num = len(category2int['category']) + 1
    
    ### Loading idx_infos
    train_ns_idx_batch = torch.load('./Adressa_4w/train/train_ns_idx_batch5.pt')
    train_user_idx_batch = torch.load('./Adressa_4w/train/train_user_idx_batch5.pt')
    
    ### Loading train graph
    g_path = "./Adressa_4w/history/g.pkl"
    with open(g_path, 'rb') as f:
        g = pickle.load(f)
        g.to(device)
    
    with open("./Adressa_4w/history/splitted_subgraphs.pkl", 'rb') as f:
        splitted_subgraphs = pickle.load(f)
        
    
    with open("./Adressa_4w/history/seed_list.pkl", 'rb') as f:
        seed_lists = pickle.load(f)
        
        
    # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
    train_file_path = './Adressa_4w/train/valid_tkg_behaviors.tsv'
    train_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
    # 'clicked_news' 열에서 '-1' 제거
    train_df['clicked_newsId'] = train_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

    # train_df에서 nan이 존재하는 행 제거
    train_df = train_df.dropna(subset=['clicked_news'])

    news_num = len(train_df['clicked_news'].unique())
    
    
    
    # ### Loading test batch infos
    # test_b_start = time.time()
    # test_batch_path = './Adressa_4w/test/test_batch5.pkl'
    # print("Loading batch file..")
    # with open(test_batch_path, 'rb') as f:
    #     test_batch = pickle.load(f)
    
    # test_users = []
    # test_news = []
    # test_snapshots = []
    # # e_ids = []
    # test_seed_list = []
    # for batch_idx, each_batch in tqdm(enumerate(test_batch), desc='Processing test batches'):   # 170개
    #     test_users.append([])
    #     test_news.append([])
    #     test_snapshots.append([])
    #     # e_ids.append([])
    #     test_seed_list.append([])
    #     for idx, snapshot_dict in enumerate(each_batch):   # 1680개
    #         batch_snapshot = snapshot_dict['subgraph']
    #         test_snapshots[batch_idx].append(batch_snapshot)
    #         # batch_eIds = snapshot_dict['edge_ids']
    #         # e_ids[batch_idx].append(batch_eIds)
    #         batch_seed_list = snapshot_dict['seed_list']
    #         test_seed_list[batch_idx].append(batch_seed_list)
    #         batch_clicked_pairs = snapshot_dict['clicked_pairs']
    #         for batch_user, batch_news in batch_clicked_pairs:
    #             test_users[batch_idx].append(batch_user)
    #             test_news[batch_idx].append(batch_news)
    # print("Batch loading finished")
    # print(f"Batch processing time: {int((time.time() - test_b_start)/60)}m")
    
    # ### Loading idx_infos
    # test_ns_idx_batch = torch.load('./Adressa_4w/test/test_ns_idx_batch.pt')
    # test_user_idx_batch = torch.load('./Adressa_4w/test/test_user_idx_batch.pt')        
        
    # # a) test dataset(0212 08:00:02 ~ 0219 08:00:01)인 valid_tkg_behaviors.tsv 로드
    # test_file_path = './Adressa_4w/test/valid_tkg_behaviors.tsv'
    # test_df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')
    # # 'clicked_news' 열에서 '-1' 제거
    # test_df['clicked_newsId'] = test_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
    # # 불필요한 컬럼 제거
    # # print(train_df.head())
    # # print(train_df.columns)
    # # print(train_df.isna().sum())
    # # print(train_df[train_df.isna()]) # 하나가 데이터 오류
    # # print(len(train_df))
    # # exit()

    # # train_df에서 nan이 존재하는 행 제거
    # test_df = test_df.dropna(subset=['clicked_news'])

    # test_news_num = len(test_df['clicked_news'].unique())
    
        
    # 필요한 정보 로드 끝 -------------------------------------------------------------------------------------------------
    
    # 2) 모델에 필요한 정보 추가 준비
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 500
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    emb_dim = 128
    history_length = 100
    
    # 3) GCRNN 모델 초기화
    model = GCRNN(pretrained_word_embedding, all_news_ids, news_id_to_info, user_num, news_num, cat_num, emb_dim=emb_dim, batch_size=500)  
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    # 4) Batch 학습을 통해 train 수행
    print("Train start !")
    print(f"# of batch: {batch_num}, # of user: {user_num}, batch size: {batch_size}, embedding dim: {emb_dim}\n")
    for epoch in tqdm(range(num_epochs), desc='... training epochs'):
        model.train()
        for b in range(batch_num):
            # train_users[b], train_news[b], 업애버림
            loss = model(seed_lists[b], g, splitted_subgraphs[b], train_user_idx_batch[b], train_ns_idx_batch[b], history_length)
            loss.backward()   # calculate gradient
            optimizer.step()   # update parameter via calculated gradient
            optimizer.zero_grad()   # initialize gradient

    # 5) test 수행
    print("Test start!")
    model.eval()
    test_batch_size = 500
    test_batch_num = user_num // test_batch_size if user_num % test_batch_size == 0 else user_num // test_batch_size + 1

    with torch.no_grad():
        # for m in range(1,6):
        for m in range(1,2):
            news_ranks = []
            job_ranks = []
            prev_test_batch_cnt = 0
            test_batch_cnt = 0
            predicted_edges_u = []
            predicted_edges_v = []
            rel_idx = []
            for test_b in tqdm(range(test_batch_num)):
                test_batch_cnt+=test_batch_size
                All_UC_score, All_UJ_score = model.inference(test_users[b], test_news[b], test_seed_list[b], g, test_snapshots[b], test_user_idx_batch[b], test_ns_idx_batch[b], history_length)
                for user_id, UC_score, UJ_score in zip(test_user_entid2[prev_test_batch_cnt:test_batch_cnt], All_UC_score, All_UJ_score):
                    if len(label_comps_index[user_id][train_until + m]):
                        user_future_companies_index = label_comps_index[user_id][train_until + m]
                        Company_label_scores = UC_score[user_future_companies_index] # 정답 회사들의 점수들을 뽑는 과정이다.
                        for cls_ in Company_label_scores:
                            gap = UC_score - cls_
                            past_gap = gap[past_companies[user_id]]
                            news_ranks.append(len(gap[gap>0]) - len(past_gap[past_gap>0]) + 1)
                        Job_label_scores = UJ_score[label_jobs_index[user_id][train_until + m]]
                        for jls_ in Job_label_scores:
                            gap = UJ_score - jls_
                            job_ranks.append(len(gap[gap>0]) + 1)
                        comp_top_k1 = np.argsort(UC_score.to(torch.device("cpu")))[-k1:] + user_id_max
                        job_top_k2 = np.argsort(UJ_score.to(torch.device("cpu")))[-k2:]
                        for c in comp_top_k1:
                            predicted_edges_u.append(user_id)
                            predicted_edges_v.append(c)
                            predicted_edges_u.append(c)
                            predicted_edges_v.append(user_id)
                            rel_idx.append(job_top_k2[0])
                            rel_idx.append(job_top_k2[0])
                prev_test_batch_cnt = test_batch_cnt
            #print("m = ", m)
            mrr, h1, h3, h5, h10 = print_metrics(company_ranks, job_ranks)
            #=============================
            #train_until+=1
            remove_list.append([])
            predicted_graph = dgl.DGLGraph()
            predicted_graph.add_nodes(Train_Graph.number_of_nodes())
            predicted_graph.add_edges(predicted_edges_u, predicted_edges_v)
            predicted_graph.edata['relation_idx'] = torch.tensor(rel_idx)
            splitted_Train_Graph.append(predicted_graph)
    

    print("\nAll train finished.")

if __name__ == "__main__":
    main()
