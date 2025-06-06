# model_name.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.make_train_datas import make_train_datas
from utils.make_test_datas import make_test_datas
from utils.time_split_batch import split_train_graph
from model.GCRNN_for_news_gcn import GCRNN
from utils.ns_indexing import ns_indexing
from utils.evaluate import ndcg_score, mrr_score
from utils.EarlyStopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from model.config import Config
import dgl



random_seed = 1024
random.seed(random_seed)

def main():
    # 0) device 및 batch_size 설정
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original_batch_size = 500
    snapshot_weeks = 6   ### history + train
    snapshots_num = snapshot_weeks * 7 * 24 * 2   # 2016
    # device = torch.device("cpu")

    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    print("dgl version:", dgl.__version__)
    
    ### history + train snapshots
    g, splitted_g = split_train_graph(
        snapshot_weeks, 
        './psj/Adressa_4w/history/total_graph_full_reciprocal.bin'
    )
    
    # Train 데이터 로드
    datas = make_train_datas()
    train_news, train_category, train_time = zip(*datas)

    # 사전 학습된 단어 로드
    word2int = pd.read_csv(
        os.path.join('./psj/Adressa_4w/history/', 'word2int.tsv'), 
        sep='\t'
    )
    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('./psj/Adressa_4w/history/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)   # (330900, 100)
    
    # df 로드
    def tokenize_title(title: str) -> list:
        """2.2) 타이틀을 공백 기준으로 단순 토크나이징"""
        return title.split()

    file_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    df['category'] = df['category'].fillna('No category|No subcategory')
    df[['category', 'subcategory']] = df['category'].str.split('|', n=1, expand=True)
    
    # 전체 뉴스 정보 로드
    combined_news_df = pd.read_csv(
        './psj/Adressa_4w/history/all_news_nyheter_splitted.tsv', 
        sep='\t'
    ).rename(columns={'newsId': 'clicked_news'})
    
    all_news_ids = pd.read_csv(
        './psj/Adressa_4w/history/news2int.tsv', 
        sep='\t'
    )['news_id']
    news_num = len(all_news_ids)
    user_num = len(df['history_user'].unique())
    all_users = [i for i in range(user_num)]
    
    # 뉴스별 제목/카테고리 집계
    news_info = combined_news_df.groupby('clicked_news', as_index=False).agg({
        'title': 'first',
        'category': 'first',
        'subcategory': 'first'
    })
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [
            word_to_idx[w.lower()] if w.lower() in word_to_idx else 0 
            for w in words
        ]
    )
    
    category2int = pd.read_csv('category2int_nyheter_splitted.tsv', sep='\t')    
    cat_num = Config.num_categories
    # category, subcategory를 index로
    news_info['category_idx'] = news_info['category'].map(category2int.to_dict())
    news_info['subcategory_idx'] = news_info['subcategory'].map(category2int.to_dict())
    
    # 필요한 컬럼만 news_info_df 생성
    news_info_df = news_info[['clicked_news', 'title_idx', 'category_idx', 'subcategory_idx']]\
        .rename(columns={'clicked_news': 'news_id'})
    
    # dict 형태(news_id_to_info)로 변환
    news_id_to_info = news_info_df.set_index('news_id')[
        ['title_idx', 'category_idx', 'subcategory_idx']
    ].to_dict(orient='index')
    
    # Negative sampling 인덱스 로드
    train_ns_idx_batch = ns_indexing(
        './psj/Adressa_4w/train/train_ns.tsv', 
        original_batch_size
    )
    
    # Test 데이터 로드
    #  - make_test_datas()가 (val_datas, test_datas)를 반환하지만 
    #    여기서는 validation이 필요 없으므로 val_datas를 test로 사용
    test_datas, _ = make_test_datas(snapshots_num)
    test_news, test_time, test_empty_check = zip(*test_datas)
    test_ns_idx_batch = ns_indexing('./psj/Adressa_4w/test/validation_ns.tsv', original_batch_size)
    # validation_ns가 train 직후의 3.5일에 대한 데이터이기 때문에 ns_indexing의 input으로 적합함
    
    print("data loading finished!")
    
    # 2) 모델에 필요한 정보 설정
    learning_rate = 0.0001
    num_epochs = 10     # 총 10epoch 진행
    batch_size = original_batch_size
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    emb_dim = Config.num_filters * 3  # 예: 300
    history_length = 100

    # 3) 모델 초기화
    model = GCRNN(
        all_news_ids,
        news_id_to_info,
        user_num,
        cat_num,
        news_num,
        pretrained_word_embedding=pretrained_word_embedding,
        emb_dim=emb_dim,
        batch_size=batch_size,
        snapshots_num=snapshots_num
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # 학습 과정에서의 로스 기록
    all_losses = []

    # 베스트 성능 저장을 위한 변수들
    best_score = -1.0
    best_epoch = 0
    best_auc = 0.0
    best_mrr = 0.0
    best_ndcg5 = 0.0
    best_ndcg10 = 0.0

    best_ckpt_path = "./best_model_ckpt.pth"

    # (2) EarlyStopping 객체 생성
    early_stopper = EarlyStopping(
        emb_dim=emb_dim,      # emb_dim 등 모델 설정에 맞춰 전달
        patience=3,           # 개선 없으면 3epoch 후 스탑(예시)
        min_delta=1e-4,
        ckpt_dir=f'./Adressa_7w/test/news_gcn_no_val_ckpt/bs_{original_batch_size}_lr_{learning_rate}', 
        verbose=True,
        save_all=False        # True로 설정하면 매 epoch마다 체크포인트 저장
    )

    print("Train start !")
    print(f"# of batch: {batch_num}, # of user: {user_num}, "
          f"batch size: {batch_size}, lr: {learning_rate}, "
          f"embedding dim: {emb_dim}, history_length: {history_length}\n")

    for epoch in range(1, num_epochs + 1):
        # -----------------------------
        # (1) Training
        # -----------------------------
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        prev_batch_cnt = 0
        batch_cnt = 0
        batch_size = original_batch_size

        for b in tqdm(range(batch_num), desc=f'Training Epoch {epoch}'):
            prev_batch_cnt = batch_cnt
            batch_cnt += batch_size
            if batch_cnt > len(train_news):
                batch_cnt = len(train_news)
            real_batch_size = batch_cnt - prev_batch_cnt

            loss = model(
                all_users[prev_batch_cnt:batch_cnt],
                train_news[prev_batch_cnt:batch_cnt],
                train_category[prev_batch_cnt:batch_cnt],
                train_time[prev_batch_cnt:batch_cnt],
                g,
                splitted_g,
                train_ns_idx_batch[b],
                history_length
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_samples += 1

        epoch_loss = epoch_loss_sum / (epoch_samples if epoch_samples else 1)
        print(f"[Epoch {epoch}] avg train_loss={epoch_loss:.6f}")

        # -----------------------------
        # (2) Test (매 epoch 종료 시)
        # -----------------------------
        model.eval()
        with torch.no_grad():
            test_batch_num = user_num // original_batch_size \
                             if user_num % original_batch_size == 0 \
                             else user_num // original_batch_size + 1

            all_scores = []
            all_labels = []
            list_mrr = []
            list_ndcg5 = []
            list_ndcg10 = []
            prev_test_batch_cnt = 0
            test_batch_cnt = 0
            empty_batch_count = 0

            for test_b in tqdm(range(test_batch_num), desc=f'Testing Epoch {epoch}'):
                prev_test_batch_cnt = test_batch_cnt
                test_batch_cnt += original_batch_size
                if test_batch_cnt > len(test_news):
                    test_batch_cnt = len(test_news)
                real_batch_size = test_batch_cnt - prev_test_batch_cnt

                # 만약 이 배치 내 유저들의 클릭 이력이 전혀 없다면 skip
                if not any(test_empty_check[prev_test_batch_cnt:test_batch_cnt]):
                    empty_batch_count += 1
                    continue

                candidate_score, test_loss = model.inference(
                    all_users[prev_test_batch_cnt:test_batch_cnt],
                    test_news[prev_test_batch_cnt:test_batch_cnt],
                    test_time[prev_test_batch_cnt:test_batch_cnt],
                    g,
                    splitted_g,
                    test_ns_idx_batch[test_b],
                    history_length
                )

                candidate_score = candidate_score.cpu().numpy()
                for i in range(real_batch_size):
                    y_score = candidate_score[i]
                    # 첫 번째가 정답
                    y_true = np.zeros(len(y_score), dtype=int)
                    y_true[0] = 1

                    all_scores.extend(y_score)
                    all_labels.extend(y_true)

                    list_mrr.append(mrr_score(y_true, y_score))
                    list_ndcg5.append(ndcg_score(y_true, y_score, k=5))
                    list_ndcg10.append(ndcg_score(y_true, y_score, k=10))

            # Test Metrics 계산
            if len(set(all_labels)) > 1:
                final_auc = roc_auc_score(all_labels, all_scores)
            else:
                final_auc = 0.0  # all_labels가 전부 1이거나 전부 0이면 AUC 계산 불가

            final_mrr = np.mean(list_mrr) if list_mrr else 0.0
            final_ndcg5 = np.mean(list_ndcg5) if list_ndcg5 else 0.0
            final_ndcg10 = np.mean(list_ndcg10) if list_ndcg10 else 0.0

            avg_metric = (final_auc + final_mrr + final_ndcg5 + final_ndcg10) / 4.0
            print(f"\n[Epoch {epoch} Test Metrics]")
            print(f"AUC={final_auc:.4f}, MRR={final_mrr:.4f}, "
                  f"nDCG@5={final_ndcg5:.4f}, nDCG@10={final_ndcg10:.4f}, "
                  f"avg={avg_metric:.4f}, (empty batch={empty_batch_count})\n")

            old_best_score = early_stopper.best_score  # 업데이트 전 점수
            early_stopper(val_score=avg_metric, model=model, epoch=epoch, lr=learning_rate)

            # best_score가 업데이트되었으면 해당 지표 저장
            if early_stopper.best_score != old_best_score:
                best_auc = final_auc
                best_mrr = final_mrr
                best_ndcg5 = final_ndcg5
                best_ndcg10 = final_ndcg10

            if early_stopper.early_stop:
                print("[EarlyStopping] Training is stopped.")
                break  # epoch 루프 종료

        if early_stopper.early_stop:
            break  # 메인 학습 루프 종료

    # -----------------------------
    # 전체 epoch 종료 or early stop 후,
    # 베스트 모델 다시 로드해서 최종 결과 출력
    # -----------------------------
    print("\n=== Training finished. Loading best checkpoint for final report ===")
    if early_stopper.best_ckpt_path is not None and os.path.exists(early_stopper.best_ckpt_path):
        model.load_state_dict(torch.load(early_stopper.best_ckpt_path))
        print(f"[Info] Best checkpoint (epoch={early_stopper.best_epoch}, avg_score={early_stopper.best_score:.4f}) loaded.")
    else:
        print("[Warning] Best checkpoint file not found. Using last model state.")

    # 최종 결과 (베스트 모델 기준) 출력
    print(f"\n[Training Completed] Best Test Performance (epoch={early_stopper.best_epoch}):")
    print(f" - AUC     : {best_auc:.4f}")
    print(f" - MRR     : {best_mrr:.4f}")
    print(f" - nDCG@5  : {best_ndcg5:.4f}")
    print(f" - nDCG@10 : {best_ndcg10:.4f}")
    print(f" - avg     : {early_stopper.best_score:.4f}\n")


if __name__ == "__main__":
    main()

    def forward(self, user_batch, news_batch, category_batch, time_batch, g, sub_g, ns_idx, history_length=100): # user_batch, news_batch,
        """
        g: DGL 이종 그래프
           - etype = ('user', 'clicked', 'news'),
                     ('user', 'clicked_reverse', 'news')
           - 
        #    - g.nodes['user'].data['feat'], g.nodes['news'].data['feat'] 에는
        #      노드 임베딩이 저장되어 있음.
           - g.edges['clicked'].data['feat'] 에는 user -> news 방향의 엣지 임베딩이 저장되어 있음.
           - g.edges['clicked_reverse'].data['feat'] 에는 news-> user 방향의 엣지 임베딩이 저장되어 있음.
           - g.nodes['user'].data['user_ids']에는 user_ids가 tensor 형태로 저장되어 있음. (user_num, 1)
           - g.nodes['news'].data['news_ids']에는 news_ids가 tensor 형태로 저장되어 있음. (news_num, 1)
        #    - g.edges['clicked'].data['category_id']에는 category_ids가 tensor 형태로 저장되어 있음. (cat_num, 1)
        #    - g.edges['clicked_reverse'].data['category_id']에는 category_ids가 tensor 형태로 저장되어 있음. (cat_num, 1)
        
        snapshots_user_ids: snapshot마다 저장된 유저 id

        return:
            updated_user_feats: (num_users, emb_dim)
            updated_news_feats: (num_news, emb_dim)
        """
        # print(f"[Before] GPU 메모리 사용량: {torch.cuda.memory_allocated(self.device)/1024**2:.2f} MiB")
        
        seed_list = []
        seed_entid = []
        train_t = []
        latest_train_time = self.snapshots_num - 1
        for i in range(latest_train_time+1):
            seed_list.append(set())
            
        for time_list, user, news_list in zip(time_batch, user_batch, news_batch):
            news_list = (news_list + self.user_num).tolist()
            for time, news in zip(time_list, news_list):
                seed_list[time].add(user)
                seed_list[time].add(news)  
                seed_entid.append(user)
                seed_entid.append(news)
                train_t.append(time)
                train_t.append(time)   # news가 seed에 포함됐기 때문
                
        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)
        
        seed_entid_t = torch.tensor(seed_entid)
        user_mask = seed_entid_t < self.user_num
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid)[user_mask] * latest_train_time + torch.tensor(train_t)[user_mask], 
                                            sorted = True, return_inverse = True)
        # 각 rnn의 마지막 hidden state, 즉 rnn 결과로 얻은 각 유저의 embedding indicies를 저장한 tensor
        # index_for_ent_emb: unique 값들의 indicies를 모아둔 tensor
        # 즉, seed_entid의 train_click_num만큼 존재하는 user indicies
        
        # for index in index_for_ent_emb:
        #     print(index)
        # print(index_for_ent_emb.shape)
        # print('\n')
        # print(len(train_t))
        # for tt in train_t:
        #     print(tt)
        # print('\n')
        # print(len(seed_entid))
        # for sid in seed_entid:    
        #     print(sid)
        # exit()
        
        user_embs = ent_embs[index_for_ent_emb]   # (train_click_num, 128)
                                                  # 이게 아마 정답) user_embs는 seed_entid 순으로 정렬된 user embeddings
                                                  # 즉, user_embs가 이미 내가 원하는 candidate_user_embs 형태!!!
        # print(entity_index.shape)
        # print(ent_embs.shape)
        # print(index_for_ent_emb.shape)
        # print(user_embs.shape)
        # exit()

        # 각 유저의 GCRNN 후 embeddings
        # *** userid 순으로 정렬됨 ***
        # train_click_num만큼 복사해줘야 함
        
        # u_time_embs = torch.cat([user_emb_0, user_embs]) # (N, emb_dim)   왜 합쳤니???????????? 난 어떻게 해야 하지...

        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch] # (N, emb_dim)
        # 원본: target_c_embs = self.ent_embedding_layer(torch.cat(comp_target_0).to(self.device0) + self.user_id_max + 1) # (N, emb_dim)
        # comp_target_0: 각 회사별 첫 번째 시간대의 news embedding의 indicies를 먼저 각각 하나의 list에 추가하고, 이후 각 회사별 첫 번째 외의 시간대 tensor가 순서대로 쌓여 있음
        
        """
        user_embs: (click 수, emb_dim)
        candidate_n_embs: (click 수, 9, emb_dim)
        내적 후 score: (click 수, 9)
        label: (click 수,) (예: 모든 값이 0, 즉 첫 번째 후보가 정답)
        """
        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch]   # (target_news_num, emb_dim); target_news_num은 batch마다 다름
        # target_score = torch.matmul(user_embs, target_n_embs.transpose(1,0))   # (batch_size=500, target_news_num)
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # g.nodes['news'].data['node_emb']는 news_int 순서대로 embedding 저장한 텐서; shape: (news_num, emb_dim)
        # candidate_n_embs: (train_click_num, (1 + 4), emb_dim); 1: target, 4: ns sample 수
        # ns_idx: (train_click_num, 5)
        candidate_user_embs = user_embs#[user_score_idx]   # user_score_idx: (train_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (train_click_num, 1, 128)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # print("candidate_score shape:", candidate_score.shape)
        # candidate_n_embs: (train_click_num, emb_dim)*(train_click_num, 5, emb_dim)
        # candidate_score: (train_click_num, 5)
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   

        return loss