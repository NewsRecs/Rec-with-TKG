import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
from make_train_datas import make_train_datas
from make_test_datas import make_test_datas
from time_split_batch import split_train_graph
from psj.GCRNN import GCRNN
from ns_indexing import ns_indexing
from psj.EarlyStopping import EarlyStopping
from evaluate import ndcg_score, mrr_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from psj.config import Config
import dgl


def main():
    # 0) device 및 batch_size 설정
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 500
    # device = torch.device("cpu")


    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    print(dgl.__version__)
    
    
    
    g, splitted_g = split_train_graph(5)
    # with open('./psj/Adressa_4w/train/train_datas.pkl', 'rb') as f:
    #     datas = pickle.load(f)
    datas = make_train_datas()
    train_news, train_category, train_time = zip(*datas)
    all_users = [i for i in range(84989)]


    # 사전 학습된 단어 로드
    word2int = pd.read_csv(os.path.join('./psj/Adressa_4w/history/', 'word2int.tsv'), sep='\t')
    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('./psj/Adressa_4w/history/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)   # (330900, 100)
    
    # df 로드
    def tokenize_title(title: str) -> list:
        """
        2.2) 타이틀을 공백 기준으로 단순 토크나이징
        """
        return title.split()
    
    # train, test의 news 데이터 로드
    train_news_file_path = './psj/Adressa_5w/train/news.tsv'
    train_news_df = pd.read_csv(train_news_file_path, sep='\t', header=None)
    train_news_df.columns = ['clicked_news', 'category', 'subcategory', 'title', 'body', 'identifier', 'publish_time', 'click_time']
    sub_train_news_df = train_news_df[['clicked_news', 'category', 'subcategory', 'title']]
    
    test_news_file_path = './psj/Adressa_5w/test/news.tsv'
    test_news_df = pd.read_csv(test_news_file_path, sep='\t', header=None)
    test_news_df.columns = ['clicked_news', 'category', 'subcategory', 'title', 'body', 'identifier', 'publish_time', 'click_time']
    sub_test_news_df = test_news_df[['clicked_news', 'category', 'subcategory', 'title']]
    
    file_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    df['category'] = df['category'].fillna('No category|No subcategory')
    df[['category', 'subcategory']] = df['category'].str.split('|', n=1, expand=True)
    sub_history_news_df = df[['clicked_news', 'category', 'subcategory', 'title']]

    # 3개의 df를 합치기 (ignore_index=True로 인덱스 재설정) - 모든 뉴스 고려
    combined_news_df = pd.concat([sub_history_news_df, sub_train_news_df, sub_test_news_df], ignore_index=True)
    
    user_num = len(df['history_user'].unique())
    # 뉴스별 제목 집계
    news_info = combined_news_df.groupby('clicked_news', as_index=False).agg({
        'title': 'first',
        'category': 'first',
        'subcategory': 'first'
    })

    # title -> token -> index
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [word_to_idx[w.lower()] if w.lower() in word_to_idx else 0 for w in words]
    )
    
    # category, subcategory -> index
    category2int = pd.read_csv('./psj/Adressa_4w/history/category2int_pio.tsv', sep='\t')    
    cat_num = Config.num_categories
    news_info['category_idx'] = news_info['category'].map(category2int.to_dict())
    news_info['subcategory_idx'] = news_info['subcategory'].map(category2int.to_dict())
    
    
    # news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
    news_info_df = news_info[['clicked_news', 'title_idx', 'category_idx', 'subcategory_idx']].rename(
        columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
    )

    # news_id_to_info
    news_id_to_info = news_info_df.set_index('news_id')\
        [['title_idx', 'category_idx', 'subcategory_idx']].to_dict(orient='index')
        # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장
    
    news2int_df = pd.read_csv('./psj/Adressa_4w/history/news2int.tsv', sep='\t')
    all_news_ids = news2int_df['news_id'].unique()
    news_num = len(all_news_ids)

    ### Loading idx_infos for calculating NLL loss
    train_ns_idx_batch = ns_indexing('./psj/Adressa_4w/train/train_ns.tsv', batch_size)
    # train_user_idx_batch = torch.load('./psj/Adressa_4w/train/train_user_idx_batch.pt')   # 사실 얘는 필요 없음...

    
    # test data 로드 시작 ---------------------------------
    val_datas, test_datas = make_test_datas()
    validation_news, validation_time, validation_empty_check = zip(*val_datas)
    validation_ns_idx_batch = ns_indexing('./psj/Adressa_4w/test/validation_ns.tsv', batch_size)
    
    test_news, test_time, test_empty_check = zip(*test_datas)
    test_ns_idx_batch = ns_indexing('./psj/Adressa_4w/test/test_ns.tsv', batch_size)
    
    
    print("data loading finished!")
    # 필요한 정보 로드 끝 -------------------------------------------------------------------------------------------------
    
    # 2) 모델에 필요한 정보 추가 준비
    learning_rate = 0.0001
    num_epochs = 5
    batch_size = 500
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    emb_dim = Config.num_filters*3   # 300
    history_length = 100
    
    es = EarlyStopping(
        emb_dim=emb_dim,
        patience=3,
        min_delta=1e-4,
        ckpt_dir=f"./Adressa_7w/train/ckpt/val3.5_seed_1024_hl_{history_length}_ndcg",  # 체크포인트 저장 디렉토리
        verbose=True,
        save_all=True  # 모든 epoch마다 저장
    )
    all_losses = []
    
    # 3) GCRNN 모델 초기화
    model = GCRNN(all_news_ids, news_id_to_info, user_num, cat_num, news_num, pretrained_word_embedding=pretrained_word_embedding, emb_dim=emb_dim, batch_size=500)  
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.01)   
    
    best_loss = float('inf')   # 7.337717744170642
    check = 0
    validation_losses = []   # 7.337717744170642, 7.415253834646256
    
    # 4) Batch 학습을 통해 train 수행
    print("Train start !")
    print(f"# of batch: {batch_num}, # of user: {user_num}, batch size: {batch_size}, lr: {learning_rate}, embedding dim: {emb_dim}, history_length: {history_length} \n")
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        prev_batch_cnt = 0
        batch_cnt = 0
        batch_size = 500
        for b in tqdm(range(batch_num), desc=f'training {epoch} epoch batches'):
            prev_batch_cnt = batch_cnt
            batch_cnt += batch_size
            # prev_batch = b * batch_size
            if batch_cnt > len(train_news):
                batch_cnt = len(train_news)
            batch_size = batch_cnt - prev_batch_cnt

            loss = model(all_users[prev_batch_cnt:batch_cnt], train_news[prev_batch_cnt:batch_cnt], train_category[prev_batch_cnt:batch_cnt], train_time[prev_batch_cnt:batch_cnt], g, splitted_g, train_ns_idx_batch[b], history_length)
            loss.backward()   # calculate gradient
            
            optimizer.step()   # update parameter via calculated gradient
            optimizer.zero_grad()   # initialize gradient
            
            all_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_samples += 1
            
        # epoch이 끝난 시점에서 epoch_loss 계산
        epoch_loss = epoch_loss_sum / (epoch_samples if epoch_samples else 1)
        print(f"[Epoch {epoch}] avg train_loss={epoch_loss:.6f}")
    
        # 5) validation 수행
        print("Validation start!")
        
        model.eval()
        with torch.no_grad():
            list_ndcg5_val = []
            epoch_losses = []
            # validation start
            prev_validation_batch_cnt = 0
            validation_batch_cnt = 0
            n_empty = 0
            batch_size = 500
            validation_batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
            for validation_b in tqdm(range(validation_batch_num)):   # 170개
                prev_validation_batch_cnt = validation_batch_cnt
                validation_batch_cnt += batch_size
                if validation_batch_cnt > len(validation_news):
                    validation_batch_cnt = len(validation_news)
                batch_size = validation_batch_cnt - prev_validation_batch_cnt
                # 해당 batch user의 클릭 기록이 하나도 없는 경우 loss 계산 pass
                if not any(validation_empty_check[prev_validation_batch_cnt:validation_batch_cnt]):
                    n_empty += 1
                    continue
                candidate_score, validation_loss = model.inference(all_users[prev_validation_batch_cnt:validation_batch_cnt], validation_news[prev_validation_batch_cnt:validation_batch_cnt], validation_time[prev_validation_batch_cnt:validation_batch_cnt], g, splitted_g, validation_ns_idx_batch[validation_b], history_length)
                epoch_losses.append(validation_loss.item())
                
                
                candidate_score = candidate_score.cpu().numpy()
                real_batch_size = candidate_score.shape[0]
                num_candidates  = candidate_score.shape[1]
                for i in range(real_batch_size):
                    # y_score: i-th 유저에 대한 각 candidate 스코어
                    y_score = candidate_score[i]

                    # y_true: [1, 0, 0, ...] 꼴로 가정 (정답이 첫 열인 경우)
                    y_true = np.zeros(num_candidates, dtype=int)
                    y_true[0] = 1

                    # B. row-wise MRR, nDCG
                    list_ndcg5_val.append(ndcg_score(y_true, y_score, k=5))
            
            val_ndcg5   = np.mean(list_ndcg5)
                
            print("# of validation empty batch:", n_empty)
        mean_epoch_loss = np.mean(np.array(epoch_losses))
        

        validation_losses.append(mean_epoch_loss)
        print(f"Validation_losses: {validation_losses}")
        # EarlyStopping 체크 (train_loss를 임시로 val_loss처럼 사용)
        es(val_ndcg5, model, epoch, learning_rate)
        if es.early_stop:
            print("[EarlyStopping] Training is stopped early.")
            print(f"Validation_losses: {validation_losses}")
            print(f"Validation_nDCG@5: {val_ndcg5}")
            break

    print("\n=== Training finished. Loading best checkpoint for Test ===")
    best_ckpt = es.best_ckpt_path  # early stopping에서 저장한 best model 경로
    print(f"Best checkpoint path: {best_ckpt}")
        
    with torch.no_grad():
        # 베스트 모델 로드
        if best_ckpt is not None and os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt))
            model.eval()
        else:
            print("[Warning] best checkpoint not found. Evaluate current model instead.")
            
        batch_size = 500
        test_batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
        for m in range(1,2):
            prev_test_batch_cnt = 0
            test_batch_cnt = 0
            all_scores = []   # 모든 batch scores
            all_labels = []   # 모든 batch의 실제 labels (row별로 [1, 0, 0, ...])
            list_mrr   = []
            list_ndcg5 = []
            list_ndcg10 = []
            n_empty = 0
            for test_b in tqdm(range(test_batch_num)):
                prev_test_batch_cnt = test_batch_cnt
                test_batch_cnt += batch_size
                if test_batch_cnt > len(test_news):
                    test_batch_cnt = len(test_news)
                batch_size = test_batch_cnt - prev_test_batch_cnt
                # 해당 batch user의 클릭 기록이 하나도 없는 경우를 방지
                if not any(test_empty_check[prev_test_batch_cnt:test_batch_cnt]):
                    n_empty += 1
                    continue
                candidate_score, test_loss = model.inference(all_users[prev_test_batch_cnt:test_batch_cnt], test_news[prev_test_batch_cnt:test_batch_cnt], 
                                                             test_time[prev_test_batch_cnt:test_batch_cnt], g, splitted_g, test_ns_idx_batch[test_b], history_length)
                candidate_score = candidate_score.cpu().numpy()
                real_batch_size = candidate_score.shape[0]
                num_candidates  = candidate_score.shape[1]

                for i in range(real_batch_size):
                    # y_score: i-th 유저에 대한 각 candidate 스코어
                    y_score = candidate_score[i]

                    # y_true: [1, 0, 0, ...] 꼴로 가정 (정답이 첫 열인 경우)
                    y_true = np.zeros(num_candidates, dtype=int)
                    y_true[0] = 1

                    # A. AUC 계산 위해서는 전체 스코어를 flatten 해서 쌓아두는 게 일반적
                    #   (row-wise로 AUC 계산하고 평균내도 되긴 합니다만, 보통 전체를 한 번에 보기도 함)
                    all_scores.extend(y_score)
                    all_labels.extend(y_true)

                    # B. row-wise MRR, nDCG
                    list_mrr.append(mrr_score(y_true, y_score))
                    list_ndcg5.append(ndcg_score(y_true, y_score, k=5))
                    list_ndcg10.append(ndcg_score(y_true, y_score, k=10))
            print("# of empty batch:", n_empty)
                
            final_auc     = roc_auc_score(all_labels, all_scores)
            final_mrr     = np.mean(list_mrr)
            final_ndcg5   = np.mean(list_ndcg5)
            final_ndcg10  = np.mean(list_ndcg10)

            print(f"Final Test Loss: {test_loss.item()}")
            print("[Final Test Metrics]")
            print(f"AUC        : {final_auc:.4f}")
            print(f"MRR        : {final_mrr:.4f}")
            print(f"nDCG@5     : {final_ndcg5:.4f}")
            print(f"nDCG@10    : {final_ndcg10:.4f}")


if __name__ == "__main__":
    main()
