# model_name.py



"""
2/27 17시 17분
check, batch size 로직 두 개랑 print들만 바꿈

해야 할 일 (2/28 1시 2분)
1. validation, test 가능하도록 main, GCRNN 코드 작성 - 완
2. chatgpt 성주dgl의 맨 위 쓰레드 참고해서 test에서만 metrics 계산하도록 코드 작성 - 완
2*** test(5일)와 validation(2일)의 유저 및 뉴스 수가 바뀌어서 문제가 생길 수 있음 - 검토 필요! (완)
3. validation set 및 test set 만드는 코드 작성 - 완
3-1) make_test_datas.py에서 validation set도 만들기 - 완
3-2) ns_idx.py를 함수화해서 main에서 import하여 사용할 수 있도록 하기 (input: batch_size) - 완

"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import random
# import matplotlib.pyplot as plt
from model.config_1w import Config
if Config.hop == 1:
    from model.GCRNN import GCRNN
elif Config.hop == 2:
    from model.GCRNN_for_2hop_ver2 import GCRNN
else:
    from model.GCRNN_for_3hop import GCRNN    
from utils.make_train_datas_1w import make_train_datas
from utils.make_test_datas_1w import make_test_datas
from utils.time_split_batch import split_train_graph
from utils.ns_indexing import ns_indexing
from utils.EarlyStopping import EarlyStopping
from utils.evaluate import ndcg_score, mrr_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import dgl
import wandb


os.environ["WANDB_API_KEY"] = "632a992df3cb5a9e7c74dce28e08a8d01229018e"
os.environ['WANDB_MODE'] = "offline"


random_seed = Config.seed
random.seed(random_seed)
np.random.seed(random_seed); torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # 0) device 및 batch_size 설정
    torch.cuda.set_device(Config.gpu_num)
    device = torch.device(f"cuda:{Config.gpu_num}" if torch.cuda.is_available() else "cpu")
    original_batch_size = Config.batch_size
    ### window size에 따른 snapshots 수 계산
    interval_minutes = Config.interval_minutes
    interval_hours = interval_minutes / 60
    snapshot_weeks = 6/7#18/7   ### history + train
    snapshots_num = int(snapshot_weeks * 7 * 24 / interval_hours)   # 2016
    print("snapshots_num:", snapshots_num)
    # device = torch.device("cpu")


    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    print(dgl.__version__)
    
    
    """
    경로 수정
    - split_train_graph에서 새로 만든 graph로 수정
    """
    ### history + train snapshots
    g, splitted_g = split_train_graph(
        snapshot_weeks,
        interval_hours, 
        f'psj/Adressa_1w/datas/total_graph_full_reciprocal_{interval_minutes}m.bin'
    )
    # print(g.number_of_nodes())
    # exit()
    # with open('./psj/Adressa_4w/train/train_datas.pkl', 'rb') as f:
    #     datas = pickle.load(f)
    datas = make_train_datas(interval_minutes=interval_minutes)
    train_news, train_category, train_time = zip(*datas)


    # 사전 학습된 단어 로드
    word2int = pd.read_csv(os.path.join('psj/Adressa_4w/history/', 'word2int.tsv'), sep='\t')
    word_to_idx = word2int.set_index('word')['int'].to_dict()
    embedding_file_path = os.path.join('psj/Adressa_4w/history/', 'pretrained_word_embedding.npy')
    embeddings = np.load(embedding_file_path)
    pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)   # (330900, 100)
    
    # df 로드
    def tokenize_title(title: str) -> list:
        """
        2.2) 타이틀을 공백 기준으로 단순 토크나이징
        """
        return title.split()

    """
    file_path: 사용할 데이터로 수정
    criteria time 변경
    
    <df에 존재하는 뉴스들만 포함하도록 combined_news_df를 바꾸는 코드>
    - 이게 필요한지 고민
    -- news2int를 기존 그대로 사용했기 때문에, user를 제외한 모든 이런 정보들은 그대로 두는 것이 좋아보임
    clicked_news_ids = df['clicked_news'].unique()
    combined_news_df = combined_news_df[combined_news_df['clicked_news'].isin(clicked_news_ids)].reset_index(drop=True)
    """
    file_path = 'psj/Adressa_1w/datas/1w_behaviors.tsv'
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
    df['click_time'] = pd.to_datetime(df['click_time'])
    df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    # df['category'] = df['category'].fillna('No category|No subcategory')
    # df[['category', 'subcategory']] = df['category'].str.split('|', n=1, expand=True)
    
    # 3개의 df를 합치기 (ignore_index=True로 인덱스 재설정) - 모든 뉴스 고려
    # 전체 뉴스 정보 로드
    combined_news_df = pd.read_csv(
        'psj/Adressa_1w/datas/all_news.tsv',   # _nyheter_splitted
        sep='\t'
    ).rename(columns={'newsId': 'clicked_news'})
    
    all_news_ids = pd.read_csv('psj/Adressa_1w/datas/news2int.tsv', sep='\t')['news_id']
    news_num = len(all_news_ids)
    user2int_df = pd.read_csv(os.path.join('psj/Adressa_1w/datas/', 'user2int.tsv'), sep='\t')
    user_num = len(user2int_df['user_int'])
    all_users = [i for i in range(user_num)]
    
    # 뉴스별 제목 집계
    news_info = combined_news_df.groupby('clicked_news', as_index=False).agg({
        'title': 'first',
        'category': 'first',
        'subcategory': 'first'
    })
    # print(news_info)

    # title -> token -> index
    news_info['title_words'] = news_info['title'].apply(tokenize_title)
    news_info['title_idx'] = news_info['title_words'].apply(
        lambda words: [word_to_idx[w.lower()] if w.lower() in word_to_idx else 0 for w in words]
    )
    
    # category, subcategory -> index
    category2int = pd.read_csv('psj/Adressa_1w/datas/category2int.tsv', sep='\t')   # _nyheter_splitted_for_NE    
    cat_num = Config.num_categories
    
    # category와 subcategory 매핑 딕셔너리 생성
    category_map = category2int.set_index('category')['int'].to_dict()
    news_info['category_idx'] = news_info['category'].map(category_map)
    news_info['subcategory_idx'] = news_info['subcategory'].map(category_map)

    # # 3) 범위 검증
    # max_idx = int(max(news_info['category_idx'].max(),
    #                 news_info['subcategory_idx'].max()))
    # assert max_idx < Config.num_categories, f"Config.num_categories({Config.num_categories}) must be > max idx {max_idx}"
    
    
    # news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
    news_info_df = news_info[['clicked_news', 'title_idx', 'category_idx', 'subcategory_idx']].rename(
        columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
    )

    # news_id_to_info
    news_id_to_info = news_info_df.set_index('news_id')\
        [['title_idx', 'category_idx', 'subcategory_idx']].to_dict(orient='index')
        # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장
    
    # news2int_df = pd.read_csv('./psj/Adressa_4w/history/news2int.tsv', sep='\t')
    # df2 = pd.merge(df, news2int_df, left_on='clicked_news', right_on='news_id', how='left')
    # df2 = df2.sort_values('news_int')

    # news2int = {nid[1:]: i for i, nid in enumerate(all_news_ids)}


    ### Loading idx_infos for calculating NLL loss
    train_ns_idx_batch, _ = ns_indexing('psj/Adressa_1w/train/train_ns.tsv', original_batch_size, user_num=user_num)
    # train_user_idx_batch = torch.load('./psj/Adressa_4w/train/train_user_idx_batch.pt')   # 사실 얘는 필요 없음...

    
    # test data 로드 시작 ---------------------------------
    # with open('./psj/Adressa_4w/test/validation_datas.pkl', 'rb') as f:
    #     datas = pickle.load(f)
    """
    ns_indexing 파일 경로 수정
    """
    test_datas = make_test_datas(snapshots_num=snapshots_num)
    test_news, test_time, test_empty_check = zip(*test_datas)
    test_ns_idx_batch, test_cand_score_weight_batch = ns_indexing('psj/Adressa_1w/test/test_ns.tsv', original_batch_size, user_num=user_num, test=True)
    
    # with open('./psj/Adressa_4w/test/test_datas.pkl', 'rb') as f:
    #     datas = pickle.load(f)
    # test_news, test_time, test_empty_check = zip(*test_datas)
    # test_ns_idx_batch = ns_indexing('./psj/Adressa_4w/test/test_ns.tsv', original_batch_size)
    
    
    print("data loading finished!")
    # 필요한 정보 로드 끝 -------------------------------------------------------------------------------------------------
    
    # 2) 모델에 필요한 정보 추가 준비
    learning_rate = 0.0001
    num_epochs = 10
    batch_size = original_batch_size
    batch_num = user_num // batch_size if user_num % batch_size == 0 else user_num // batch_size + 1
    emb_dim = Config.num_filters*3   # 300
    history_length = 100
    # snapshots_num = snapshot_weeks * 7 * 24 * 2   # 2016

    # wandb 초기화 및 config 설정
    wandb.init(project="TKG_for_NewsRec_1w", config={
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": original_batch_size,
        "emb_dim": emb_dim,
        "history_length": history_length,
        "snapshot_weeks": snapshot_weeks,
        "snapshots_num": snapshots_num,
    }, name=f"{int(snapshot_weeks) + 1}w Adressa_method_{Config.method}_score adjust_{Config.adjust_score}_batch size {original_batch_size}_seed {random_seed}")
    
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
        snapshots_num=snapshots_num,
        config=Config
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.01)   
    
    # 모델 파라미터 및 그라디언트 로깅 설정 (옵션)
    wandb.watch(model, log="all")
    
    # 학습 과정에서의 로스 기록
    all_losses = []

    # 베스트 성능 저장을 위한 변수들
    best_score = -1.0
    best_epoch = 0
    best_auc = 0.0
    best_mrr = 0.0
    best_ndcg5 = 0.0
    best_ndcg10 = 0.0

    # (2) EarlyStopping 객체 생성
    early_stopper = EarlyStopping(
        emb_dim=emb_dim,      # emb_dim 등 모델 설정에 맞춰 전달
        patience=3,           # 개선 없으면 5epoch 후 스탑(예시)
        min_delta=1e-4,
        ckpt_dir=f'psj/Adressa_1w/1_hop_ckpt/bs_{original_batch_size}_lr_{learning_rate}_seed_{random_seed}', 
        verbose=True,
        save_all=False        # True로 설정하면 매 epoch마다 체크포인트 저장
    )

    # 4) Batch 학습을 통해 train 수행
    print(f"# of batch: {batch_num}, # of user: {user_num}, "
          f"batch size: {batch_size}, lr: {learning_rate}, "
          f"embedding dim: {emb_dim}, history_length: {history_length}",
          f"window size: {round(Config.interval_minutes/60, 2)}h",
          f"snapshots_num: {snapshots_num}",
          f"# of hops: {Config.hop}\n")
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        prev_batch_cnt = 0
        batch_cnt = 0
        batch_size = original_batch_size
        for b in tqdm(range(batch_num), desc=f'training {epoch} epoch batches'):
            prev_batch_cnt = batch_cnt
            batch_cnt += batch_size
            # prev_batch = b * batch_size
            if batch_cnt > len(train_news):
                batch_cnt = len(train_news)
            real_batch_size = batch_cnt - prev_batch_cnt

            # batch_cnt = 6000
            # prev_batch_cnt = 5500
            # b = 11
            # train_users[b], train_news[b], 없애버림
            # ns_val = train_ns_idx_batch[b] + model.user_num  # shape: (train_click_num, k_neg)
            # print("min(ns_val) =", ns_val.min().item(), "max(ns_val) =", ns_val.max().item())
            
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
            loss.backward()   # calculate gradient    
            
            # # --- gradient 확인 코드 시작 ---
            # if b <= 5:
            #     total_norm = 0.0
            #     for name, param in model.named_parameters():
            #         if 'user_embedding_layer.weight' in name:
            #             param.register_hook(lambda g, n=name: print(f"{n} grad norm =", g.norm()))        
                
                #     if param.grad is not None:
                #         # L2 노름 계산
                #         grad_norm = param.grad.data.norm(2).item()
                #         total_norm += grad_norm ** 2
                #         print(f"[Grad] {name:40s} | norm: {grad_norm:.6f}")
                # total_norm = total_norm ** 0.5
                # print(f"[Grad] 전체 gradient L2 norm: {total_norm:.6f}")
                # # wandb에 로깅
                # wandb.log({"grad_norm": total_norm}, step=epoch)
                # print(f"user embedding layer: {model.user_embedding_layer.weight[:10]}")
                # print(f"c0_embedding_layer_u: {model.c0_embedding_layer_u.weight[:10]}")
                # print(f"user_cn: {model.user_cn[:10]}")
                # print(f"user_hn: {model.user_hn[:10]}")
                # print(f"user_cn: {model.user_cn[:10]}")
                
                # print()
                # # user embedding, c0 embedding 파라미터 grad 확인
                # ue_grad = model.user_embedding_layer.weight.grad   # (user_num, emb_dim)
                # c0_grad = model.c0_embedding_layer_u.weight.grad   # (user_num+news_num, emb_dim)
                # print("user_embedding_layer grad norm:", ue_grad.norm().item())
                # print("c0_embedding_layer_u grad norm:", c0_grad.norm().item())
            # --- gradient 확인 코드 끝 ---
                  
            optimizer.step()   # update parameter via calculated gradient
            optimizer.zero_grad()   # initialize gradient
            
            all_losses.append(loss.item())
            epoch_loss_sum += loss.item()
            epoch_samples += 1
            
            # 메모리 해제
            # del splitted_subgraphs[b]
            # if b % 5 == 0:
            #     torch.cuda.empty_cache()
            
        # epoch이 끝난 시점에서 epoch_loss 계산
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

            for test_b in tqdm(range(test_batch_num), desc=f'Testing Epoch {epoch}', miniters=5, leave=False):
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
                if Config.adjust_score:
                    test_cand_score_weight = np.array(test_cand_score_weight_batch[test_b])
                    assert candidate_score.shape == test_cand_score_weight.shape
                for i in range(candidate_score.shape[0]):
                    y_score = candidate_score[i]
                    ### 수명 고려한 스코어 조정
                    if Config.adjust_score:
                        y_score = y_score*test_cand_score_weight[i]
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

            # wandb에 테스트 지표 로깅
            wandb.log({
                # "epoch": epoch,
                "train_loss": epoch_loss,
                "auc": final_auc,
                "mrr": final_mrr,
                "ndcg5": final_ndcg5,
                "ndcg10": final_ndcg10,
                "avg_score": avg_metric,
                "empty_batch_count": empty_batch_count,
            })
            
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
    print(f"window size: {round(Config.interval_minutes/60, 2)}h")

    # wandb 세션 종료
    wandb.finish()

if __name__ == "__main__":
    main()