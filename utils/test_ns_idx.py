### train_user_tensor, train_news_tensor, g, splitted_g 구성
import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import pickle
import dgl
# from ..model.config import Config

"""
train_ns.tsv를 만드는 파일 (tkg_train_negative_samples_lt36_ns4 -> train_ns; 즉, 후보 뉴스들를 int로 매핑)

test의 data를 처리할 때는 
train_news_file_path, train_ns_path, train_file_path, train_ns_idx_batch와 train_user_idx_batch 그리고 train_batch를 저장하는 paths
를 바꿔줘야 함!!!
"""

"""
category 이미 추가했으니 batch 데이터에서 고려해주기만 하면 됨!!
"""



# 각 뉴스의 카테고리만 가져오기
train_news_file_path = './psj/Adressa_4w/history/all_news_nyheter_splitted.tsv'
train_news_df = pd.read_csv(train_news_file_path, sep='\t')
train_news_df.columns = ['index_col', 'newsId','category','subcategory', 'title']
sub_train_news_df = train_news_df[['newsId', 'category']]

# train_ns data 가져오기
train_ns_path = "./psj/Adressa_4w/test/tkg_test_negative_samples_lt36_ns20.tsv"   # half
train_ns = pd.read_csv(train_ns_path, sep='\t')

# a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
# train_file_path = './psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
# test_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
# ### half 버전!!!!!!!!!!!!!!!!!!!!!!!!!
# test_time = pd.Timestamp('2017-02-15 20:00:01')
# test_df['click_time'] = pd.to_datetime(test_df['click_time'])
# test_df  = test_df[test_df['click_time'] <= test_time]

# 'clicked_newsId'를 기준으로 'category' 매칭
# test_df = test_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')

# train_users = test_df['history_user']

# print(train_df.head())
# print(train_df.columns)
# print(train_df.isna().sum())
# print(train_df[train_df.isna()]) # 하나가 데이터 오류
# print(len(train_df))
# exit()

# train_df에서 nan이 존재하는 행 제거
# test_df = test_df.dropna(subset=['clicked_news'])
# test_df = test_df[test_df.notna()]




########################################### 여기부터 negative sampling을 위해 추가된 부분

# news2int 가져오기
news2int_file_path = './psj/Adressa_4w/history/news2int.tsv'
news2int = pd.read_csv(news2int_file_path, sep='\t')
# news2int를 dictionary로 변환
news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
# user2int mapping
# file_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
# history_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
# history_df['click_time'] = pd.to_datetime(history_df['click_time'])
# end_time = pd.Timestamp('2017-02-05 08:00:01')
# history_df = history_df[history_df['click_time'] <= end_time]   # 정확히 5주 데이터만 사용하도록 필터링
# history_data_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
# df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
# df = df.dropna(subset=['clicked_news'])
# users = df['history_user'].unique()
# df_criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
# df_criteria_time2 = pd.Timestamp('2017-01-26 00:00:00')
# df = df[(df_criteria_time1 <= df['click_time']) & (df['click_time'] < df_criteria_time2)]
# all_users = sorted(df['history_user'].unique())  # 정렬된 리스트
# user2int_df = pd.DataFrame({
#     'user_id': all_users,
#     'user_int': range(len(all_users))
# }) 
user2int_df = pd.read_csv('./psj/Adressa_4w/history/user2int.tsv', sep='\t')
user2int = dict(zip(user2int_df['user_id'], user2int_df['user_int']))
# user2int_df.to_csv('./psj/Adressa_1w/datas/user2int.tsv', sep='\t', index=False)


# train_df에 ns와 int 추가
train_ns['news_int'] = train_ns['clicked_news'].map(news2int_mapping)
def map_negative_samples(ns_str):
    if pd.isna(ns_str):
        return ns_str
    # 공백으로 분리하여 개별 뉴스 아이디 리스트 생성
    news_ids = ns_str.split()
    # 각 뉴스 아이디를 매핑 (없는 경우 -1 또는 원하는 기본값으로 대체)
    news_ints = [str(news2int_mapping.get(nid, -1)) for nid in news_ids]
    # 다시 공백으로 연결하여 문자열 형태로 반환
    return " ".join(news_ints)
train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
train_ns['user_int'] = train_ns['user'].map(user2int)
# test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)


# ===== 여기서부터 publish_time 추가 =====
# publish_time 로드 및 사전(dict)화
import numpy as np

pub_path = "./psj/Adressa_4w/datas/news_publish_times.tsv"
pub_df   = pd.read_csv(pub_path, sep='\t', usecols=['news_id', 'publish_time'])

# ➊ publish_time 타입 보정 (이미 datetime이면 자동으로 넘어감)
pub_df['publish_time'] = pd.to_datetime(pub_df['publish_time'])
pub_df['news_int'] = pub_df['news_id'].map(news2int_mapping)
pub_dict = dict(zip(pub_df['news_int'], pub_df['publish_time']))  

# ➋ click_time # publish_times 타입 보정
train_ns['click_time'] = pd.to_datetime(train_ns['click_time'])
train_ns['publish_times_list'] = (
    train_ns['publish_times']
    .str.split(',')                                    # ["2017-01-22 05:07:45", ...]
    .apply(lambda lst: pd.to_datetime(lst))  
    # errors='coerce'를 주면 파싱 실패 시 NaT로 처리
)

train_ns['pos_publish_time'] = train_ns['news_int'].map(pub_dict)

# train_ns['publish_times'] = pd.to_datetime(train_ns['publish_times'])


THRESHOLD_HOUR = 36 #72#48

all_diffs_hr = []          # 클릭~퍼블리시 차이(시간) 누적
all_neg_diffs = []
for _, row in train_ns.iterrows():
    click_t = row['click_time']

    # ① positive
    pos_pub = row['pos_publish_time']
    if isinstance(pos_pub, pd.Timestamp):
        all_diffs_hr.append((click_t - pos_pub).total_seconds() / 3600)

    # ② negatives
    for neg_pub_t in row['publish_times_list']:
        if isinstance(neg_pub_t, pd.Timestamp):
            all_diffs_hr.append((click_t - neg_pub_t).total_seconds() / 3600)
            all_neg_diffs.append((click_t - neg_pub_t).total_seconds()/3600)
# --- 결과 ------------------------------------------------------
all_diffs_hr = np.array(all_diffs_hr)                  # ndarray 변환
avg_diff_hr  = all_diffs_hr.mean()                     # (1) 평균
ratio_over36 = (all_diffs_hr > THRESHOLD_HOUR).mean()  # (2) 비율
all_neg_diffs = np.array(all_neg_diffs)
ratio_neg_over36 = (np.array(all_neg_diffs) > THRESHOLD_HOUR).mean() * 100

print(f"평균 클릭 & 퍼블리쉬 시간 차이: {avg_diff_hr:.2f} h")
print(f"{THRESHOLD_HOUR} 시간 초과 후보 비율  : {ratio_over36*100:.2f} %")
print()
print(len(all_diffs_hr))
print()
print(ratio_neg_over36) 
print()
# exit()


ALPHA = 0.1#Config.ALPHA              # 0.2
LIFETIME_MIN = THRESHOLD_HOUR #* 60     # 36h

def weight_by_lifetime(click_t, pub_t):
    remaining = LIFETIME_MIN - abs((click_t - pub_t).total_seconds()) / 60 / 60  # 음수 허용 x
    return 1 / (1 + np.exp(-ALPHA * remaining))     # remaining<0 → σ<0.5

def compute_row_weight(row):
    click_t = row['click_time']
    weights = []

    # ① positive
    pos_pub = row['pos_publish_time']
    weights.append(f'{weight_by_lifetime(click_t, pos_pub):.6f}')

    # ② negatives
    for neg_pub_t in row['publish_times_list']:
        # pub_t = pub_dict[int(nid)]
        weights.append(f'{weight_by_lifetime(click_t, neg_pub_t):.6f}')

    return ' '.join(weights)


# ➍ weight 열 생성
train_ns['candidate_weight'] = train_ns.apply(compute_row_weight, axis=1)

train_ns.drop(columns=['pos_publish_time', 'publish_times_list'], inplace=True)

# train_ns 저장
train_ns.to_csv('./psj/Adressa_4w/test/test_ns.tsv', sep='\t', index=False)   # _half



# ### train_user_tensor, train_news_tensor, g, splitted_g 구성
# import torch
# import pickle  
# import os
# from tqdm import tqdm
# import pandas as pd
# import pickle
# import dgl

# """
# test_ns.tsv를 만드는 파일 (tkg_test_negative_samples_lt36_ns20 -> validation_ns, test_ns; 즉, 후보 뉴스들를 int로 매핑)

# test의 data를 처리할 때는 
# train_news_file_path, train_ns_path, train_file_path, train_ns_idx_batch와 train_user_idx_batch 그리고 train_batch를 저장하는 paths
# 를 바꿔줘야 함!!!
# """

# """
# category 이미 추가했으니 batch 데이터에서 고려해주기만 하면 됨!!
# """



# # 각 뉴스의 카테고리만 가져오기
# train_news_file_path = './psj/Adressa_4w/history/all_news_nyheter_splitted.tsv'
# train_news_df = pd.read_csv(train_news_file_path, sep='\t', header=None)
# train_news_df.columns = ['newsId','category','subcategory', 'title']
# sub_train_news_df = train_news_df[['newsId', 'category']]

# # news2int 가져오기
# news2int_file_path = './psj/Adressa_4w/history/news2int.tsv'
# news2int = pd.read_csv(news2int_file_path, sep='\t')

# # train_ns data 가져오기
# train_ns_path = "./psj/Adressa_4w/test/tkg_test_negative_samples_lt36_ns20.tsv"
# train_ns = pd.read_csv(train_ns_path, sep='\t')

# # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
# train_file_path = './psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
# train_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
# train_users = train_df['history_user']
# # 'clicked_news' 열에서 '-1' 제거
# train_df['clicked_news'] = train_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
# # 'clicked_newsId'를 기준으로 'category' 매칭
# train_df = train_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')

# # print(train_df.head())
# # print(train_df.columns)
# # print(train_df.isna().sum())
# # print(train_df[train_df.isna()]) # 하나가 데이터 오류
# # print(len(train_df))
# # exit()

# # train_df에서 nan이 존재하는 행 제거
# train_df = train_df.dropna(subset=['clicked_news'])



# ########################################### 여기부터 negative sampling을 위해 추가된 부분
# # news2int를 dictionary로 변환
# news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))

# # users = train_df['history_user'].unique()
# # user2int = {uid: i for i, uid in enumerate(users)}
# # all_user_ids = [i for i in range(len(users))]
# # history_data_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
# # df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
# # df = df.dropna(subset=['clicked_news'])
# # users = df['history_user'].unique()
# # user2int = {uid: i for i, uid in enumerate(users)}
# user2int_df = pd.read_csv('./psj/Adressa_4w/history/user2int.tsv', sep='\t')
# user2int = dict(zip(user2int_df['user_id'], user2int_df['user_int']))

# # train_df에 ns와 int 추가
# train_ns['news_int'] = train_ns['clicked_news'].map(news2int_mapping)
# def map_negative_samples(ns_str):
#     if pd.isna(ns_str):
#         return ns_str
#     # 공백으로 분리하여 개별 뉴스 아이디 리스트 생성
#     news_ids = ns_str.split()
#     # 각 뉴스 아이디를 매핑 (없는 경우 -1 또는 원하는 기본값으로 대체)
#     news_ints = [str(news2int_mapping.get(nid, -1)) for nid in news_ids]
#     # 다시 공백으로 연결하여 문자열 형태로 반환
#     return " ".join(news_ints)
# train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
# train_ns['user_int'] = train_ns['user'].map(user2int)
# train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)

# # validation과 test 나누는 부분
# criteria_time = pd.Timestamp('2017-02-15 20:00:01')
# train_ns['click_time'] = pd.to_datetime(train_ns['click_time'])
# validation_ns_df = train_ns[train_ns['click_time'] <= criteria_time]
# test_5d_ns_df = train_ns[train_ns['click_time'] > criteria_time]

# # validation/train_ns_df 저장
# validation_ns_df.to_csv('./psj/Adressa_4w/test/validation_ns.tsv', sep='\t', index=False)
# test_5d_ns_df.to_csv('./psj/Adressa_4w/test/test_ns.tsv', sep='\t', index=False)

# exit()





# # ns에 필요한 데이터 저장하는 변수들
# train_ns_idx_batch = []
# train_user_idx_batch = []
# ########################################### negative sampling 위한 부분 끝



# news_num = len(train_df['clicked_news'])
# total_user_num = len(users)
# batch_size = 500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # batch_user_emb = []

# # *** train_batch_embs의 shape: batch 수
# # *** train_batch_embs의 각 요소: [snapshot idx, batch user embeddings]
# # A = 0
# prev_batch = 0
# batch = 0
# # train_batch = []   # 최종 정보들을 담은 리스트
# # batch_subgraph = []
# # batch_clicked_pairs = []
# # batch_e_ids = []
# # batch_seed_list = []
# batch_num = len(users) // batch_size if len(users) % batch_size == 0 else len(users) // batch_size + 1
# # batch_seed_list = [[] for _ in range(len(batch_num))]
# b_num = 0
# for b in tqdm(range(batch_num), desc="processing batches"):
#     # b_num += 1
#     # if b_num > 5:
#     #     break
#     prev_batch = b * batch_size
#     batch = min((b+1) * batch_size, total_user_num)
#     batch_user_ids = all_user_ids[prev_batch:batch]   # 0 ~ 499, 500 ~ 999, ..., 84500 ~ 84989
    
#     """
#     추가한 부분 (for negative sampling)
#     목표: user_score_idx 생성 및 저장
#     """
#     batch_ns_df = train_ns[train_ns['user_int'].isin(batch_user_ids)]
#     # user_idx 처리
#     user_tensor = torch.tensor(batch_ns_df['user_int'].tolist(), dtype=torch.long)
#     train_user_idx_batch.append(user_tensor)
#     # negative samples 포함한 news_idx 처리
#     ns_idx_list = []
#     for _, row in batch_ns_df.iterrows():
#         # positive 뉴스 id (이미 news2int 매핑된 정수값)
#         pos = int(row['news_int'])
        
#         # negative_samples 처리: 공백으로 구분된 문자열을 리스트로 변환
#         neg_str = row['negative_samples']
#         # 각 요소를 int로 변환
#         neg_list = [int(x) for x in neg_str.split()]
#         # 요소가 4개보다 적으면 패딩, 4개보다 많으면 앞의 4개만 사용
#         negs = neg_list
#         ns_idx_list.append([pos] + negs)
    
#     # 리스트를 텐서로 변환 (shape: [row_num, 5])
#     ns_idx_tensor = torch.tensor(ns_idx_list, dtype=torch.long)
#     train_ns_idx_batch.append(ns_idx_tensor)
    
#     # continue
#     """
#     추가한 부분 (for negative sampling) 끝
#     """
    


# from time import time
# print("start saving file")
# start = time()

# # ns에 필요한 idx 저장
# torch.save(train_ns_idx_batch, './Adressa_4w/test/test_ns_idx_batch.pt')
# torch.save(train_user_idx_batch, './Adressa_4w/test/test_user_idx_batch.pt')

# end = time()
# print(f"Saving file time: {(end - start) // 60}m {(end - start) % 60}s")
# print("finished!")