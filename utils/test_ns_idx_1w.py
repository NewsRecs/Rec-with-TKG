### train_user_tensor, train_news_tensor, g, splitted_g 구성
import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import pickle
import dgl

"""
test_ns.tsv를 만드는 파일 (tkg_test_negative_samples_lt36_ns20 -> validation_ns, test_ns; 즉, 후보 뉴스들를 int로 매핑)

test의 data를 처리할 때는 
train_news_file_path, train_ns_path, train_file_path, train_ns_idx_batch와 train_user_idx_batch 그리고 train_batch를 저장하는 paths
를 바꿔줘야 함!!!
"""

"""
category 이미 추가했으니 batch 데이터에서 고려해주기만 하면 됨!!
"""



# 각 뉴스의 카테고리만 가져오기
train_news_file_path = './psj/Adressa_4w/history/all_news_nyheter_splitted.tsv'
train_news_df = pd.read_csv(train_news_file_path, sep='\t', header=None)
train_news_df.columns = ['newsId','category','subcategory', 'title']
sub_train_news_df = train_news_df[['newsId', 'category']]

# news2int 가져오기
news2int_file_path = './psj/Adressa_4w/history/news2int.tsv'
news2int = pd.read_csv(news2int_file_path, sep='\t')

# train_ns data 가져오기
train_ns_path = "./psj/Adressa_1w/test/tkg_test_negative_samples_lt36_ns20.tsv"
train_ns = pd.read_csv(train_ns_path, sep='\t')

# a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
train_file_path = './psj/Adressa_1w/train/history_tkg_behaviors.tsv'
df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
# 'clicked_newsId'를 기준으로 'category' 매칭
train_df = train_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')
train_df = train_df.dropna(subset=['clicked_news'])

train_users = train_df['history_user']




########################################### 여기부터 negative sampling을 위해 추가된 부분
# news2int를 dictionary로 변환
news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))

# users = train_df['history_user'].unique()
# user2int = {uid: i for i, uid in enumerate(users)}
# all_user_ids = [i for i in range(len(users))]
# history_data_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
# df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
# df = df.dropna(subset=['clicked_news'])
# users = df['history_user'].unique()
# user2int = {uid: i for i, uid in enumerate(users)}
user2int_df = pd.read_csv('./psj/Adressa_1w/datas/user2int.tsv', sep='\t')
user2int = dict(zip(user2int_df['user_id'], user2int_df['user_int']))

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
train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)

# # validation과 test 나누는 부분
# criteria_time = pd.Timestamp('2017-02-15 20:00:01')
# train_ns['click_time'] = pd.to_datetime(train_ns['click_time'])
# validation_ns_df = train_ns[train_ns['click_time'] <= criteria_time]
# test_5d_ns_df = train_ns[train_ns['click_time'] > criteria_time]


train_ns.to_csv('./psj/Adressa_1w/test/test_ns.tsv', sep='\t', index=False)
# # validation/train_ns_df 저장
# validation_ns_df.to_csv('./psj/Adressa_4w/test/validation_ns.tsv', sep='\t', index=False)
# test_5d_ns_df.to_csv('./psj/Adressa_4w/test/test_ns.tsv', sep='\t', index=False)

exit()





# ns에 필요한 데이터 저장하는 변수들
train_ns_idx_batch = []
train_user_idx_batch = []
########################################### negative sampling 위한 부분 끝



news_num = len(train_df['clicked_news'])
total_user_num = len(users)
batch_size = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
batch_num = len(users) // batch_size if len(users) % batch_size == 0 else len(users) // batch_size + 1
# batch_seed_list = [[] for _ in range(len(batch_num))]
b_num = 0
for b in tqdm(range(batch_num), desc="processing batches"):
    # b_num += 1
    # if b_num > 5:
    #     break
    prev_batch = b * batch_size
    batch = min((b+1) * batch_size, total_user_num)
    batch_user_ids = all_user_ids[prev_batch:batch]   # 0 ~ 499, 500 ~ 999, ..., 84500 ~ 84989
    
    """
    추가한 부분 (for negative sampling)
    목표: user_score_idx 생성 및 저장
    """
    batch_ns_df = train_ns[train_ns['user_int'].isin(batch_user_ids)]
    # user_idx 처리
    user_tensor = torch.tensor(batch_ns_df['user_int'].tolist(), dtype=torch.long)
    train_user_idx_batch.append(user_tensor)
    # negative samples 포함한 news_idx 처리
    ns_idx_list = []
    for _, row in batch_ns_df.iterrows():
        # positive 뉴스 id (이미 news2int 매핑된 정수값)
        pos = int(row['news_int'])
        
        # negative_samples 처리: 공백으로 구분된 문자열을 리스트로 변환
        neg_str = row['negative_samples']
        # 각 요소를 int로 변환
        neg_list = [int(x) for x in neg_str.split()]
        # 요소가 4개보다 적으면 패딩, 4개보다 많으면 앞의 4개만 사용
        negs = neg_list
        ns_idx_list.append([pos] + negs)
    
    # 리스트를 텐서로 변환 (shape: [row_num, 5])
    ns_idx_tensor = torch.tensor(ns_idx_list, dtype=torch.long)
    train_ns_idx_batch.append(ns_idx_tensor)
    
    # continue
    """
    추가한 부분 (for negative sampling) 끝
    """
    


from time import time
print("start saving file")
start = time()

# ns에 필요한 idx 저장
torch.save(train_ns_idx_batch, './Adressa_4w/test/test_ns_idx_batch.pt')
torch.save(train_user_idx_batch, './Adressa_4w/test/test_user_idx_batch.pt')

end = time()
print(f"Saving file time: {(end - start) // 60}m {(end - start) % 60}s")
print("finished!")