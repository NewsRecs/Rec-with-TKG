### train_user_tensor, train_news_tensor, g, splitted_g 구성
import torch
import pickle  
import os
from tqdm import tqdm
import pandas as pd
import pickle
import dgl

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
train_news_file_path = './psj/Adressa_1w/datas/all_news_nyheter_splitted.tsv'
train_news_df = pd.read_csv(train_news_file_path, sep='\t')
train_news_df.columns = ['index_col', 'newsId','category','subcategory', 'title']
sub_train_news_df = train_news_df[['newsId', 'category']]

# train_ns data 가져오기
train_ns_path = "./psj/Adressa_1w/train/tkg_train_negative_samples_ns4.tsv"
train_ns = pd.read_csv(train_ns_path, sep='\t')

# a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
train_file_path = './psj/Adressa_1w/datas/1w_behaviors.tsv'
df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
# 'clicked_newsId'를 기준으로 'category' 매칭
train_df = train_df.merge(sub_train_news_df, left_on='clicked_news', right_on='newsId', how='left')

train_users = train_df['history_user']

# print(train_df.head())
# print(train_df.columns)
# print(train_df.isna().sum())
# print(train_df[train_df.isna()]) # 하나가 데이터 오류
# print(len(train_df))
# exit()

# train_df에서 nan이 존재하는 행 제거
train_df = train_df.dropna(subset=['clicked_news'])
train_df = train_df[train_df.notna()]




########################################### 여기부터 negative sampling을 위해 추가된 부분

# news2int 가져오기
news2int_file_path = './psj/Adressa_1w/datas/news2int.tsv'
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
user2int_df = pd.read_csv('./psj/Adressa_1w/datas/user2int.tsv', sep='\t')
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
train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)


# train_ns 저장
train_ns.to_csv('./psj/Adressa_1w/train/train_ns.tsv', sep='\t', index=False)