import pickle
import pandas as pd
import torch
from tqdm import tqdm
import datetime

"""
full_news_encoder.py를 사용하기 위해 category2int_pio.tsv로 변경
"""

def make_train_datas():
    # 각 뉴스의 카테고리만 가져오기
    all_news_df = pd.read_csv('./psj/Adressa_4w/history/all_news_nyheter_splitted.tsv', sep='\t')
    sub_all_news_df = all_news_df[['newsId', 'category']]



    # snapshots에 카테고리 정보 추가하기
    # 1. history, train, test에 대해 전역 category2int를 만든다 (이미 있음)
    # 2. category2int를 g.edges['clicked'].data['category']에 저장한다 (이미 있음)
    # 3. main.py에서 데이터 로드할 때 category idx를 사용한다
    # *** 전역 news2int도 필요!!!


    # news2int 가져오기
    news2int_file_path = './psj/Adressa_4w/history/news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    # # train_ns data 가져오기
    # train_ns_path = "./psj/Adressa_4w/train/tkg_train_negative_samples_lt36_ns4.tsv"
    # train_ns = pd.read_csv(train_ns_path, sep='\t')

    # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
    train_file_path = './psj/Adressa_4w/train/valid_tkg_behaviors.tsv'
    train_df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
    # 'clicked_news' 열에서 '-1' 제거
    train_df['clicked_news'] = train_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
    # 'clicked_newsId'를 기준으로 'category' 매칭
    train_df = train_df.merge(sub_all_news_df, left_on='clicked_news', right_on='newsId', how='left')

    # print(train_df.head())
    # print(train_df.columns)
    # print(train_df.isna().sum())
    # print(train_df[train_df.isna()]) # 하나가 데이터 오류
    # print(len(train_df))
    # exit()

    # train_df에서 nan이 존재하는 행 제거
    train_df = train_df.dropna(subset=['clicked_news'])



    ########################################### 여기부터 negative sampling을 위해 추가된 부분
    # news2int를 dictionary로 변환
    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    # user2int mapping
    file_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
    history_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    history_df['click_time'] = pd.to_datetime(history_df['click_time'])
    end_time = pd.Timestamp('2017-02-05 08:00:01')
    history_df = history_df[history_df['click_time'] <= end_time]   # 정확히 5주 데이터만 사용하도록 필터링
    users = history_df['history_user'].unique()
    user2int = {uid: i for i, uid in enumerate(users)}
    all_user_ids = sorted([user2int[u] for u in users])

    # train_df에 ns와 int 추가
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
    train_df['user_int'] = train_df['history_user'].map(user2int)
    train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)
    category2int = pd.read_csv('./psj/Adressa_4w/history/category2int_pio.tsv', sep='\t')
    # 필요시 category2int에 'No category' 추가
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)
    cat2int = category2int.set_index('category')['int'].to_dict()
    train_df['cat_int'] = train_df['category'].map(cat2int)#.fillna(0)
    # has_nan = train_df['cat_int'].isna().any()
    # print("nan exists:", has_nan)

    # # nan 개수 세기
    # nan_count = train_df['cat_int'].isna().sum()
    # print("Number of nan values:", nan_count)

    # # print(train_df['category'].values)
    # exit()

    # period_start -> time_idx 매핑(0부터 시작)
    def get_period_start(click_time, interval_minutes=1440, start_time=datetime.time(8, 0, 2)):

        base_start = datetime.datetime.combine(click_time.date(), start_time)
        if click_time < base_start:
            base_start -= datetime.timedelta(days=1)
        delta = click_time - base_start
        periods = int(delta.total_seconds() // (interval_minutes * 60))

        return base_start + datetime.timedelta(minutes=interval_minutes * periods)

    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['Period_Start'] = train_df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=30))
        
    unique_period_starts = train_df['Period_Start'].unique()
    time_dict = {ps: i+1680 for i, ps in enumerate(sorted(unique_period_starts))}
    train_df['time_idx'] = train_df['Period_Start'].map(time_dict)

    """
    train_news: 각 요소(리스트)는 train data에서 각 유저가 클릭한 news_ids
    - shape: (user_num, train data에서 각 유저의 클릭 수)

    train_category: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 categories
    - shape: (user_num, train data에서 각 유저의 클릭 수)

    train_time: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 times
    - shape: (user_num, train data에서 각 유저의 클릭 수)
    """
    train_news = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(train_df[train_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        train_news.append(u_news)
        
    train_category = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_category = torch.tensor(train_df[train_df['user_int'] == u_id]['cat_int'].values, dtype=torch.long)
        train_category.append(u_category)

    train_time = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_time = torch.tensor(train_df[train_df['user_int'] == u_id]['time_idx'].values, dtype=torch.long)
        train_time.append(u_time)
        
    # print(train_time[0])
    # print(len(train_time[0]))


    # with open('./psj/Adressa_4w/train/train_datas.pkl', 'wb') as f:
    #     pickle.dump(list(zip(train_news, train_category, train_time)), f)
    
    return list(zip(train_news, train_category, train_time))