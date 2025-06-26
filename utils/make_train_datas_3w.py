import pickle
import pandas as pd
import torch
from tqdm import tqdm
import datetime
import os

"""
train_news, train_category는 어차피 안 쓰임
model의 forward input이지만, 안 쓰임
-> category2int 바꾸든 안 바꾸든 현재는 의미 없음
"""

def make_train_datas(interval_minutes, week = 3):

    # snapshots에 카테고리 정보 추가하기
    # 1. history, train, test에 대해 전역 category2int를 만든다 (이미 있음)
    # 2. category2int를 g.edges['clicked'].data['category']에 저장한다 (이미 있음)
    # 3. main.py에서 데이터 로드할 때 category idx를 사용한다
    # *** 전역 news2int도 필요!!!


    # news2int 가져오기
    news2int_file_path = f'psj/Adressa_{week}w/datas/news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    # a) train dataset(0205 08:00:02 ~ 0212 08:00:01)인 valid_tkg_behaviors.tsv 로드
    train_file_path = f'psj/Adressa_{week}w/datas/{week}w_behaviors.tsv'
    df = pd.read_csv(train_file_path, sep='\t', encoding='utf-8')
    # click_time을 string에서 datetime으로 변환
    df['click_time'] = pd.to_datetime(df['click_time'])
    
    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-23 00:00:00')
    train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    # train_df에서 nan이 존재하는 행 제거
    train_df = train_df.dropna(subset=['clicked_news'])



    ########################################### 여기부터 negative sampling을 위해 추가된 부분
    # news2int를 dictionary로 변환
    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
    # user2int mapping
    user2int_df = pd.read_csv(os.path.join(f'psj/Adressa_{week}w/datas/', 'user2int.tsv'), sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    all_user_ids = user2int_df['user_int'].tolist()   # 0 ~ 84988

    # train_ns['negative_samples'] = train_ns['negative_samples'].apply(map_negative_samples)
    train_df['user_int'] = train_df['history_user'].map(user2int)
    train_df['news_int'] = train_df['clicked_news'].map(news2int_mapping)
    category2int = pd.read_csv(f'psj/Adressa_{week}w/datas/category2int_nyheter_splitted.tsv', sep='\t')
    # 필요시 category2int에 'No category' 추가
    if 'No category' not in category2int['category'].values:
        new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
        category2int = pd.concat([new_row, category2int], ignore_index=True)
    cat2int = category2int.set_index('category')['int'].to_dict()
    
    ############# category가 nyheter이면 subcategory로 매핑, 그렇지 않으면 category로 매핑
    def get_cat_int(row):
        if row['category'] == 'nyheter':
            # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
            return cat2int.get(row['subcategory'], cat2int['No category'])
        else:
            return cat2int.get(row['category'], cat2int['No category'])

    train_df['cat_int'] = train_df.apply(get_cat_int, axis=1)
    
    ### interval_minutes가 <= 24h인 경우에는 ok
    # # period_start -> time_idx 매핑(0부터 시작)
    # def get_period_start(click_time, interval_minutes, start_time=datetime.time(0, 0, 0)):

    #     base_start = datetime.datetime.combine(click_time.date(), start_time)
    #     if click_time < base_start:
    #         base_start -= datetime.timedelta(days=1)
    #     delta = click_time - base_start
    #     periods = int(delta.total_seconds() // (interval_minutes * 60))

    #     return base_start + datetime.timedelta(minutes=interval_minutes * periods)
    
    GLOBAL_START = pd.Timestamp('2017-01-05 00:00:00')  # criteria_time1와 동일
    def get_period_start_global(click_time, interval_minutes):
        """36시간(2160분) 단위 전역 버킷 시작 시각 반환"""
        delta = click_time - GLOBAL_START
        periods = int(delta.total_seconds() // (interval_minutes * 60))
        return GLOBAL_START + pd.Timedelta(minutes=interval_minutes * periods)


    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['Period_Start'] = train_df['click_time'].apply(lambda x: get_period_start_global(x, interval_minutes=interval_minutes))
    
    ### 매우 중요!!!
    history_weeks = 15/7#15 / 7
    interval_hours = interval_minutes / 60
    his_snapshots_num = int(history_weeks * 7 * 24 / interval_hours)
    
    unique_period_starts = train_df['Period_Start'].unique()
    time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
    train_df['time_idx'] = train_df['Period_Start'].map(time_dict)
    train_df = train_df[train_df['time_idx'] >= his_snapshots_num]
    print(train_df['time_idx'].max())

    """
    train_news: 각 요소(리스트)는 train data에서 각 유저가 클릭한 news_ids
    - shape: (user_num, train data에서 각 유저의 클릭 수)

    train_category: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 categories
    - shape: (user_num, train data에서 각 유저의 클릭 수)

    train_time: 각 요소(리스트)는 train data에서 각 유저가 클릭한 뉴스의 times (snapshot or timestamp indicies)
    - shape: (user_num, train data에서 각 유저의 클릭 수)
    """
    train_news = []
    train_category = []
    train_time = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(train_df[train_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        u_category = torch.tensor(train_df[train_df['user_int'] == u_id]['cat_int'].values, dtype=torch.long)
        u_time = torch.tensor(train_df[train_df['user_int'] == u_id]['time_idx'].values, dtype=torch.long)
        train_news.append(u_news)
        train_category.append(u_category)
        train_time.append(u_time)
        
    # for u_id in tqdm(range(len(all_user_ids))):

    # for u_id in tqdm(range(len(all_user_ids))):
        
    # print(train_time[0])
    # print(len(train_time[0]))


    # with open('./psj/Adressa_4w/train/train_datas.pkl', 'wb') as f:
    #     pickle.dump(list(zip(train_news, train_category, train_time)), f)
    
    return list(zip(train_news, train_category, train_time))