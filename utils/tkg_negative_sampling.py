"""
코드 계획
0. make_history_tkg_behaviors.ipynb에서 valid_users에 해당하는 뉴스들을 train, test에서 뽑아내기
*** train data와 36시간 이전이 포함된 data를 다르게 변수로 지정해야 함
- train data: './Adressa_4w/train/valid_tkg_behaviors.tsv'
- train prev involved data: './Adressa_4w/train/prev_involved_train_tkg_behaviors.tsv'
- test data: './Adressa_4w/test/valid_tkg_behaviors.tsv'
- test prev involved data: './Adressa_4w/test/prev_involved_test_behaviors.tsv'
1. 08:00:01을 기준으로 train, test의 데이터 불러 오기
*** train, test마다 다르게 불러와서 처리해야 함
***** train data의 클릭마다 train prev involved data의 기록을 가져오도록 처리
2. train data에 대해 이전 36시간까지 포함한 데이터 (변수) 구성하기
3. 구성한 데이터를 바탕으로 train의 각 클릭에 대해 아래 동작을 수행
3.1) 클릭 시간을 기준으로 이전 36시간 이내의 데이터 중에 이 유저의 클릭 히스토리에 해당하지 않는 뉴스를 random sampling
3.2) 각 클릭마다 df 형태로 random sampling된 데이터를 저장 - columns = ['user', 'click_time', 'clicked_news', 'negative_samples']
4. 3에서 만든 df를 tkg_negative_samples_lt36_ns8.tsv로 저장
"""

import torch
import pandas as pd
import random
import numpy as np



# behavior 데이터 로드
file_path = './psj/Adressa_4w/train/valid_tkg_behaviors.tsv'
train_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
file_path = './psj/Adressa_4w/train/prev_involved_train_tkg_behaviors.tsv'
train_prev_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
file_path = './psj/Adressa_4w/test/valid_tkg_behaviors.tsv'
test_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
file_path = './psj/Adressa_4w/test/prev_involved_test_behaviors.tsv'
test_prev_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

train_df['click_time'] = pd.to_datetime(train_df['click_time'])
train_df['clicked_news'] = train_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
train_prev_df['click_time'] = pd.to_datetime(train_prev_df['click_time'])
train_prev_df['clicked_news'] = train_prev_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_df['clicked_news'] = test_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)
test_prev_df['click_time'] = pd.to_datetime(test_prev_df['click_time'])
test_prev_df['clicked_news'] = test_prev_df['clicked_news'].str.replace(r'-\d+$', '', regex=True)


# df들에서 nan이 존재하는 행 제거
train_df = train_df.dropna(subset=['clicked_news'])
train_prev_df = train_prev_df.dropna(subset=['clicked_news'])
test_df = test_df.dropna(subset=['clicked_news'])
test_prev_df = test_prev_df.dropna(subset=['clicked_news'])



# history behaviors 모음
file_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
history_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# 모든 behaviors.tsv에서 유저가 클릭한 뉴스 전부 모으기
def build_user_history_dict(dfs):
    user_hist = {}
    for df in dfs:
        for user, news in zip(df['history_user'], df['clicked_news']):
            user_hist.setdefault(user, set()).add(news)
    return user_hist

user_hist_dict = build_user_history_dict([
    history_df, train_df, test_df
])



from tqdm import tqdm
def generate_negative_samples_optimized(df, df_prev, negative_sample_size=4):
    prev_times  = df_prev['click_time'   ].values
    prev_clicks = df_prev['clicked_news' ].values
    prev_users  = df_prev['history_user'].values

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        user        = row['history_user']
        click_time  = row['click_time']
        clicked_news= row['clicked_news']

        # 36시간 창
        start_time  = click_time - pd.Timedelta(hours=36)
        # numpy.searchsorted을 위해 numpy.datetime64로 변환
        start_np = np.datetime64(start_time)
        click_np = np.datetime64(click_time)
        
        s_idx = np.searchsorted(prev_times, start_np, side='left')
        e_idx = np.searchsorted(prev_times, click_np , side='left')

        slice_users  = prev_users [s_idx:e_idx]
        slice_clicks = prev_clicks[s_idx:e_idx]

        # (1) 창 안 전체 뉴스
        all_news_in_window = set(slice_clicks)

        # (2) 창 안에서 이 유저가 본 뉴스
        user_indices   = np.where(slice_users == user)[0]
        user_clicked_news = set(slice_clicks[user_indices])

        # (3) 전(全) 히스토리에서 이 유저가 본 뉴스
        history_clicked = user_hist_dict.get(user, set())

        # (4) negative 후보 = 창 전체 − (창‑내 클릭 ∪ 전‑히스토리 클릭)
        negative_candidates = list(
            all_news_in_window - user_clicked_news - history_clicked
        )

        # (5) 후보 부족 시 중복 허용
        sampled_negatives = (random.sample(negative_candidates, negative_sample_size)
                             if len(negative_candidates) >= negative_sample_size
                             else random.choices(negative_candidates, k=negative_sample_size))

        results.append((user, click_time, clicked_news, " ".join(sampled_negatives)))

    return pd.DataFrame(results,
                        columns=['user','click_time','clicked_news','negative_samples'])



# ---------------------------
# (A) train 데이터 (8개)
# ---------------------------
train_neg_df = generate_negative_samples_optimized(
    df=train_df,
    df_prev=train_prev_df,
    negative_sample_size=4
)

train_neg_df.to_csv(
    "./psj/Adressa_4w/train/tkg_train_negative_samples_lt36_ns4_revised.tsv",
    sep='\t',
    index=False,
    encoding='utf-8'
)

# ---------------------------
# (B) test 데이터 (20개)
# ---------------------------
test_neg_df = generate_negative_samples_optimized(
    df=test_df,
    df_prev=test_prev_df,
    negative_sample_size=20
)

test_neg_df.to_csv(
    "./psj/Adressa_4w/test/tkg_test_negative_samples_lt36_ns20_revised.tsv",
    sep='\t',
    index=False,
    encoding='utf-8'
)



def check_duplicates(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    dup_rows = df[df['negative_samples'].str.split().apply(lambda x: len(x)!=len(set(x)))]
    print(f"{len(dup_rows)} / {len(df)} rows have duplicate negatives")

check_duplicates("./psj/Adressa_4w/train/tkg_train_negative_samples_lt36_ns4_revised.tsv")
check_duplicates("./psj/Adressa_4w/test/tkg_test_negative_samples_lt36_ns20_revised.tsv")