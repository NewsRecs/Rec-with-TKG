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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# behavior 데이터 로드
file_path = './psj/Adressa_1w/train/history_tkg_behaviors.tsv'
df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

criteria_time1 = pd.Timestamp('2017-01-10 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-11 00:00:00')
prev_train_criteria_time = pd.Timestamp('2017-01-08 12:00:00')
train_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
train_prev_df = df[(prev_train_criteria_time <= df['click_time']) & (df['click_time'] < criteria_time2)]

criteria_time1 = pd.Timestamp('2017-01-11 00:00:00')
criteria_time2 = pd.Timestamp('2017-01-12 00:00:00')
prev_test_criteria_time = pd.Timestamp('2017-01-09 12:00:00')
test_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
test_prev_df = df[(prev_test_criteria_time <= df['click_time']) & (df['click_time'] < criteria_time2)]



# df들에서 nan이 존재하는 행 제거
train_df = train_df.dropna(subset=['clicked_news'])
train_prev_df = train_prev_df.dropna(subset=['clicked_news'])
test_df = test_df.dropna(subset=['clicked_news'])
test_prev_df = test_prev_df.dropna(subset=['clicked_news'])



from tqdm import tqdm
def generate_negative_samples_optimized(df, df_prev, negative_sample_size=4):
    """
    - df        : 메인(실제 negative sample 생성할) 데이터프레임
      (columns: ['user', 'click_time', 'clicked_news', ...])
    - df_prev   : 36시간 이전 포함된 기록 데이터프레임
      (columns: ['user', 'click_time', 'clicked_news', ...])
    - negative_sample_size : 각 클릭마다 추출할 negative 샘플 수
    """

    # (1) df, df_prev 모두 click_time 기준으로 정렬
    # df_sorted = df.sort_values('click_time').reset_index(drop=True)
    # df_prev_sorted = df_prev.sort_values('click_time').reset_index(drop=True)

    # df_prev['click_time'] = pd.to_datetime(df_prev['click_time']) # , errors='coerce'
    # 변환 실패한 행이 있으면 NaT가 생김
    # NaT가 있으면 삭제 또는 다른 방식으로 처리

    # mask_invalid = df_prev['click_time'].isna()
    # print("Invalid click_time rows:\n", df_prev[mask_invalid])
    
    # print(df.head(10))
    # print(df_sorted.head(10))
    # print(df.equals(df_sorted))
    # print(df_prev.equals(df_prev_sorted))
    # exit()

    
    # (2) df_prev에서 click_time, clicked_news, user를 numpy array로 꺼내두기
    prev_times = df_prev['click_time'].values#.astype('datetime64[ns]')
    prev_clicks = df_prev['clicked_news'].values
    prev_users = df_prev['history_user'].values
    # prev_times = df_prev['click_time'].values
    # prev_clicks = df_prev['clicked_news'].values
    # prev_users = df_prev['history_user'].values

    results = []

    # (3) df_sorted를 순회하며 negative sample 생성
    for _, row in tqdm(df.iterrows(), desc='Processing negative sampling', total=len(df)):
        user = row['history_user']
        click_time = row['click_time'].to_datetime64()
        clicked_news = row['clicked_news']

        # 36시간 전 시점
        start_time = click_time - pd.Timedelta(hours=36)
        start_time = start_time.to_datetime64()
        # print("type:", type(prev_times[0]))
        
        # (a) searchsorted로 36시간 범위를 빠르게 찾음
        start_idx = np.searchsorted(prev_times, start_time, side='left')
        end_idx = np.searchsorted(prev_times, click_time, side='left')

        # (b) slice 추출 (큰 df에서 작은 구간만 떼옴)
        slice_users = prev_users[start_idx:end_idx]
        slice_clicks = prev_clicks[start_idx:end_idx]

        # (c) 36시간 구간 내 "전체 뉴스" 집합
        #     (클릭은 전체 유저것이므로 set() 씌움)
        all_news_in_window = set(slice_clicks)

        # (d) 이 유저가 실제로 클릭한 뉴스 (36시간 구간 내)
        #     np.where로 user가 같은 인덱스 추출 -> 그 인덱스로 slice_clicks 가져오기
        user_indices = np.where(slice_users == user)[0]
        user_clicked_news = set(slice_clicks[user_indices])

        # (e) negative candidate = 전체 뉴스 - 유저가 본 뉴스
        negative_candidates = list(all_news_in_window - user_clicked_news)

        # (f) negative 샘플 랜덤 추출
        sampled_negatives = random.sample(negative_candidates, negative_sample_size)
        
        # (g) 결과 저장
        negative_samples_str = " ".join(sampled_negatives)
        results.append((user, click_time, clicked_news, negative_samples_str))

    # (4) 결과 DF 생성
    neg_df = pd.DataFrame(results, columns=['user', 'click_time', 'clicked_news', 'negative_samples'])
    # # (5) click_time 순으로 정렬 (이미 순서대로라면 불필요할 수 있음)
    # neg_df = neg_df.sort_values('click_time').reset_index(drop=True)

    return neg_df


# ---------------------------
# (A) train 데이터 (4개)
# ---------------------------
train_neg_df = generate_negative_samples_optimized(
    df=train_df,
    df_prev=train_prev_df,
    negative_sample_size=4
)

train_neg_df.to_csv(
    "./psj/Adressa_1w/train/tkg_train_negative_samples_lt36_ns4.tsv",
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
    "./psj/Adressa_1w/test/tkg_test_negative_samples_lt36_ns20.tsv",
    sep='\t',
    index=False,
    encoding='utf-8'
)
