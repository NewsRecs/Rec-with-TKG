import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import os

# -----------------------------
# 0. 파일 위치 설정
# -----------------------------
BEH_PATH   = 'psj/Adressa_1w/datas/1w_behaviors.tsv'
PUB_PATH   = 'psj/Adressa_1w/datas/news_publish_times.tsv'
SAVE_DIR1   = 'psj/Adressa_1w/train'          # 결과 저장 폴더
SAVE_DIR2   = 'psj/Adressa_1w/test'          
os.makedirs(SAVE_DIR1, exist_ok=True)
os.makedirs(SAVE_DIR2, exist_ok=True)

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
df = pd.read_csv(BEH_PATH, sep='\t', encoding='utf-8')
df['click_time']   = pd.to_datetime(df['click_time'])
df['clicked_news'] = df['clicked_news'].str.replace(r'-\d+$', '', regex=True)

publish_df = pd.read_csv(PUB_PATH, sep='\t', encoding='utf-8')
publish_df['publish_time'] = pd.to_datetime(publish_df['publish_time'])
# → 빠른 조회를 위해 사전(dict)으로 변환
news2time = dict(zip(publish_df['news_id'], publish_df['publish_time']))

# -----------------------------
# 2. 유저별 클릭 뉴스 집합 생성
# -----------------------------
user2clicked = (
    df.groupby('history_user')['clicked_news']
      .apply(set)
      .to_dict()
)

# -----------------------------
# 3. 학습·테스트 구간 분할
# -----------------------------
train_mask = (df['click_time'] >= '2017-01-10') & (df['click_time'] < '2017-01-11')
test_mask  = (df['click_time'] >= '2017-01-11') & (df['click_time'] < '2017-01-12')

train_df = df.loc[train_mask].copy()
test_df  = df.loc[test_mask].copy()

# -----------------------------
# 4. 36 시간 이내 뉴스 후보 사전 구축
#    (publish_df를 시간순 정렬하여 리스트로 두고,
#     각 클릭마다 브루트-포스로 필터링하면 충분히 빠름)
# -----------------------------
PUBLISH_TIMES = publish_df['publish_time'].values
NEWS_IDS      = publish_df['news_id'].values

def get_candidates(click_time):
    """click_ts 이전 36 시간 내에 발행된 뉴스 ID 리스트 반환"""
    lower = click_time - timedelta(hours=36)
    mask  = (PUBLISH_TIMES >= lower) & (PUBLISH_TIMES < click_time) & (PUBLISH_TIMES < click_time) & (PUBLISH_TIMES < click_time)
    return NEWS_IDS[mask]

# -----------------------------
# 5. 샘플링 함수
# -----------------------------
rng = np.random.default_rng(28)        # 재현성을 위한 seed

def sample_negatives(row, k):
    """각 클릭(row)에 대해 k개의 무작위 음성 샘플 추출"""
    cand = get_candidates(row['click_time'])
    
    # 유저가 한 번이라도 클릭한 뉴스 제외
    user_clicked = user2clicked.get(row['history_user'], set())
    cand = cand[~np.isin(cand, list(user_clicked))]
    choice_idx = rng.choice(len(cand), size=k, replace=False)
    neg_ids    = cand[choice_idx]
    neg_times  = [news2time[n].strftime('%Y-%m-%d %H:%M:%S') for n in neg_ids]
    return neg_ids, neg_times

def attach_negative_samples(df_clicks, k):
    neg_cols, time_cols = [], []
    for _, row in tqdm(df_clicks.iterrows(), total=len(df_clicks) ):
        ids, times = sample_negatives(row, k)
        neg_cols.append(' '.join(ids))          # 공백으로 연결
        time_cols.append(','.join(times))       # 쉼표로 연결
    df_clicks['negative_samples'] = neg_cols
    df_clicks['publish_times']    = time_cols
    return df_clicks

# -----------------------------
# 5. train(4개) / test(20개) 처리
# -----------------------------
train_df = attach_negative_samples(train_df, 4)
test_df  = attach_negative_samples(test_df, 20)

# 필요한 열만 선택하고 history_user → user 로 명칭 통일
cols = ['history_user', 'click_time', 'clicked_news',
        'negative_samples', 'publish_times']
train_df = train_df[cols].rename(columns={'history_user': 'user'})
test_df  = test_df[cols].rename(columns={'history_user': 'user'})

# -----------------------------
# 6. 저장
# -----------------------------
train_path = os.path.join(SAVE_DIR1,
                          'tkg_train_negative_samples_lt36_ns4.tsv')
test_path  = os.path.join(SAVE_DIR2,
                          'tkg_test_negative_samples_lt36_ns20.tsv')

train_df.to_csv(train_path, sep='\t', index=False)
test_df.to_csv(test_path,  sep='\t', index=False)

print(f"✅ 저장 완료:\n  • {train_path}\n  • {test_path}")
