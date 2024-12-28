# 1) Import
# 1.1) 필요한 라이브러리 import
import pandas as pd
import datetime
from tqdm import tqdm
import dgl
import torch
import os
import numpy as np

# 1.2) Config, NewsEncoder Import
from config import Config
from news_encoder import NewsEncoder

# 1.3) Config 불러오기
config = Config()



# 2. 데이터 전처리
# 2.1) 전처리 함수 정의 (1) - snapshot 구성
def get_period_start(click_time, interval_minutes=1440, start_time=datetime.time(8, 0, 2)):
    """
    2.1) 클릭 시간이 속하는 기간의 시작 시간을 계산
    """
    base_start = datetime.datetime.combine(click_time.date(), start_time)
    if click_time < base_start:
        base_start -= datetime.timedelta(days=1)
    delta = click_time - base_start
    periods = int(delta.total_seconds() // (interval_minutes * 60))
    return base_start + datetime.timedelta(minutes=interval_minutes * periods)
# 2.1) 전처리 함수 정의 (2) - 제목 단어 구분
def tokenize_title(title: str) -> list:
    """
    2.2) 타이틀을 공백 기준으로 단순 토크나이징
    """
    return title.split()


# 2.2) Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2.3) 사전(단어/카테고리) 로드
word2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'word2int.tsv'), sep='\t')
category2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'category2int.tsv'), sep='\t')

# 2.3.1) 'No category' & 'No subcategory'가 없으면 추가
if 'No category' not in category2int['category'].values:
    new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
    category2int = pd.concat([category2int, new_row], ignore_index=True)
if 'No subcategory' not in category2int['category'].values:
    new_row = pd.DataFrame([{'category': 'No subcategory', 'int': 0}])
    category2int = pd.concat([category2int, new_row], ignore_index=True)

# 2.4) word_to_idx, pretrained embedding
# words = word2int['word'].tolist()
word_to_idx = word2int.set_index('word')['int'].to_dict()
embedding_file_path = os.path.join('./Adressa_4w/history/', 'pretrained_word_embedding.npy')
embeddings = np.load(embedding_file_path)
pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)

# 2.5) NewsEncoder 초기화
news_encoder = NewsEncoder(config, pretrained_word_embedding).to(device)
# news_encoder.eval()  # 필요 시, 추론 모드

# 2.6) behavior 데이터 로드 및 필터링
file_path = './Adressa_4w/history/history_tkg_behaviors.tsv'
df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
df['click_time'] = pd.to_datetime(df['click_time'])

end_time = pd.Timestamp('2017-02-05 08:00:01')
df = df[df['click_time'] <= end_time]   # 정확히 5주 데이터만 사용하도록 필터링

# 2.6.1) period_start 컬럼 생성
df['Period_Start'] = df['click_time'].apply(
    lambda x: get_period_start(x, interval_minutes=30)
)

# 2.6.2) category 결측치 처리 + 분리
df['category'] = df['category'].fillna('No category|No subcategory')
df[['Category', 'Subcategory']] = df['category'].str.split('|', n=1, expand=True)

# 2.7) groupby(Period_Start)
grouped = df.groupby('Period_Start')

""" 3. News 정보, 임베딩 생성 """
# 3.1) 뉴스별 데이터(카테고리, 서브카테고리, 타이틀) 집계
news_info = df.groupby('clicked_news', as_index=False).agg({
    'Category': 'first',
    'Subcategory': 'first',
    'title': 'first'
})

# 3.2) 카테고리와 서브카테고리를 정수로 매핑
news_info['category_id'] = news_info['Category'].map(
    category2int.set_index('category')['int']
    )
news_info['subcategory_id'] = news_info['Subcategory'].map(
    category2int.set_index('category')['int']
    )

# 3.3) title -> token -> index
news_info['title_words'] = news_info['title'].apply(tokenize_title)
news_info['title_idx'] = news_info['title_words'].apply(
    lambda words: [word_to_idx[w] if w in word_to_idx else 0 for w in words]
)

# news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
news_info_df = news_info[['clicked_news', 'category_id', 'subcategory_id', 'title_idx']].rename(
    columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
)

# news_id_to_info
news_id_to_info = news_info_df.set_index('news_id')\
    [['category_id', 'subcategory_id', 'title_idx']].to_dict(orient='index')
    # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장

# 3.4) 전체 유저, 뉴스, 카테고리 목록
all_user_ids = df['history_user'].unique()
all_news_ids = df['clicked_news'].unique()
all_cat_strs = df['category'].unique()

all_user_ids.sort()  # 정렬
# all_news_ids.sort(...)  # 필요하다면 정렬

# 3.5) 전역 user2int, news2int, cat2idx
user2int = {uid: i for i, uid in enumerate(all_user_ids)}
news2int = {nid: i for i, nid in enumerate(all_news_ids)}
cat2idx = {cat: i for i, cat in enumerate(all_cat_strs)}

# 3.6) 임베딩 준비 (유저 / 카테고리)
user_embeddings = torch.randn(len(all_user_ids), 128, device=device)
category_embeddings = torch.randn(len(all_cat_strs), 128, device=device)

# 4) Main 함수
def main():
    """ 
    전체 모델 파이프라인을 실행 
    1. News Encoder로 뉴스 embedding 생성
    2. Snapshots 생성
    2.1) snapshots 엣지 수 검증
    2.2) snapshots 정보 저장
    -------------------------------------
    3. GCN
    4. GRNN
    5. Loss function
    """
    print("main function start!")
    # 4.1) 뉴스 임베딩 (NewsEncoder)
    news_vectors = []
    for nid in tqdm(all_news_ids, desc="Making news embeddings"):
        if nid in news_id_to_info:
            info = news_id_to_info[nid]
            title_idx_list = info['title_idx']
            title_tensor = torch.tensor(title_idx_list, dtype=torch.long, device=device)
            nv = news_encoder(title_tensor)  # shape: (1, num_filters) 가 되도록 내부 처리
            news_vectors.append(nv.squeeze(0))
        else:
            news_vectors.append(torch.randn(config.num_filters, device=device))

    news_embeddings = torch.stack(news_vectors)  # (전체 뉴스 수, num_filters)


    # 4.2) 스냅샷 생성 (DGL 그래프)
    snapshots = []
    for period_start, group in tqdm(grouped, desc="Making snapshots"):
        '''
        period_start: snapshot 시작 시점
        group: 각 snapshot의 데이터를 담은 dataframe
        '''
        # 4.2.1) user, news, category 전역 인덱스 맵핑
        edges_src = group['history_user'].map(user2int).values
        edges_dst = group['clicked_news'].map(news2int).values
        edge_cats = group['category'].map(cat2idx).values  # 각 클릭의 카테고리 인덱스

        # 4.2.2) 그래프 생성: 전체 유저/뉴스 노드 수 지정
        g = dgl.heterograph(
            {('user', 'clicked', 'news'): (edges_src, edges_dst)},
            num_nodes_dict={'user': len(all_user_ids), 'news': len(all_news_ids)}
        ).to(device)

        # 4.2.3) 노드 (유저, 뉴스) 임베딩 할당
        g.nodes['user'].data['feat'] = user_embeddings
        g.nodes['news'].data['feat'] = news_embeddings

        # 4.2.4) 엣지 (카테고리) 임베딩 할당
        edge_cat_embeddings = category_embeddings[edge_cats]  # (# of edges, 128)
        g.edges['clicked'].data['feat'] = edge_cat_embeddings

        # 4.2.5) 검증
        if g.number_of_edges(('user', 'clicked', 'news')) != len(group):
            print(f"[Warning] period {period_start} - mismatch edges vs group size")
            exit()

        snapshots.append(g)

    """ 5. 스냅샷 정보 저장 """
    snapshot_info_list = []
    for period_start, group_df in tqdm(grouped, desc="Processing Groups' infos"):
        # 유저별 클릭 수 등 간단 집계
        group_user_info = group_df.groupby('history_user', as_index=False).agg({
            'clicked_news': lambda x: list(x)
        })
        group_user_info['# of clicks'] = group_user_info['clicked_news'].apply(len)

        snapshot_info_list.append({
            'Period_Start': period_start,
            'User_Nodes_Num': group_df['history_user'].nunique(),
            'News_Nodes_Num': group_df['clicked_news'].nunique(),
            'Edges_Num': len(group_df),
            'Avg.#_Clicks_per_User': group_user_info['# of clicks'].mean(),
            'Category_Edges_Num': group_df['Category'].nunique()
        })

    snapshot_info_df = pd.DataFrame(snapshot_info_list)
    snapshot_info_df['Period_Start'] = snapshot_info_df['Period_Start']\
        .dt.strftime('%Y-%m-%d %H:%M:%S')

    snapshot_info_path = './Adressa_4w/history/30m_lf_snapshots_info.tsv'
    snapshot_info_df.to_csv(snapshot_info_path, sep='\t', index=False, encoding='utf-8')

    """ 6. 매핑 파일 저장 """
    user_map_df = pd.DataFrame(list(user2int.items()), columns=['user_id', 'user_int'])
    user_map_df.to_csv('./Adressa_4w/history/user2int.tsv', sep='\t', index=False, encoding='utf-8')

    news_map_df = pd.DataFrame(list(news2int.items()), columns=['news_id', 'news_int'])
    news_map_df.to_csv('./Adressa_4w/history/news2int.tsv', sep='\t', index=False, encoding='utf-8')

    print("[INFO] Snapshots creation done!")
    print(f"Total snapshots: {len(snapshots)}")


if __name__ == "__main__":
    # 7. main() 실행
    main()
