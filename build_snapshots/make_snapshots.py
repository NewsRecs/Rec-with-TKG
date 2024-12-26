import pandas as pd
import datetime
from tqdm import tqdm
import dgl
import torch
import os
import numpy as np
from news_encoder import NewsEncoder


# NewsEncoder 설정
class Config:
    num_words = 1 + 330899   # 실제 단어 수(330899)에 패딩 토큰(index=0)을 더함; index = 0: 존재하지 않는 단어들
    word_embedding_dim = 100   # 사전 학습된 단어 embedding 차원
    num_categories = 1 + 127   # 실제 카테고리 수(127)에 패딩 토큰(index=0)을 더함; index = 0: No category, No subcategory 케이스
    num_filters = 128   # snapshots에서 news, user, category embedding 차원
    query_vector_dim = 200   # NewsEncoder query vector 차원
    window_size = 3
    dropout_probability = 0.2

config = Config()

# 사전 학습된 임베딩 로드
# words = [...]  # 단어 리스트
# embeddings = [...]  # 단어 임베딩 리스트
word2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'word2int.tsv'), sep='\t')
category2int = pd.read_csv(os.path.join('./Adressa_4w/history/', 'category2int.tsv'), sep='\t')
# 'No category'와 'No subcategory'가 category2int에 없을 경우 추가
if 'No category' not in category2int['category'].values:
    new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
    category2int = pd.concat([category2int, new_row], ignore_index=True)
if 'No subcategory' not in category2int['category'].values:
    new_row = pd.DataFrame([{'category': 'No subcategory', 'int': 0}])
    category2int = pd.concat([category2int, new_row], ignore_index=True)
    
words = word2int['word'].tolist()
# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 단어와 인덱스 매핑
word_to_idx = word2int.set_index('word')['int'].to_dict()
# 단어 embedding loading
embedding_file_path = os.path.join('./Adressa_4w/history/', 'pretrained_word_embedding.npy')
embeddings = np.load(embedding_file_path)
pretrained_word_embedding = torch.tensor(embeddings, dtype=torch.float, device=device)

# NewsEncoder 모델 초기화
news_encoder = NewsEncoder(config, pretrained_word_embedding)
news_encoder = news_encoder.to(device)
# news_encoder.eval()  # 추론 모드


# 데이터 불러오기
file_path = './Adressa_4w/history/history_tkg_behaviors.tsv'  
df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

# click_time을 datetime 형식으로 변환
df['click_time'] = pd.to_datetime(df['click_time'])

# 지정된 시간까지의 데이터로 필터링
end_time = pd.Timestamp('2017-02-05 08:00:01')
df = df[df['click_time'] <= end_time]

# 각 행마다 Period_Start 계산
def get_period_start(click_time, interval_minutes=1440, start_time=datetime.time(8, 0, 2)):
    """
    클릭 시간이 속하는 기간의 시작 시간을 계산합니다.
    
    Parameters:
    - click_time: pandas.Timestamp, 클릭 시간
    - interval_minutes: int, 기간의 길이(분 단위)
    - start_time: datetime.time, 하루 시작 시간
    
    Returns:
    - period_start: pandas.Timestamp, 해당 기간의 시작 시간
    """
    # 기준 날짜의 시작 시간
    base_start = datetime.datetime.combine(click_time.date(), start_time)
    
    if click_time >= base_start:
        delta = click_time - base_start
    else:
        base_start = base_start - datetime.timedelta(days=1)
        delta = click_time - base_start
    
    # 경과한 전체 기간 수
    periods = delta.total_seconds() // (interval_minutes * 60)
    
    # 현재 클릭 시간이 속하는 기간의 시작 시간
    period_start = base_start + datetime.timedelta(minutes=interval_minutes * periods)
    
    return period_start

df['Period_Start'] = df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=30))
# category 컬럼의 결측치(NaN)를 'No category|No subcategory'로 채우기
df['category'] = df['category'].fillna('No category|No subcategory')
# 'category'를 'Category'와 'Subcategory'로 분리
df[['Category', 'Subcategory']] = df['category'].str.split('|', n=1, expand=True)
# # 분리 후 남아있는 NaN 값을 'No category'와 'No subcategory'로 채우기
# df['Category'] = df['Category'].fillna('No category')
# df['Subcategory'] = df['Subcategory'].fillna('No subcategory')

# Period_Start로 그룹화 - 즉, 각 행이 snapshot이 됨
grouped = df.groupby('Period_Start')

# news_info 생성 - NewsEncoder 실행을 위해
# 뉴스별로 카테고리, 서브카테고리, 제목 단어 리스트를 추출
news_info = df.groupby('clicked_news').agg({
    'Category': 'first',
    'Subcategory': 'first',
    'title': 'first'
}).reset_index()



# 카테고리와 서브카테고리를 정수로 매핑
news_info['category_id'] = news_info['Category'].map(category2int.set_index('category')['int'])
news_info['subcategory_id'] = news_info['Subcategory'].map(category2int.set_index('category')['int'])

# 제목을 단어 리스트로 변환하고, 단어를 인덱스로 매핑
def tokenize_title(title):
    return title.split()  # 간단한 공백 기준 토크나이징

# 제목 단어 집합
news_info['title_words'] = news_info['title'].apply(tokenize_title)
# 제목 단어 index 집합
news_info['title_idx'] = news_info['title_words'].apply(
    lambda words: [word_to_idx[w] if w in word_to_idx else 0 for w in words]
)

# news_info에서 필요한 컬럼만 선택하여 news_info_df 생성
news_info_df = news_info[['clicked_news', 'category_id', 'subcategory_id', 'title_idx']].rename(
    columns={'clicked_news': 'news_id'}   # clicked_news를 news_id로 열 이름 변경
)

# print(news_info_df)

# newsId를 key로 하는 news_id_to_info 딕셔너리 생성
news_id_to_info = news_info_df.set_index('news_id').to_dict(orient='index') # orient='index': index를 key로, 그 행의 데이터를 dict형태의 value로 저장

# 6. 각 그룹별로 그래프 생성
snapshots = []
### 예외 cnt
no_category_cnts = []
no_news_cnts = []

# 전체 유저 ID와 뉴스 ID 추출
all_user_ids = df['history_user'].unique()
all_news_ids = df['clicked_news'].unique()
all_category_ids = df['category'].unique()
# 정렬
all_user_ids.sort()
# all_news_ids.tolist()
# all_news_ids.sort(key=lambda x: int(x[1:]))

# 유저, 카테고리 embeddings 생성
user_embeddings = torch.randn(len(all_user_ids), 128, device=device)  # 유저 임베딩 차원은 128
category_embeddings = torch.randn(len(all_category_ids), 128, device=device)  # 유저 임베딩 차원은 128
category2idx = {cat_str: idx for idx, cat_str in enumerate(all_category_ids)}

# 뉴스 인코더로 뉴스 embeddings 생성
news_vectors = []
# 예외 cnt; 0이어야 함
no_news_cnt = 0
for nid in tqdm(all_news_ids, desc="Making news embeddings"):
    if nid in news_id_to_info:
        info = news_id_to_info[nid]
        # category_id = info['category_id']
        # subcategory_id = info['subcategory_id']
        title_idx = torch.tensor(info['title_idx'], dtype=torch.long, device=device)   # 뉴스 제목 단어들의 idx
        # import torch.nn as nn
        # word_embedding = nn.Embedding.from_pretrained(
        #         pretrained_word_embedding, freeze=False, padding_idx=0)
        # title_vector = word_embedding(title_idx)
        # print(title_vector.shape)
        # exit()
        
        # 텐서 변환 (batch_size=1)
        # news_input = {
        #     'category': torch.tensor([category_id], dtype=torch.long, device=device),
        #     'subcategory': torch.tensor([subcategory_id], dtype=torch.long, device=device),
        #     'title': torch.tensor([title_idx], dtype=torch.long, device=device)
        # }
        nv = news_encoder(title_idx)  # shape: (1, num_filters)
        news_vectors.append(nv.squeeze(0)) 
    else:
        # 정보가 없는 경우 임의 벡터 할당
        news_vectors.append(torch.randn(config.num_filters))
        no_news_cnt += 1
        no_news_cnts.append(no_news_cnt)

# snapshots 생성
for period_start, group in tqdm(grouped, desc="Making snapshots"): 
    '''
    period_start: snapshot 시작 시점
    group: 각 snapshot의 데이터를 담은 dataframe
    '''
    # group의 유저 ID, 뉴스 ID, 카테고리 ID의 목록 생성
    user_ids = group['history_user'].unique()
    news_ids = group['clicked_news'].unique()
    category_ids = group['category'].unique()
    
    # 유저와 뉴스 ID를 인덱스로 매핑
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    news_id_map = {nid: idx for idx, nid in enumerate(news_ids)}
    
    num_users = len(user_ids)
    num_news = len(news_ids)
    
    # 그래프 생성
    # DGL의 heterograph를 사용하여 노드와 엣지 생성
    # 노드 타입: 'user', 'news'
    # 엣지 타입: ('user', 'clicked', 'news')
    
    # 엣지 리스트 생성 (user_idx, news_idx)
    edges_src = group['history_user'].map(user_id_map).values
    edges_dst = group['clicked_news'].map(news_id_map).values
    
    # 그래프 데이터 정의
    data_dict = {('user', 'clicked', 'news'): (edges_src, edges_dst)}
    
    # 모든 노드를 갖는 그래프 생성 후 edge 추가
    g = dgl.heterograph(
        {('user', 'clicked', 'news'): (edges_src, edges_dst)},
        num_nodes_dict={'user': len(all_user_ids), 'news': len(all_news_ids)}
    ).to(device)
    
    # 유저 노드에 임베딩 할당
    g.nodes['user'].data['feat'] = user_embeddings  # 유저 임베딩을 유저 노드의 임베딩으로 할당
    
    # 뉴스 임베딩 스택 후 뉴스 노드에 임베딩 할당
    news_embeddings = torch.stack(news_vectors)  # shape: (num_news, num_filters)
    g.nodes['news'].data['feat'] = news_embeddings  # 뉴스 노드에 NewsEncoder로 구한 뉴스 임베딩 할당
    
    # # 엣지에 카테고리 임베딩 할당
    # # 각 엣지에 해당하는 뉴스의 카테고리 가져오기
    # edge_categories = group['Category'].values
    # category_num = len(edge_categories)
    # edge_category_embeddings = []
    # # 예외 cnt
    # no_category_cnt = 0
    # for cat in edge_categories:
    #     if cat in word_to_idx:
    #         idx = word_to_idx[cat]
    #         emb = embeddings[idx]
    #     else:
    #         # 카테고리가 사전에 없으면 임의의 임베딩 할당 또는 처리
    #         emb = torch.randn(100)  # 또는 다른 방법으로 처리 가능
    #         no_category_cnt += 1
    #         no_category_cnts.append(no_category_cnt)
    #     edge_category_embeddings.append(emb)
    # edge_category_embeddings = torch.stack(edge_category_embeddings)
    
    # g.edges['clicked'].data['feat'] = edge_category_embeddings
    
    # edge에 임의의 임베딩 할당
    edge_categories = group['category'].values  # 클릭당 하나씩
    cat_idx_list = [category2idx[c] for c in edge_categories]  # 카테고리 문자열 -> 인덱스
    edge_cat_embeddings = category_embeddings[cat_idx_list]  # shape: (#edges, 128)
    g.edges['clicked'].data['feat'] = edge_cat_embeddings  # 카테고리 임베딩을 엣지의 임베딩으로 할당
    
    # **엣지 수와 클릭 수 비교**
    num_edges_in_g = g.number_of_edges(('user', 'clicked', 'news'))
    num_clicks_in_group = len(group)

    if num_edges_in_g != num_clicks_in_group:
        print(f"[Warning] Period_Start: {period_start} - Number of edges in graph ({num_edges_in_g}) does not match number of clicks in data ({num_clicks_in_group}).")
        exit()
    else:
        pass
        
    # 스냅샷 리스트에 그래프 추가
    snapshots.append(g)
    

# 이제 snapshots 리스트에는 각 기간별로 생성된 그래프가 저장되어 있습니다.
# snapshots[0], snapshots[1], ..., snapshots[34]

# 필요한 경우 그래프를 파일로 저장하거나 추가적인 처리를 할 수 있습니다.


# 8. 유저와 뉴스 ID 매핑 저장
# 전체 그룹에서 고유한 유저와 뉴스 ID를 추출하여 매핑 생성

# # 전체 유저 ID와 뉴스 ID 추출
# all_user_ids = df['history_user'].unique()
# all_news_ids = df['clicked_news'].unique()

# 유저 ID 매핑
user2int = {uid: idx for idx, uid in enumerate(all_user_ids)}
user_map_df = pd.DataFrame(list(user2int.items()), columns=['user_id', 'user_int'])
user_map_path = './Adressa_4w/history/user2int.tsv'
user_map_df.to_csv(user_map_path, sep='\t', index=False, encoding='utf-8')

# 뉴스 ID 매핑
news2int = {nid: idx for idx, nid in enumerate(all_news_ids)}
news_map_df = pd.DataFrame(list(news2int.items()), columns=['news_id', 'news_int'])
news_map_path = './Adressa_4w/history/news2int.tsv'
news_map_df.to_csv(news_map_path, sep='\t', index=False, encoding='utf-8')

# 9. 스냅샷 정보 저장
snapshot_info_list = []

for period_start, group in tqdm(grouped, desc="Processing Groups"):
    # 스냅샷 정보 저장
    snapshot_info_list.append({
        'Period_Start': period_start,
        'Unique_Users': len(group['history_user'].unique()),
        'Unique_News': len(group['clicked_news'].unique())
    })

# 스냅샷 정보 데이터프레임 생성
snapshot_info_df = pd.DataFrame(snapshot_info_list)
snapshot_info_df['Period_Start'] = snapshot_info_df['Period_Start'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 스냅샷 정보 저장 경로 지정
snapshot_info_path = './Adressa_4w/history/30m_lf_snapshots_info.tsv'
snapshot_info_df.to_csv(snapshot_info_path, sep='\t', index=False, encoding='utf-8')
