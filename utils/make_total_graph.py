import pandas as pd
import numpy as np
import dgl
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from dgl.data.utils import save_graphs


"""
full_news_encoder.py를 사용하기 위해 category2int_pio로 변경함
"""

# exit()
# def time_split_graph(train_until):
#     Total_Graph = load_graphs(snapshot_save_path)[0][0]
#     time_mask = [time_ < train_until+1 for time_ in time_list_id]
#     total_edge_index = list(range(len(time_list_id)*2))
#     train_edge_index = torch.tensor(total_edge_index)[time_mask+time_mask]
#     Train_Graph_byedge = Total_Graph.edge_subgraph(train_edge_index, preserve_nodes = True)
#     Train_Graph_byedge.copy_from_parent()
#     splitted = []
#     for i in range(train_until+1):
#         graph_at_i = Total_Graph.edge_subgraph(np.where(Total_Graph.edata['time_idx'] == i)[0], preserve_nodes = True)
#         graph_at_i.copy_from_parent()
#         splitted.append(graph_at_i)
#     return Train_Graph_byedge, splitted

# print("finished!")



# (1) 30분 단위 구간 계산 함수 (코드1의 get_period_start와 동일/유사)
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


# (2) 데이터 로드
history_data_path = './psj/Adressa_4w/history/history_tkg_behaviors.tsv'
df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
df = df.dropna(subset=['clicked_news'])

# click_time이 string일 경우 datetime으로 변환
df['click_time'] = pd.to_datetime(df['click_time'])

# (2-1) 30분 단위 구간열(Period_Start) 생성
df['Period_Start'] = df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=30))
# period_start -> time_idx 매핑(0부터 시작)
unique_period_starts = df['Period_Start'].unique()
time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
df['time_idx'] = df['Period_Start'].map(time_dict)
# print(len(df[df['time_idx'] == 28]))
# exit()


# (2-2) news2int 적용
DIRECTORY = './psj/Adressa_4w'
i = 1
data_type = ['train', 'test']
for dt in data_type:
    data_folder = os.path.join(DIRECTORY, dt)
    news_file = os.path.join(data_folder, 'valid_tkg_behaviors.tsv')
        
    print(f'{len(os.listdir(data_folder))}     파일:{news_file}',end=' #')

    # TSV 파일 읽기
    i += 1
    globals()[f"data{i}"] = pd.read_csv(news_file, sep='\t', header=None)
    globals()[f"data{i}"].columns = ['click_user', 'click_time', 'history', 'clicked_news']
    
history_news = set(df['clicked_news'].unique().tolist())
data2_news = data2['clicked_news'].dropna().unique().tolist()
data3_news = data3['clicked_news'].dropna().unique().tolist()

invalid_rows_data2 = data2[
    ~data2['clicked_news'].astype(str).str.endswith('-1') |
    data2['clicked_news'].isna()
]
invalid_idx = invalid_rows_data2.index.tolist()

data2_news = [n[:-2] for n in data2_news]
data3_news = [n[:-2] for n in data3_news]
total_news = set(history_news) | set(data2_news) | set(data3_news)
total_news = list(total_news)
news2int = {news_id: idx for idx, news_id in enumerate(total_news)}
news2int = pd.DataFrame({
    'news_id': list(news2int.keys()),
    'news_int': list(news2int.values())
})
df = pd.merge(df, news2int[['news_id', 'news_int']], 
              left_on='clicked_news', right_on='news_id', how='left')
df.drop(columns=['news_id'], inplace=True)

# (2-3) user2int 적용
users = df['history_user'].unique()
user2int = {uid: i for i, uid in enumerate(users)}
df['user_int'] = df['history_user'].map(user2int)
all_user_ids = sorted([user2int[u] for u in users])


# (2-4) category2int 적용
df['category'] = df['category'].fillna('No category|No subcategory')
df[['Category', 'Subcategory']] = df['category'].str.split('|', n=1, expand=True)

category2int = pd.read_csv('./psj/Adressa_4w/history/category2int_pio.tsv', sep='\t')
# 필요시 category2int에 'No category' 추가
if 'No category' not in category2int['category'].values:
    new_row = pd.DataFrame([{'category': 'No category', 'int': 0}])
    category2int = pd.concat([new_row, category2int], ignore_index=True)

cat2int = category2int.set_index('category')['int'].to_dict()
df['category_int'] = df['Category'].map(cat2int).fillna(0)  # 혹시 매핑 안 되면 0으로 처리


# group 생성
# times = df['Period_Start'].unique().tolist()
# time_dict = {time: i for i, time in enumerate(times)}
# df['time_idx'] = df['Period_Start'].map(time_dict)
"""seed_list?"""
grouped = df.groupby('Period_Start')
group_user_int = []
for i in range(len(df['time_idx'].unique())):
    group_user_int.append(df[df['time_idx'] == i]['user_int'].values)

# # 그룹별로 정렬된 순서대로 'time_idx'를 추출
# time_idx_ordered = []
# for group_key, group_df in grouped:
#     # group_df는 해당 그룹의 원본 순서를 유지합니다.
#     print(group_key, group_df['time_idx'])
#     exit()
#     time_idx_ordered.extend(group_df['time_idx'].tolist())


# (2-5) df 인덱스를 0부터 차례대로 재정렬 -> 그래프 생성 시 forward edge와 df 행 1:1 매핑
df = df.reset_index(drop=True)

num_news_nodes = len(news2int) 
num_user_nodes = len(users)

# (3) 그래프 생성
src_edges = df['news_int'].values + num_user_nodes      # (forward) news 노드
dst_edges = df['user_int'].values      # (forward) user 노드
cat_idx = df['category_int'].values    # 각 edge의 카테고리 인덱스
edge_time_idx = df['time_idx'].values  # 각 edge(행)의 time_idx



# forward edges: ('user','clicked','news') = (dst_edges, src_edges)
# reverse edges: ('news','clicked_reverse','user') = (src_edges, dst_edges)
g = dgl.DGLGraph()
g.add_nodes(num_news_nodes + num_user_nodes)
g.add_edges(src_edges, dst_edges)


# # 노드 데이터 저장 (필요한 경우)
# g.nodes['user'].data['user_ids'] = torch.tensor(df['user_int'].unique(), dtype=torch.long, device=device)
# useless = np.full(3301, -1)
# g.nodes['news'].data['news_ids'] = torch.tensor(np.concatenate([df['news_int'].unique(), useless]), dtype=torch.long, device=device)

# 엣지별 cat_idx 저장
g.edata['cat_idx'] = torch.tensor(cat_idx, dtype=torch.long)
g.edata['time_idx'] = torch.tensor(edge_time_idx, dtype=torch.long)
# print(cat_idx)
# print(edge_time_idx)
# print(g)
# exit()

# (3-1) 전체 그래프 g를 저장 (원한다면)
g_save_path = "./psj/Adressa_4w/history/total_graph_full.bin"
save_graphs(g_save_path, [g])

print("Total graph g saved!")