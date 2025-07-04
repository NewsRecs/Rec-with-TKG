import pandas as pd
import numpy as np
import dgl
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from dgl.data.utils import save_graphs

interval_minutes = [30, 720, 1440, 2160]

for interval_minute in interval_minutes:
    # (1) 30분 단위 구간 계산 함수 (코드1의 get_period_start와 동일/유사)
    def get_period_start(click_time, interval_minutes, start_time=datetime.time(0, 0, 0)):
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
    history_data_path = 'psj/Adressa_3w/datas/3w_behaviors.tsv'
    df = pd.read_csv(history_data_path, sep='\t', encoding='utf-8')
    df = df.dropna(subset=['clicked_news'])

    # click_time을 string에서 datetime으로 변환
    df['click_time'] = pd.to_datetime(df['click_time'])

    criteria_time1 = pd.Timestamp('2017-01-05 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-23 00:00:00')
    df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]


    # (2-1) 30분 단위 구간열(Period_Start) 생성
    df['Period_Start'] = df['click_time'].apply(lambda x: get_period_start(x, interval_minutes=interval_minute))
    # period_start -> time_idx 매핑(0부터 시작)
    unique_period_starts = df['Period_Start'].unique()
    time_dict = {ps: i for i, ps in enumerate(sorted(unique_period_starts))}
    df['time_idx'] = df['Period_Start'].map(time_dict)
    # print(df)
    # exit()
    # print(len(df[df['time_idx'] == 28]))
    # exit()


    # (2-2) news2int 적용
    news2int = pd.read_csv('./psj/Adressa_3w/datas/news2int.tsv', sep='\t')
    df['clicked_news'] = df['clicked_news'].astype(str).str.strip()
    news2int['news_id'] = news2int['news_id'].astype(str).str.strip()
    df = pd.merge(df, news2int, left_on='clicked_news', right_on='news_id', how='left')
    df.drop(columns=['news_id'], inplace=True)
    # print(df['clicked_news'].head(10))
    # print(news2int['news_id'].head(10))
    # print(df['clicked_news'].dtype, news2int['news_id'].dtype)
    # print(df['news_int'].describe())
    # print(df[df.isna()])
    # print(df[df['news_int'].isna()])  
    # print(df[df['clicked_news'].isna()])
    # print(df[df['category'].isna()]['category'])
    # exit()

    # (2-3) user2int 적용
    user2int_df = pd.read_csv(os.path.join('./psj/Adressa_3w/datas/', 'user2int.tsv'), sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    df['user_int'] = df['history_user'].map(user2int)

    # (2-4) category2int 적용
    category2int = pd.read_csv('psj/Adressa_3w/datas/category2int_nyheter_splitted.tsv', sep='\t')
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

    nyheter_mask = df['category'] == 'nyheter'
    df.loc[nyheter_mask, 'category'] = (
        df.loc[nyheter_mask, 'subcategory'] + '_nyheter'
    )
    df['category_int'] = df.apply(get_cat_int, axis=1)
    print(df[['category', 'subcategory', 'category_int']].head(10))

    # group 생성
    # times = df['Period_Start'].unique().tolist()
    # time_dict = {time: i for i, time in enumerate(times)}
    # df['time_idx'] = df['Period_Start'].map(time_dict)

    grouped = df.groupby('Period_Start')
    # group_user_int = []
    # for i in range(len(df['time_idx'].unique())):
    #     group_user_int.append(df[df['time_idx'] == i]['user_int'].values)

    # # 그룹별로 정렬된 순서대로 'time_idx'를 추출
    # time_idx_ordered = []
    # for group_key, group_df in grouped:
    #     # group_df는 해당 그룹의 원본 순서를 유지합니다.
    #     print(group_key, group_df['time_idx'])
    #     exit()
    #     time_idx_ordered.extend(group_df['time_idx'].tolist())

    # # (2-5) category_count 적용
    # category_count = pd.read_csv("combined_category_counts.tsv", sep="\t")
    # cat2count = category_count.set_index('category')['count'].to_dict()
    # def get_cat_count(row):
    #     if row['category'] == 'nyheter':
    #         # subcategory를 dict에서 찾되, 없다면 'No category'(또는 0)로 처리
    #         return cat2count.get(row['subcategory'], cat2count['No category'])
    #     else:
    #         return cat2count.get(row['category'], cat2count['No category'])

    # df['category_count'] = df.apply(get_cat_count, axis=1)


    # (2-6) df 인덱스를 0부터 차례대로 재정렬 -> 그래프 생성 시 forward edge와 df 행 1:1 매핑
    df = df.reset_index(drop=True)

    num_news_nodes = len(news2int) 
    num_user_nodes = len(user2int_df)

    # (3) 그래프 생성
    src_edges = df['news_int'].values + num_user_nodes      # (forward) news 노드
    dst_edges = df['user_int'].values      # (forward) user 노드
    cat_idx = df['category_int'].values    # 각 edge의 카테고리 인덱스
    edge_time_idx = df['time_idx'].values  # 각 edge(행)의 time_idx
    # cat_counts = df['category_count'].values    # 각 edge(행)의 카테고리 수


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
    # g.edata['cat_count'] = torch.tensor(cat_counts, dtype=torch.long)


    """
    reciprocal edges 추가하는 부분
    """
    ### reciprocal edges
    g.add_edges(dst_edges, src_edges)
    # 전체 엣지 데이터가 이 형식 (forward + reciprocal)이 되도록 해줌
    g.edata['cat_idx'][len(cat_idx):] = torch.tensor(cat_idx, dtype=torch.long)
    g.edata['time_idx'][len(edge_time_idx):] = torch.tensor(edge_time_idx, dtype=torch.long)
    # g.edata['cat_count'][len(cat_counts):] = torch.tensor(cat_counts, dtype=torch.long)

    # print(cat_idx)
    # print(edge_time_idx)
    # print(g)
    # exit()

    # (3-1) 전체 그래프 g를 저장 (원한다면)
    g_save_path = f"./psj/Adressa_3w/datas/total_graph_full_reciprocal_{interval_minute}m.bin"
    save_graphs(g_save_path, [g])

    print(df['time_idx'].max())
    print(g.number_of_nodes())   # 42751
    print(f"Total graph g({g_save_path}) saved!")