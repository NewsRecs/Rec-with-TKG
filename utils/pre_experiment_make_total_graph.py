import pandas as pd
import numpy as np
import dgl
import pickle
import torch
import os
import datetime
from tqdm import tqdm
from dgl.data.utils import save_graphs


# (0) 통합데이터 만들기
his_df = pd.read_csv('psj/Adressa_4w/history/history_tkg_behaviors.tsv', sep='\t', encoding='utf-8')
train_df = pd.read_csv('psj/Adressa_4w/train/valid_tkg_behaviors.tsv', sep='\t', encoding='utf-8')
test_df = pd.read_csv('psj/Adressa_4w/test/valid_tkg_behaviors.tsv', sep='\t', encoding='utf-8')

sub_his_df = his_df[['history_user', 'click_time', 'clicked_news']]
sub_train_df = train_df[['history_user', 'click_time', 'clicked_news']]
sub_test_df = test_df[['history_user', 'click_time', 'clicked_news']]
df = pd.concat([sub_his_df, sub_train_df, sub_test_df])

# click_time을 string에서 datetime으로 변환
df['click_time'] = pd.to_datetime(df['click_time'])


# (2-2) news2int 적용
news2int = pd.read_csv('./psj/Adressa_4w/history/news2int.tsv', sep='\t')
df['clicked_news'] = df['clicked_news'].astype(str).str.strip()
news2int['news_id'] = news2int['news_id'].astype(str).str.strip()
df = pd.merge(df, news2int, left_on='clicked_news', right_on='news_id', how='left')
df.drop(columns=['news_id'], inplace=True)


# (2-3) user2int 적용
user2int_df = pd.read_csv(os.path.join('./psj/Adressa_4w/history/', 'user2int.tsv'), sep='\t')
user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
df['user_int'] = df['history_user'].map(user2int)


# (2-6) df 인덱스를 0부터 차례대로 재정렬 -> 그래프 생성 시 forward edge와 df 행 1:1 매핑
df = df.reset_index(drop=True)

num_news_nodes = len(news2int) 
num_user_nodes = len(user2int_df)


# (3) 그래프 생성
src_edges = df['news_int'].values + num_user_nodes      # (forward) news 노드
dst_edges = df['user_int'].values      # (forward) user 노드

g = dgl.DGLGraph()
g.add_nodes(num_news_nodes + num_user_nodes)
g.add_edges(src_edges, dst_edges)

### reciprocal edges
g.add_edges(dst_edges, src_edges)


# (3-1) 전체 그래프 g를 저장 (원한다면)
g_save_path = f"./psj/Adressa_4w/datas/total_graph_pre_experiment.bin"
save_graphs(g_save_path, [g])

print(g.number_of_nodes())   # 42751
print("Total graph g saved!")