# import dgl
from dgl.data.utils import load_graphs
# import pickle
# import torch
# from tqdm import tqdm
import numpy as np


def split_train_graph(history_week, graph_path):
    """
    1) df['Period_Start'] 기준으로 30분 단위 구간을 순회
    2) 각 구간 내에서 user_int를 500명씩 잘라서 edge_subgraph
    3) preserve_nodes=True로, 모든 노드는 유지하되 해당 edge만 있는 스냅샷을 만든다.
    
    Returns:
        splitted: list of DGLGraph (각 30분 구간 x 유저 chunk 별)
    """
    Total_Graph = load_graphs(graph_path)[0][0]
    
    # with open('./psj/Adressa_4w/history/group_users.pickle', 'rb') as f:
    #     seeds = pickle.load(f)
    
        
    splitted = []
    # seed_list = []
    for i in range(int(history_week*7*24*2)):
        # 해당 batch_user_ids가 클릭한 df 행만 추출
        # time_window_users = seeds[i]
        # batch_src = group[batch_mask]['news_int'].values
        # batch_dst = group[batch_mask]['user_int'].values
        
        # subgraph 생성 (모든 노드를 preserve)
        graph_at_i = Total_Graph.edge_subgraph(np.where(Total_Graph.edata['time_idx'] == i)[0], preserve_nodes = True)
        graph_at_i.copy_from_parent()
        splitted.append(graph_at_i)

    # seed_list.append(set(time_window_users))
    
    return Total_Graph, splitted   # 모든 유저, 뉴스 포함하는 total graph, 시간 별로 분할된 sub_g들