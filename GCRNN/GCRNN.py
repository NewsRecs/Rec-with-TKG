from torch import nn
import dgl
import torch
from tqdm import tqdm
# from utils import binary_search
import numpy as np
import gc
import random
from psj.nce_loss import NCELoss
"""
해야 할 일들:
1. GCN 업데이트 처리 (완)
2. GRNN 유저 id로 각 g에 존재하는 유저들 embedding만 호출 (완)
3. 뉴스 인코더 추가 (완)
"""
from psj.full_news_encoder import NewsEncoder
from psj.config import Config

torch.cuda.set_device(0)
random_seed = 1024
random.seed(random_seed)
torch.manual_seed(random_seed)

# ------------------------------------------------------
# GCN 클래스 정의
# ------------------------------------------------------
class GCRNN(nn.Module):
    """
    News Encoder
    뉴스 인코더도 학습이 되려면 여기에 추가해줘야 함!
    
    GCN
    유저/뉴스 노드 모두에 대해 다음을 수행:
    1) 1-hop 이웃 노드들의 임베딩과 엣지 임베딩을 element-wise product
    2) 결과를 평균처리 해준 뒤, 자기 임베딩과 더해주기 (residual-like)
    3) batch 단위(500개)로 나눠서 처리 
    
    GRNN
    유저 노드만 수행:
    1) LSTM 정의
    2) LSTM 수행
    
    Loss function
    1) 각 뉴스에 대한 negative samples 불러오기
    2) NLL loss 정의
    3) NLL loss 학습
    
    forward
    1) 그래프 정보 가져오기
    2) GCN 수행
    3) GRNN 수행
    4) Loss 계산
    5) Backpropagation
    """
    def __init__(self, all_news_ids, news_id_to_info, user_num, cat_num, news_num, pretrained_word_embedding=None, emb_dim=100, batch_size=500, snapshots_num=1680):
        """
        학습 대상
        1. user embeddings
        2. category (edge) embeddings
        3. GRNN parameters
        4. NewsEncoder parameters
        *** GCN은 따로 파라미터가 존재하지 않음
        *** g의 유저, 뉴스 노드 idx는 user2int와 news2int의 순서대로 만들어짐 (edge도 마찬가지)
        """
        super(GCRNN, self).__init__()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.snapshots_num = snapshots_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.user_embedding_layer = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN user 초기값
        self.cat_embedding_layer = nn.Embedding(num_embeddings=cat_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # GCN relation 초기값
        # GRNN에 필요한 변수들
        self.user_num = user_num
        self.cat_num = cat_num
        self.news_num = news_num
        # self.prev_hn = nn.Embedding(num_embeddings=user_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # for hidden state in LSTM_GCN
        self.c0_embedding_layer = nn.Embedding(num_embeddings=user_num+news_num, embedding_dim=emb_dim, sparse = False).to(self.device)   # for cell state in LSTM_GCN
        self.user_RNN = nn.LSTMCell(emb_dim, emb_dim, bias = True).to(self.device)   # input dim, hn dim
        # News_Encoder에 필요한 정보들
        self.config = Config
        self.pretrained_word_embedding = pretrained_word_embedding
        self.news_encoder = NewsEncoder(self.config, self.pretrained_word_embedding).to(self.device)
        self.all_news_ids = all_news_ids   # news_int 순서대로 id 저장됨
        self.news_id_to_info = news_id_to_info
        
        
    def News_Encoder(self, news_ids):
        # print("Processing news embeddings \n")
        # 뉴스 embeddings 생성
        news_embeddings = torch.zeros((len(news_ids), self.emb_dim)).to(self.device)   # (전체 뉴스 수, 뉴스 embedding 차원)
        for i, nid in enumerate(news_ids):
            if nid in self.news_id_to_info:
                info = self.news_id_to_info[nid]
                title_idx_list = info['title_idx']
                title_tensor = torch.tensor(title_idx_list, dtype=torch.long, device=self.device)
                category_idx = info['category_idx']
                subcategory_idx = info['subcategory_idx']
                nv = self.news_encoder(title_tensor, category_idx, subcategory_idx)  # shape: (1, num_filters) 가 되도록 내부 처리
                news_embeddings[i] = nv.squeeze(0)
            else:
                news_embeddings[i] = torch.randn(self.emb_dim, device=self.device)
        
        return news_embeddings
    
    def message_func(self, edges):
        # 엣지 메시지는 뉴스 임베딩과 엣지 임베딩의 element-wise product
        return {'msg': edges.src['node_emb'] * self.rel_embedding[edges.data['cat_idx'].type(torch.LongTensor)]}

    def reduce_func(self, nodes):
        # 메시지를 상수 c로 나눈 뒤, 기존 임베딩과 더해줌 (residual)
        aggregated = nodes.mailbox['msg'].mean(1)
        return {'node_emb2': aggregated}   # new_feat -> feat: GCN 결과가 바로 update되도록 바꿈
                                                               # 위 주석처럼 할 경우 batch로 나눠서 GCN을 실행하기 때문에 오류 발생 가능성
                                                               # 다시 new_feat으로 생성하는 방식 사용        

    
    def seq_GCRNN_batch(self, g, sub_g, latest_train_time, seed_list, history_length):
        gcn_seed_per_time = []
        gcn_seed_1hopedge_per_time = []
        gcn_1hopneighbor_per_time = []
        gcn_seed_2hopedge_per_time = []
        future_needed_nodes = set()
        check_lifetime = np.zeros(self.user_num)
        for i in range(latest_train_time, -1, -1): # latest -> 0 미래부터 본다.
            
            # seed_list는 유저와 뉴스의 인덱스가 순서대로 저장돼 있도록 먼저 처리를 해주고, 이 작업을 수행할 수 있도록 해야 함
            # 이때 뉴스 인덱스는 유저 인덱스의 최대치만큼 다 더한 상태로 저장돼야 함 그래야 check_lifetime의 인덱스 형태로 들어갈 수 있으니까!
            check_lifetime[list(seed_list[i])] = history_length # seed_list: time별로 seed user가 들어있음

            # seed list에 들어있는 user들을 future needed nodes에 추가함(과거로 미래를 initialize하기 때문)
            future_needed_nodes = future_needed_nodes.union(torch.tensor(list(seed_list[i])).tolist())
            # 따라서 해당 seed들은 과거에도 계속 seed에 들어가게 되지만, 과거에 edge가 존재하는지 여부는 모름
            # 또한 history length가 full(100)인 현 상황에서는 초반(앞 시간대)구간의 경우 seed수가 계속 같을 수 있음
            # 또한, 데이터 특성상 과거로 갈 수록 edge가 적음

            # 1hop edges of seed at i
            hop1_u, hop1_v = sub_g[i].in_edges(v = list(future_needed_nodes), form = 'uv')
            # hop1_u, hop1_v = sub_g[i].in_edges(v = valid_future_nodes, form = 'uv', etype='clicked_reverse')   # u (news) -> v (user) 이다
            ### sub_g를 쓰는 것은 snapshot 시간을 조절하기 위함임 - 해당 시간에 존재하는 edges만 뽑아내려고!
            # hop1_neighbors_at_i, _, seed_edges_at_i = splitted_g[i].in_edges(v = list(future_needed_nodes), form = 'all')
            # node는 그대로 가져와지지만, splitted에서 추출한 edge id는 g의 edge id와 다를 수 있다.
            # 따라서 'edge id'가 아니라 'node id 쌍'로 edge를 기록해야 한다.

            # sample한 user들(entity)에 대해 in-edge들 찾는다.
            # hop 1 neighbors는 2layer를 위해 찾아둔것

            gcn_seed_per_time.append(list(future_needed_nodes)) # Seed
            # gcn에 seed로 사용되는 entity들이다. 사실 edge를 사용하기는 하지만..
            # 미래에서부터 쌓아왔기 때문에(사용하는 history length가 100이라서) 과거로 갈수록 꽤나 양이 커진다.
            
            gcn_seed_1hopedge_per_time.append((hop1_u, hop1_v))

            check_lifetime[check_lifetime>0] -= 1
            try:
                future_needed_nodes = future_needed_nodes - set(np.where(check_lifetime==0)[0]) # seed next
            except:
                pass
        
        self.rel_embedding = self.cat_embedding_layer(torch.tensor(range(self.cat_num)).to(self.device))
        # 초기화해줘야 밑에서 슬라이싱 가능
        g.ndata['node_emb'] = torch.zeros(g.number_of_nodes(), self.emb_dim, device=self.device)
        g.ndata['node_emb'][:self.user_num] = self.user_embedding_layer(torch.tensor(range(self.user_num)).to(self.device))
        # history_index = [nid[1:] for nid in self.all_news_ids]
        g.ndata['node_emb'][self.user_num:] = self.News_Encoder(self.all_news_ids)
        # print(g.device)
        # print()
        g.ndata['cx'] = self.c0_embedding_layer(torch.tensor(range(g.number_of_nodes())).to(self.device))
        entity_embs = []
        entity_index = []
        # register함수는 DGL 0.9이상에서는 없어졌다.
        g.register_message_func(self.message_func)
        g.register_reduce_func(self.reduce_func)
        for i in range(latest_train_time+1): # 0 -> latest
            # g_now = splitted_g[i]
            inverse = latest_train_time - i   # 1680-i
            # gcn_seed_per_time -> 미래부터 들어있다
            if len(gcn_seed_per_time[inverse]) > 0:   # inverse를 했을 때, gcn_seed_per_time이 애초에 시간 역순이라, 다시 시간 순서대로 보겠다는 의미
                changed = sorted(gcn_seed_per_time[inverse])   # 해당 time의 seed user 리스트를 user_id 순으로 정렬

                user_seed_ = changed   # g.nodes['user'].data['user_ids']
                user_prev_hn = g.ndata['node_emb'][user_seed_]#.to(self.device1)
                user_prev_cn = g.ndata['cx'][user_seed_]#.to(self.device1)

                edge_num = len(gcn_seed_1hopedge_per_time[inverse][0])
                g.send_and_recv(edges = gcn_seed_1hopedge_per_time[inverse])
                if edge_num > 0:
                    try:
                        g.ndata['node_emb'] = g.ndata['node_emb2'] + g.ndata['node_emb']
                        g.ndata.pop('node_emb2')
                    except:
                        pass
                user_input = g.ndata['node_emb'][user_seed_]

                user_hn, user_cn = self.user_RNN(user_input, (user_prev_hn, user_prev_cn))
                g.ndata['node_emb'][user_seed_] = user_hn
                g.ndata['cx'][user_seed_] = user_cn
                seed_emb = g.ndata['node_emb'][list(seed_list[i])]   # user_id 순으로 정렬되진 않음
                user_changed_in_global = torch.tensor(list(seed_list[i])) * latest_train_time + i   # user_id 순으로 index 크기가 정렬되게 함
                # 같은 유저라도 timestamp에 따라 고유한 index를 갖도록 해줌
                # 즉, 각 user_changed_in_global은 유저의 고유한 embedding 순서를 나타냄
                entity_embs.append(seed_emb)   # 아직 user_id 순으로 정렬되지 않은 embeddings
                entity_index.append(user_changed_in_global.type(torch.FloatTensor))

        entity_embs = torch.cat(entity_embs).to(self.device)   # (각 snapshot마다 존재하는 유저 수의 총 합, emb_dim)
                                                               # 우리 방법의 경우, 각 유저가 모두 마지막 seed_list(seed_list[1679])에만 존재
                                                               # -> 실제 크기는 (batch_size, emb_dim)

        # entity_index torch.cat 전: len=100(각 요소: [0], [0], ..., [500]), 후: shape=(500,)
        entity_index = torch.cat(entity_index)   # entity_index의 값들을 순서대로 정렬할 때, 이들의 indicies를 순서대로 반환
                                                 # shape: (각 snapshot마다 존재하는 user_num * snapshots_num, )
                                                 # 따라서 entity_index는 각 user의 idx를 갖고 있는 것과 같음

        ent_embs = entity_embs[entity_index.argsort()]   # 드디어 user_id 순으로 정렬됨
        # entity_index.argsort(): user indicies가 시간 순대로 나열됨
        # ent_embs: GCRNN으로 구한 user embedding이 시간 순대로 나열됨 - 각 유저의 마지막 값이 rnn의 최종 결과값 (마지막 hidden state)
        # ent_embs shape: (500, 128)
        return ent_embs
        
    
    def forward(self, user_batch, news_batch, category_batch, time_batch, g, sub_g, ns_idx, history_length=100): # user_batch, news_batch,  
        seed_list = []
        seed_entid = []
        train_t = []
        for time_list in time_batch:
            for time in time_list:
                train_t.append(time)
        
        seed_entid = []
        train_t = []
        latest_train_time = 1679
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                train_t.append(time)
                
        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(train_t), 
                                            sorted = True, return_inverse = True)
        # 각 rnn의 마지막 hidden state, 즉 rnn 결과로 얻은 각 유저의 embedding indicies를 저장한 tensor
        # index_for_ent_emb: unique 값들의 indicies를 모아둔 tensor
        # 즉, seed_entid의 train_click_num만큼 존재하는 user indicies
        
        user_embs = ent_embs[index_for_ent_emb]   # (train_click_num, 128)
                                                  # 이게 아마 정답) user_embs는 seed_entid 순으로 정렬된 user embeddings
                                                  # 즉, user_embs가 이미 내가 원하는 candidate_user_embs 형태!!!

        # 각 유저의 GCRNN 후 embeddings
        # *** userid 순으로 정렬됨 ***
        # train_click_num만큼 복사해줘야 함
        
        # u_time_embs = torch.cat([user_emb_0, user_embs]) # (N, emb_dim)   왜 합쳤니???????????? 난 어떻게 해야 하지...

        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch] # (N, emb_dim)
        # 원본: target_c_embs = self.ent_embedding_layer(torch.cat(comp_target_0).to(self.device0) + self.user_id_max + 1) # (N, emb_dim)
        # comp_target_0: 각 회사별 첫 번째 시간대의 news embedding의 indicies를 먼저 각각 하나의 list에 추가하고, 이후 각 회사별 첫 번째 외의 시간대 tensor가 순서대로 쌓여 있음
        
        """
        user_embs: (click 수, emb_dim)
        candidate_n_embs: (click 수, 9, emb_dim)
        내적 후 score: (click 수, 9)
        label: (click 수,) (예: 모든 값이 0, 즉 첫 번째 후보가 정답)
        """
        # target_n_embs = g.nodes['news'].data['node_emb'][news_batch]   # (target_news_num, emb_dim); target_news_num은 batch마다 다름
        # target_score = torch.matmul(user_embs, target_n_embs.transpose(1,0))   # (batch_size=500, target_news_num)
        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # g.nodes['news'].data['node_emb']는 news_int 순서대로 embedding 저장한 텐서; shape: (news_num, emb_dim)
        # candidate_n_embs: (train_click_num, (1 + 4), emb_dim); 1: target, 4: ns sample 수
        # ns_idx: (train_click_num, 5)
        candidate_user_embs = user_embs#[user_score_idx]   # user_score_idx: (train_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (train_click_num, 1, 128)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # candidate_n_embs: (train_click_num, emb_dim)*(train_click_num, 5, emb_dim)
        # candidate_score: (train_click_num, 5)
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   

        return loss
    
    
    def inference(self, user_batch, news_batch, time_batch, g, sub_g, ns_idx, history_length=100):
        seed_list = []
        seed_entid = []
        test_t = []
        
        for time_list in time_batch:
            for time in time_list:
                test_t.append(time)
        
        latest_train_time = 1679
        seed_entid = []
        test_t = []
        for i in range(latest_train_time+1):
            seed_list.append(set())
        for time_list, user in zip(time_batch, user_batch):
            for time in time_list:
                seed_list[time].add(user)  
                seed_entid.append(user)
                test_t.append(time)

        ent_embs = self.seq_GCRNN_batch(g, sub_g, latest_train_time, seed_list, history_length)   # (batch_size, emb_dim)
        _, index_for_ent_emb = torch.unique(torch.tensor(seed_entid) * latest_train_time + torch.tensor(test_t), 
                                            sorted = True, return_inverse = True)
        # (batch_size, )
        u_time_embs = ent_embs[index_for_ent_emb]

        candidate_n_embs = g.ndata['node_emb'][ns_idx + self.user_num]   
        # g.nodes['news'].data['node_emb']는 news_int 순서대로 embedding 저장한 텐서; shape: (news_num, emb_dim)
        # candidate_n_embs: (test_click_num, (1 + 20), emb_dim); 1: target, 20: ns sample 수
        # ns_idx: (test_click_num, 21)
        candidate_user_embs = u_time_embs#[user_score_idx]   # user_score_idx: (test_click_num, )
        candidate_user_embs = candidate_user_embs.unsqueeze(1)   # (test_click_num, 1, 128)            
        candidate_score = (candidate_user_embs * candidate_n_embs).sum(dim=-1)
        # candidate_n_embs: (test_click_num, emb_dim)*(test_click_num, 21, emb_dim)
        # candidate_score: (test_click_num, 21)        
        label_tensor = torch.zeros(len(candidate_score), dtype=torch.long, device=self.device)   # (train_click_num, )
        nce_loss = NCELoss()
        loss = nce_loss(candidate_score, label_tensor)   
        
        return candidate_score, loss
